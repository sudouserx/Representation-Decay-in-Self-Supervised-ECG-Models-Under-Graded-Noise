#!/usr/bin/env python3
"""Script 02e3 — CLOCS-adapted + ResNet18 (single-run sensitivity arm)."""
import os, sys, json, time
import numpy as np, torch, pandas as pd
from torch.utils.data import TensorDataset

CLEAN_DIR = os.environ.get("CLEAN_DIR", "/kaggle/input/ptbxl-clean-processed")
UTILS_DIR = os.environ.get("UTILS_DIR", "/kaggle/input/ecg-ssl-utils")
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.artifact import write_artifact_snapshot
from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.models.projectors import MLPProjector
from ecg_ssl_utils.models.resnet18_1d import ResNet18_1D
from ecg_ssl_utils.repro import make_deterministic_loader, parse_pretrain_seed, set_global_seed
from ecg_ssl_utils.ssl.clocs import CLOCSTrainer


def main():
    cfg = get_config()
    seed = parse_pretrain_seed(42)
    set_global_seed(seed)
    OUTPUT_DIR = f"/kaggle/working/ssl-clocs-resnet18-seed{seed}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    signals = np.load(os.path.join(CLEAN_DIR, "signals_train.npy"))
    meta = pd.read_parquet(os.path.join(CLEAN_DIR, "metadata.parquet"))
    pids = torch.tensor(meta[meta["split"] == "train"].reset_index(drop=True)["patient_id"].values, dtype=torch.long)
    loader = make_deterministic_loader(
        TensorDataset(torch.tensor(signals, dtype=torch.float32), pids),
        cfg.ssl_training.batch_size, seed,
        workers=cfg.ssl_training.num_workers, drop_last=True,
    )
    encoder = ResNet18_1D(in_channels=12, output_dim=cfg.backbone.embed_dim)
    projector = MLPProjector(cfg.backbone.embed_dim, cfg.clocs.proj_hidden_dim, cfg.clocs.proj_output_dim)
    params = list(encoder.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.ssl_training.lr, weight_decay=cfg.ssl_training.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=cfg.ssl_training.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.ssl_training.epochs - cfg.ssl_training.warmup_epochs, eta_min=cfg.ssl_training.min_lr)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], [cfg.ssl_training.warmup_epochs])

    start_epoch = 0
    ckpt_path = os.path.join(OUTPUT_DIR, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        projector.load_state_dict(ckpt["projector"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        for _ in range(start_epoch):
            scheduler.step()

    trainer = CLOCSTrainer(
        encoder, projector, optimizer, scheduler,
        temperature=cfg.clocs.temperature,
        lambda_temporal=cfg.clocs.lambda_temporal,
        lambda_spatial=cfg.clocs.lambda_spatial,
        lambda_patient=cfg.clocs.lambda_patient,
        use_amp=cfg.ssl_training.use_amp, device=device,
        grad_clip_norm=cfg.ssl_training.grad_clip_norm,
        grad_accum_steps=cfg.ssl_training.grad_accum_steps,
    )
    log = open(os.path.join(OUTPUT_DIR, "train_log.csv"), "a" if start_epoch else "w")
    if start_epoch == 0:
        log.write("epoch,loss,lr,time_s,embed_std,avg_cosine_sim\n")
    for epoch in range(start_epoch, cfg.ssl_training.epochs):
        t0 = time.time()
        losses = [trainer.train_step(b[0], b[1], step_idx=i)["total"] for i, b in enumerate(loader)]
        scheduler.step()
        metrics = trainer.compute_collapse_metrics()
        log.write(f"{epoch},{np.mean(losses):.6f},{optimizer.param_groups[0]['lr']:.8f},"
                  f"{time.time()-t0:.1f},{metrics['embed_std']:.6f},{metrics['avg_cosine_sim']:.6f}\n")
        log.flush()
        if (epoch + 1) % cfg.ssl_training.checkpoint_every == 0:
            torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                        "projector": projector.state_dict(),
                        "optimizer": optimizer.state_dict()}, ckpt_path)
    log.close()
    torch.save(encoder.state_dict(), os.path.join(OUTPUT_DIR, "encoder.pt"))
    snap = {"paradigm": "clocs_adapted", "backbone": "resnet18_1d", "seed": seed, "sensitivity_arm": True}
    json.dump(snap, open(os.path.join(OUTPUT_DIR, "config.json"), "w"), indent=2)
    write_artifact_snapshot(OUTPUT_DIR, cfg, seed=seed, extra=snap, filename="run_snapshot.json")
    print(f"✓ CLOCS-ResNet18 sensitivity arm saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
