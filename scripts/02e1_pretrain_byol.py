#!/usr/bin/env python3
"""Script 02e1 — Pretrain BYOL (single-run sensitivity arm)."""
import os, sys, json, time
import numpy as np, torch
from torch.utils.data import TensorDataset

CLEAN_DIR = os.environ.get("CLEAN_DIR", "/kaggle/input/ptbxl-clean-processed")
UTILS_DIR = os.environ.get("UTILS_DIR", "/kaggle/input/ecg-ssl-utils")
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.artifact import write_artifact_snapshot
from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.models.projectors import BYOLProjector, MLPPredictor
from ecg_ssl_utils.models.vit_small_1d import ViTSmall1D
from ecg_ssl_utils.repro import make_deterministic_loader, parse_pretrain_seed, set_global_seed
from ecg_ssl_utils.ssl.augmentations import ECGAugmentation
from ecg_ssl_utils.ssl.byol import BYOLTrainer

HEARTBEAT_EVERY = 20


def _p(msg):
    print(msg, flush=True)


def main():
    cfg = get_config()
    seed = parse_pretrain_seed(42)
    set_global_seed(seed)
    OUTPUT_DIR = f"/kaggle/working/ssl-byol-vit-small-seed{seed}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _p(f"Device: {device} | seed: {seed}")
    _p(f"Output dir: {OUTPUT_DIR}")

    _p("Loading training signals...")
    signals = np.load(os.path.join(CLEAN_DIR, "signals_train.npy"))
    _p(f"Training signals: {signals.shape}")
    loader = make_deterministic_loader(
        TensorDataset(torch.tensor(signals, dtype=torch.float32)),
        cfg.ssl_training.batch_size, seed,
        workers=cfg.ssl_training.num_workers, drop_last=True,
    )
    encoder = ViTSmall1D(patch_size=cfg.backbone.patch_size, embed_dim=cfg.backbone.embed_dim,
                         depth=cfg.backbone.depth, num_heads=cfg.backbone.num_heads)
    projector = BYOLProjector(cfg.backbone.embed_dim, cfg.byol.projector_hidden_dim, cfg.byol.projector_output_dim)
    predictor = MLPPredictor(cfg.byol.projector_output_dim, cfg.byol.predictor_hidden_dim, cfg.byol.predictor_output_dim)
    params = list(encoder.parameters()) + list(projector.parameters()) + list(predictor.parameters())
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
        predictor.load_state_dict(ckpt["predictor"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        for _ in range(start_epoch):
            scheduler.step()
        _p(f"Resuming from epoch {start_epoch}")

    trainer = BYOLTrainer(
        encoder, projector, predictor, ECGAugmentation(), optimizer, scheduler,
        ema_start=cfg.byol.ema_momentum_start, ema_end=cfg.byol.ema_momentum_end,
        total_epochs=cfg.ssl_training.epochs, use_amp=cfg.ssl_training.use_amp, device=device,
        grad_clip_norm=cfg.ssl_training.grad_clip_norm,
        grad_accum_steps=cfg.ssl_training.grad_accum_steps,
    )
    log = open(os.path.join(OUTPUT_DIR, "train_log.csv"), "a" if start_epoch else "w")
    if start_epoch == 0:
        log.write("epoch,loss,lr,time_s,embed_std,avg_cosine_sim\n")
    log.flush()

    n_steps = len(loader)
    effective_batch = cfg.ssl_training.batch_size * cfg.ssl_training.grad_accum_steps
    _p(f"\nTraining BYOL for {cfg.ssl_training.epochs} epochs")
    _p(f"  Batch size: {cfg.ssl_training.batch_size} × {cfg.ssl_training.grad_accum_steps} accum = {effective_batch} effective")
    _p(f"  Steps/epoch: {n_steps} | AMP: {cfg.ssl_training.use_amp}")
    _p(f"  Warmup: {cfg.ssl_training.warmup_epochs} epochs")
    _p(f"  EMA momentum: {cfg.byol.ema_momentum_start} → {cfg.byol.ema_momentum_end}")
    _p(f"  Grad clip norm: {cfg.ssl_training.grad_clip_norm}")

    for epoch in range(start_epoch, cfg.ssl_training.epochs):
        t0 = time.time()
        _p(f"  Epoch {epoch:3d}/{cfg.ssl_training.epochs} starting ({n_steps} steps)")
        losses = []
        for i, b in enumerate(loader):
            loss = trainer.train_step(b[0], epoch, step_idx=i)
            losses.append(loss)
            if i == 0 or (i + 1) % HEARTBEAT_EVERY == 0 or (i + 1) == n_steps:
                _p(f"    step {i + 1}/{n_steps} | loss {loss:.4f}")
        scheduler.step()
        metrics = trainer.compute_collapse_metrics()
        avg = np.mean(losses)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        log.write(f"{epoch},{avg:.6f},{lr:.8f},{elapsed:.1f},"
                  f"{metrics['embed_std']:.6f},{metrics['avg_cosine_sim']:.6f}\n")
        log.flush()
        if epoch % 10 == 0 or epoch < 5 or epoch == cfg.ssl_training.epochs - 1:
            _p(f"  Epoch {epoch:3d} | Loss: {avg:.4f} | LR: {lr:.6f} | {elapsed:.0f}s | "
               f"std: {metrics['embed_std']:.4f} | cos_sim: {metrics['avg_cosine_sim']:.4f}")
        if (epoch + 1) % cfg.ssl_training.checkpoint_every == 0:
            torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                        "projector": projector.state_dict(), "predictor": predictor.state_dict(),
                        "optimizer": optimizer.state_dict()}, ckpt_path)
            _p(f"  Saved checkpoint at epoch {epoch}")
    log.close()
    torch.save(trainer.online_encoder.state_dict(), os.path.join(OUTPUT_DIR, "encoder.pt"))
    snap = {"paradigm": "byol", "backbone": "vit_small_1d", "seed": seed, "sensitivity_arm": True}
    json.dump(snap, open(os.path.join(OUTPUT_DIR, "config.json"), "w"), indent=2)
    write_artifact_snapshot(OUTPUT_DIR, cfg, seed=seed, extra=snap, filename="run_snapshot.json")
    _p(f"\n✓ BYOL sensitivity arm saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
