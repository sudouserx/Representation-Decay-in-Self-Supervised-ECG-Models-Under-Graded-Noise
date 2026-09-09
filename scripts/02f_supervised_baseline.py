#!/usr/bin/env python3
"""
Script 02f — Supervised ViT-S baseline
======================================
End-to-end supervised training on canonical superclass labels, same splits
as SSL (folds 1–8 train, 9 val). Saved like SSL encoders so 03–07 pick it up.
"""
import os, sys, json, time
import numpy as np, torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from sklearn.metrics import roc_auc_score

CLEAN_DIR = os.environ.get("CLEAN_DIR", "/kaggle/input/ptbxl-clean-processed")
UTILS_DIR = os.environ.get("UTILS_DIR", "/kaggle/input/ecg-ssl-utils")
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.artifact import write_artifact_snapshot
from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.models.vit_small_1d import ViTSmall1D
from ecg_ssl_utils.repro import make_deterministic_loader, parse_pretrain_seed, set_global_seed


def main():
    cfg = get_config()
    seed = parse_pretrain_seed(42)
    set_global_seed(seed)
    OUTPUT_DIR = f"/kaggle/working/ssl-supervised-vit-small-seed{seed}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_tr = np.load(os.path.join(CLEAN_DIR, "signals_train.npy"))
    y_tr = np.load(os.path.join(CLEAN_DIR, "superclass_labels_train.npy"))
    x_va = np.load(os.path.join(CLEAN_DIR, "signals_val.npy"))
    y_va = np.load(os.path.join(CLEAN_DIR, "superclass_labels_val.npy"))
    n_classes = y_tr.shape[1]

    loader = make_deterministic_loader(
        TensorDataset(torch.tensor(x_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)),
        cfg.ssl_training.batch_size, seed,
        workers=cfg.ssl_training.num_workers, drop_last=True,
    )
    encoder = ViTSmall1D(
        patch_size=cfg.backbone.patch_size, embed_dim=cfg.backbone.embed_dim,
        depth=cfg.backbone.depth, num_heads=cfg.backbone.num_heads,
    ).to(device)
    head = nn.Linear(cfg.backbone.embed_dim, n_classes).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.ssl_training.lr, weight_decay=cfg.ssl_training.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=cfg.ssl_training.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.ssl_training.epochs - cfg.ssl_training.warmup_epochs, eta_min=cfg.ssl_training.min_lr)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], [cfg.ssl_training.warmup_epochs])
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda") if cfg.ssl_training.use_amp else None

    start_epoch, best_auroc, wait, best_enc = 0, 0.0, 0, None
    ckpt_path = os.path.join(OUTPUT_DIR, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        head.load_state_dict(ckpt["head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_auroc = ckpt.get("best_auroc", 0.0)
        for _ in range(start_epoch):
            scheduler.step()

    log = open(os.path.join(OUTPUT_DIR, "train_log.csv"), "a" if start_epoch else "w")
    if start_epoch == 0:
        log.write("epoch,loss,val_auroc,lr,time_s\n")

    val_x = torch.tensor(x_va, dtype=torch.float32)
    for epoch in range(start_epoch, cfg.ssl_training.epochs):
        t0 = time.time()
        encoder.train(); head.train()
        losses = []
        optimizer.zero_grad()
        for i, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)
            with torch.amp.autocast("cuda", enabled=cfg.ssl_training.use_amp):
                loss = criterion(head(encoder(xb)), yb) / cfg.ssl_training.grad_accum_steps
            if scaler:
                scaler.scale(loss).backward()
                if (i + 1) % cfg.ssl_training.grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(params, cfg.ssl_training.grad_clip_norm)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            else:
                loss.backward()
                if (i + 1) % cfg.ssl_training.grad_accum_steps == 0:
                    nn.utils.clip_grad_norm_(params, cfg.ssl_training.grad_clip_norm)
                    optimizer.step(); optimizer.zero_grad()
            losses.append(loss.item() * cfg.ssl_training.grad_accum_steps)
        scheduler.step()

        encoder.eval(); head.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(val_x), 256):
                h = encoder(val_x[i:i + 256].to(device))
                probs.append(torch.sigmoid(head(h)).cpu().numpy())
        val_auroc = float(roc_auc_score(y_va, np.vstack(probs), average="macro"))
        log.write(f"{epoch},{np.mean(losses):.6f},{val_auroc:.6f},{optimizer.param_groups[0]['lr']:.8f},{time.time()-t0:.1f}\n")
        log.flush()
        print(f"  Supervised {epoch:3d} | loss {np.mean(losses):.4f} | val AUROC {val_auroc:.4f}")
        if val_auroc > best_auroc:
            best_auroc, wait = val_auroc, 0
            best_enc = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.probe.patience:
                print("Early stop on val AUROC")
                break
        if (epoch + 1) % cfg.ssl_training.checkpoint_every == 0:
            torch.save({"epoch": epoch, "encoder": encoder.state_dict(), "head": head.state_dict(),
                        "optimizer": optimizer.state_dict(), "best_auroc": best_auroc}, ckpt_path)

    log.close()
    if best_enc:
        encoder.load_state_dict(best_enc)
    torch.save(encoder.state_dict(), os.path.join(OUTPUT_DIR, "encoder.pt"))
    snap = {"paradigm": "supervised", "backbone": "vit_small_1d", "seed": seed, "sensitivity_arm": False,
            "val_auroc": best_auroc}
    json.dump(snap, open(os.path.join(OUTPUT_DIR, "config.json"), "w"), indent=2)
    write_artifact_snapshot(OUTPUT_DIR, cfg, seed=seed, extra=snap, filename="run_snapshot.json")
    print(f"✓ Supervised baseline saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
