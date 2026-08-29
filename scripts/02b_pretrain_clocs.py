#!/usr/bin/env python3
"""
Script 02b — Pretrain CLOCS
==============================
Temporal + Spatial + Patient contrastive learning on clean PTB-XL.

Kaggle Inputs:  ptbxl-clean-processed, ecg-ssl-utils
Kaggle Output:  /kaggle/working/ssl-clocs-vit-small/ → 'ssl-clocs-vit-small'
Est. Runtime:   ~8-10 h (GPU T4)
"""
import os, sys, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

CLEAN_DIR = os.environ.get('CLEAN_DIR', '/kaggle/input/ptbxl-clean-processed')
UTILS_DIR = os.environ.get('UTILS_DIR', '/kaggle/input/ecg-ssl-utils')
OUTPUT_DIR = '/kaggle/working/ssl-clocs-vit-small'

if UTILS_DIR not in sys.path: sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.models.vit_small_1d import ViTSmall1D
from ecg_ssl_utils.models.projectors import MLPProjector
from ecg_ssl_utils.ssl.clocs import CLOCSTrainer
import pandas as pd


# ── Collapse early-stopping thresholds ──
COLLAPSE_STD_THRESHOLD = 1e-4
COLLAPSE_SIM_THRESHOLD = 0.95
COLLAPSE_PATIENCE = 5


def main():
    cfg = get_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    signals = np.load(os.path.join(CLEAN_DIR, 'signals_train.npy'))
    meta = pd.read_parquet(os.path.join(CLEAN_DIR, 'metadata.parquet'))
    train_meta = meta[meta['split'] == 'train'].reset_index(drop=True)
    patient_ids = torch.tensor(train_meta['patient_id'].values, dtype=torch.long)
    print(f"Training signals: {signals.shape}")

    # CLOCS runs the encoder 6× per batch (2× temporal + 2× spatial + 2× patient),
    # requiring ~3× SimCLR's GPU memory. Override batch_size to avoid T4 OOM.
    clocs_batch_size = min(cfg.ssl_training.batch_size, 128)

    dataset = TensorDataset(torch.tensor(signals, dtype=torch.float32), patient_ids)
    loader = DataLoader(dataset, batch_size=clocs_batch_size,
                        shuffle=True, num_workers=cfg.ssl_training.num_workers,
                        pin_memory=True, drop_last=True)

    encoder = ViTSmall1D(patch_size=cfg.backbone.patch_size, embed_dim=cfg.backbone.embed_dim,
                         depth=cfg.backbone.depth, num_heads=cfg.backbone.num_heads)
    projector = MLPProjector(cfg.backbone.embed_dim, cfg.clocs.proj_hidden_dim,
                             cfg.clocs.proj_output_dim)

    params = list(encoder.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.ssl_training.lr,
                                  weight_decay=cfg.ssl_training.weight_decay)

    # ── LR scheduler with warmup ──
    warmup_epochs = cfg.ssl_training.warmup_epochs
    total_epochs = cfg.ssl_training.epochs
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=cfg.ssl_training.min_lr,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # Check for resume
    start_epoch = 0
    ckpt_path = os.path.join(OUTPUT_DIR, 'checkpoint.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt['encoder'])
        projector.load_state_dict(ckpt['projector'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resuming from epoch {start_epoch}")

    trainer = CLOCSTrainer(encoder, projector, optimizer, scheduler,
                           temperature=cfg.clocs.temperature,
                           lambda_temporal=cfg.clocs.lambda_temporal,
                           lambda_spatial=cfg.clocs.lambda_spatial,
                           lambda_patient=cfg.clocs.lambda_patient,
                           use_amp=cfg.ssl_training.use_amp, device=device,
                           grad_clip_norm=cfg.ssl_training.grad_clip_norm)

    log = open(os.path.join(OUTPUT_DIR, 'train_log.csv'),
               'a' if start_epoch > 0 else 'w')
    if start_epoch == 0:
        log.write('epoch,loss_total,loss_temporal,loss_spatial,loss_patient,'
                  'lr,time_s,embed_std,avg_cosine_sim\n')

    print(f"\nTraining CLOCS for {total_epochs} epochs")
    print(f"  Batch size: {clocs_batch_size} (reduced from {cfg.ssl_training.batch_size} — 6 encoder fwd passes/step)")
    print(f"  Temperature: {cfg.clocs.temperature}")
    print(f"  Warmup: {warmup_epochs} epochs")
    print(f"  Grad clip norm: {cfg.ssl_training.grad_clip_norm}")

    collapse_counter = 0

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        ep_losses = []
        for batch_sig, batch_pid in loader:
            losses = trainer.train_step(batch_sig, batch_pid)
            ep_losses.append(losses)
        scheduler.step()

        avg = {k: np.mean([l[k] for l in ep_losses]) for k in ep_losses[0]}
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        # Compute collapse metrics periodically
        if epoch % 10 == 0 or epoch < 5 or epoch == total_epochs - 1:
            metrics = trainer.compute_collapse_metrics()
            embed_std = metrics['embed_std']
            avg_cosine_sim = metrics['avg_cosine_sim']

            print(f"  Epoch {epoch:3d} | Total: {avg['total']:.4f} | "
                  f"T: {avg['temporal']:.4f} | S: {avg['spatial']:.4f} | "
                  f"P: {avg['patient']:.4f} | LR: {lr:.6f} | {elapsed:.0f}s | "
                  f"std: {embed_std:.4f} | cos_sim: {avg_cosine_sim:.4f}")

            # Collapse early-stopping
            if embed_std < COLLAPSE_STD_THRESHOLD or avg_cosine_sim > COLLAPSE_SIM_THRESHOLD:
                collapse_counter += 1
                print(f"  ⚠ COLLAPSE WARNING ({collapse_counter}/{COLLAPSE_PATIENCE}): "
                      f"embed_std={embed_std:.6f}, avg_cos_sim={avg_cosine_sim:.4f}")
                if collapse_counter >= COLLAPSE_PATIENCE:
                    print(f"\n✗ TRAINING HALTED: Representation collapse detected.")
                    break
            else:
                collapse_counter = 0
        else:
            embed_std = float('nan')
            avg_cosine_sim = float('nan')

        log.write(f"{epoch},{avg['total']:.6f},{avg['temporal']:.6f},"
                  f"{avg['spatial']:.6f},{avg['patient']:.6f},"
                  f"{lr:.8f},{elapsed:.1f},{embed_std:.6f},{avg_cosine_sim:.6f}\n")
        log.flush()

        if (epoch + 1) % cfg.ssl_training.checkpoint_every == 0:
            torch.save({'epoch': epoch, 'encoder': encoder.state_dict(),
                        'projector': projector.state_dict(),
                        'optimizer': optimizer.state_dict()}, ckpt_path)

    log.close()
    torch.save(encoder.state_dict(), os.path.join(OUTPUT_DIR, 'encoder.pt'))
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump({'paradigm': 'clocs', 'backbone': 'vit_small_1d',
                   'epochs': total_epochs, 'temperature': cfg.clocs.temperature,
                   'warmup_epochs': warmup_epochs,
                   'grad_clip_norm': cfg.ssl_training.grad_clip_norm}, f, indent=2)
    print(f"\n✓ CLOCS pretraining complete! Saved to {OUTPUT_DIR}")


if __name__ == '__main__': main()
