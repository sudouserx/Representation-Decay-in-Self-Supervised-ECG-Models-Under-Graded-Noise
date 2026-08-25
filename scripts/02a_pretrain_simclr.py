#!/usr/bin/env python3
"""
Script 02a — Pretrain SimCLR
==============================
SimCLR with NT-Xent loss on clean PTB-XL, ViT-Small backbone.

Kaggle Inputs:  ptbxl-clean-processed, ecg-ssl-utils
Kaggle Output:  /kaggle/working/ssl-simclr-vit-small/ → publish as 'ssl-simclr-vit-small'
Est. Runtime:   ~8-10 h (GPU T4)
"""
import os, sys, json, time, csv
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

CLEAN_DIR = os.environ.get('CLEAN_DIR', '/kaggle/input/ptbxl-clean-processed')
UTILS_DIR = os.environ.get('UTILS_DIR', '/kaggle/input/ecg-ssl-utils')
OUTPUT_DIR = '/kaggle/working/ssl-simclr-vit-small'

if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.models.vit_small_1d import ViTSmall1D
from ecg_ssl_utils.models.projectors import MLPProjector
from ecg_ssl_utils.ssl.augmentations import ECGAugmentation
from ecg_ssl_utils.ssl.simclr import SimCLRTrainer


# ── Collapse early-stopping thresholds ──
COLLAPSE_STD_THRESHOLD = 1e-4       # embed_std below this → collapsed
COLLAPSE_SIM_THRESHOLD = 0.95       # avg_cosine_sim above this → collapsed
COLLAPSE_PATIENCE = 5               # consecutive bad checks before halt


def main():
    cfg = get_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load training data
    signals_train = np.load(os.path.join(CLEAN_DIR, 'signals_train.npy'))
    print(f"Training signals: {signals_train.shape}")

    dataset = TensorDataset(torch.tensor(signals_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=cfg.ssl_training.batch_size,
                        shuffle=True, num_workers=cfg.ssl_training.num_workers,
                        pin_memory=cfg.ssl_training.pin_memory, drop_last=True)

    # Model
    encoder = ViTSmall1D(
        patch_size=cfg.backbone.patch_size, embed_dim=cfg.backbone.embed_dim,
        depth=cfg.backbone.depth, num_heads=cfg.backbone.num_heads,
        mlp_ratio=cfg.backbone.mlp_ratio, drop_path_rate=cfg.backbone.drop_path_rate,
    )
    projector = MLPProjector(cfg.backbone.embed_dim, cfg.simclr.proj_hidden_dim,
                             cfg.simclr.proj_output_dim)
    augmentation = ECGAugmentation()

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
        # Advance scheduler to the correct epoch
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resuming from epoch {start_epoch}")

    grad_accum_steps = cfg.ssl_training.grad_accum_steps
    effective_batch = cfg.ssl_training.batch_size * grad_accum_steps

    trainer = SimCLRTrainer(
        encoder, projector, augmentation, optimizer, scheduler,
        temperature=cfg.simclr.temperature, use_amp=cfg.ssl_training.use_amp,
        device=device,
        grad_clip_norm=cfg.ssl_training.grad_clip_norm,
        grad_accum_steps=grad_accum_steps,
    )

    # Training loop
    log_path = os.path.join(OUTPUT_DIR, 'train_log.csv')
    log_file = open(log_path, 'a' if start_epoch > 0 else 'w')
    if start_epoch == 0:
        log_file.write('epoch,loss,lr,time_s,embed_std,avg_cosine_sim\n')

    # Expected initial loss for reference
    expected_init_loss = np.log(2 * cfg.ssl_training.batch_size - 1)

    print(f"\nTraining SimCLR for {total_epochs} epochs")
    print(f"  Batch size: {cfg.ssl_training.batch_size} × {grad_accum_steps} accum = {effective_batch} effective")
    print(f"  Temperature: {cfg.simclr.temperature}")
    print(f"  Warmup: {warmup_epochs} epochs")
    print(f"  Grad clip norm: {cfg.ssl_training.grad_clip_norm}")
    print(f"  Expected initial loss: ~{expected_init_loss:.2f}")

    collapse_counter = 0  # consecutive collapse detections

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        losses = []
        for step_idx, (batch,) in enumerate(loader):
            loss = trainer.train_step(batch, step_idx)
            losses.append(loss)
        scheduler.step()

        avg_loss = np.mean(losses)
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        # Compute collapse metrics every 10 epochs (or first 5 epochs for early detection)
        if epoch % 10 == 0 or epoch < 5 or epoch == total_epochs - 1:
            metrics = trainer.compute_collapse_metrics()
            embed_std = metrics['embed_std']
            avg_cosine_sim = metrics['avg_cosine_sim']

            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | LR: {lr:.6f} | "
                  f"Time: {elapsed:.1f}s | std: {embed_std:.4f} | cos_sim: {avg_cosine_sim:.4f}")

            # ── Collapse early-stopping check ──
            if embed_std < COLLAPSE_STD_THRESHOLD or avg_cosine_sim > COLLAPSE_SIM_THRESHOLD:
                collapse_counter += 1
                print(f"  ⚠ COLLAPSE WARNING ({collapse_counter}/{COLLAPSE_PATIENCE}): "
                      f"embed_std={embed_std:.6f}, avg_cos_sim={avg_cosine_sim:.4f}")
                if collapse_counter >= COLLAPSE_PATIENCE:
                    print(f"\n✗ TRAINING HALTED: Representation collapse detected for "
                          f"{COLLAPSE_PATIENCE} consecutive checks.")
                    print(f"  Last embed_std: {embed_std:.6f} (threshold: {COLLAPSE_STD_THRESHOLD})")
                    print(f"  Last avg_cosine_sim: {avg_cosine_sim:.4f} (threshold: {COLLAPSE_SIM_THRESHOLD})")
                    break
            else:
                collapse_counter = 0  # reset on healthy check
        else:
            embed_std = float('nan')
            avg_cosine_sim = float('nan')

        log_file.write(f'{epoch},{avg_loss:.6f},{lr:.8f},{elapsed:.1f},'
                       f'{embed_std:.6f},{avg_cosine_sim:.6f}\n')
        log_file.flush()

        if (epoch + 1) % cfg.ssl_training.checkpoint_every == 0:
            torch.save({
                'epoch': epoch, 'encoder': encoder.state_dict(),
                'projector': projector.state_dict(), 'optimizer': optimizer.state_dict(),
            }, ckpt_path)

    log_file.close()

    # Save final encoder (no projector)
    torch.save(encoder.state_dict(), os.path.join(OUTPUT_DIR, 'encoder.pt'))
    config_snap = {'paradigm': 'simclr', 'backbone': 'vit_small_1d',
                   'epochs': cfg.ssl_training.epochs, 'batch_size': cfg.ssl_training.batch_size,
                   'effective_batch_size': effective_batch,
                   'lr': cfg.ssl_training.lr, 'temperature': cfg.simclr.temperature,
                   'warmup_epochs': warmup_epochs,
                   'grad_accum_steps': grad_accum_steps,
                   'grad_clip_norm': cfg.ssl_training.grad_clip_norm}
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(config_snap, f, indent=2)

    print(f"\n✓ SimCLR pretraining complete! Saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
