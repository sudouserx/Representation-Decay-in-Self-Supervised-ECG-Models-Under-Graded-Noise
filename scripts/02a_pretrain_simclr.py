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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.ssl_training.epochs, eta_min=cfg.ssl_training.min_lr
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
        print(f"Resuming from epoch {start_epoch}")

    trainer = SimCLRTrainer(
        encoder, projector, augmentation, optimizer, scheduler,
        temperature=cfg.simclr.temperature, use_amp=cfg.ssl_training.use_amp,
        device=device,
    )

    # Training loop
    log_path = os.path.join(OUTPUT_DIR, 'train_log.csv')
    log_file = open(log_path, 'a' if start_epoch > 0 else 'w')
    if start_epoch == 0:
        log_file.write('epoch,loss,lr,time_s\n')

    print(f"\nTraining SimCLR for {cfg.ssl_training.epochs} epochs")
    for epoch in range(start_epoch, cfg.ssl_training.epochs):
        t0 = time.time()
        losses = []
        for (batch,) in loader:
            loss = trainer.train_step(batch)
            losses.append(loss)
        scheduler.step()

        avg_loss = np.mean(losses)
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        log_file.write(f'{epoch},{avg_loss:.6f},{lr:.8f},{elapsed:.1f}\n')
        log_file.flush()

        if epoch % 10 == 0 or epoch == cfg.ssl_training.epochs - 1:
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | LR: {lr:.6f} | Time: {elapsed:.1f}s")

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
                   'lr': cfg.ssl_training.lr, 'temperature': cfg.simclr.temperature}
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(config_snap, f, indent=2)

    print(f"\n✓ SimCLR pretraining complete! Saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
