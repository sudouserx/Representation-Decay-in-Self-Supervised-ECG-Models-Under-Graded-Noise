#!/usr/bin/env python3
"""
Script 02c — Pretrain MAE
==========================
Masked Autoencoder on clean PTB-XL, 75% mask ratio.
Kaggle Output: /kaggle/working/ssl-mae-vit-small/ → 'ssl-mae-vit-small'
"""
import os, sys, json, time
import numpy as np, torch
from torch.utils.data import TensorDataset

CLEAN_DIR = os.environ.get('CLEAN_DIR', '/kaggle/input/ptbxl-clean-processed')
UTILS_DIR = os.environ.get('UTILS_DIR', '/kaggle/input/ecg-ssl-utils')
if UTILS_DIR not in sys.path: sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.artifact import write_artifact_snapshot
from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.models.vit_small_1d import ViTSmall1D
from ecg_ssl_utils.repro import (
    checkpoint_runtime_state, make_deterministic_loader, parse_pretrain_seed,
    restore_runtime_state, set_global_seed,
)
from ecg_ssl_utils.ssl.mae import MAEDecoder, MAEModel, MAETrainer


def main():
    cfg = get_config()
    seed = parse_pretrain_seed(cfg.ssl_training.pretrain_seeds[0])
    set_global_seed(seed)
    OUTPUT_DIR = f'/kaggle/working/ssl-mae-vit-small-seed{seed}'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} | seed: {seed}")

    signals = np.load(os.path.join(CLEAN_DIR, 'signals_train.npy'))
    val_signals = np.load(os.path.join(CLEAN_DIR, 'signals_val.npy'), mmap_mode='r')
    val_batch = torch.tensor(
        np.asarray(val_signals[:cfg.ssl_training.batch_size]), dtype=torch.float32,
    )
    print(f"Training signals: {signals.shape}")

    loader = make_deterministic_loader(
        TensorDataset(torch.tensor(signals, dtype=torch.float32)),
        cfg.ssl_training.batch_size, seed,
        workers=cfg.ssl_training.num_workers, pin_memory=True, drop_last=True,
    )

    encoder = ViTSmall1D(patch_size=cfg.backbone.patch_size, embed_dim=cfg.backbone.embed_dim,
                         depth=cfg.backbone.depth, num_heads=cfg.backbone.num_heads)
    decoder = MAEDecoder(num_patches=encoder.num_patches, enc_dim=cfg.backbone.embed_dim,
                         dec_dim=cfg.mae.decoder_embed_dim, depth=cfg.mae.decoder_depth,
                         nheads=cfg.mae.decoder_num_heads, patch_size=cfg.backbone.patch_size)
    mae = MAEModel(encoder, decoder, mask_ratio=cfg.mae.mask_ratio)

    optimizer = torch.optim.AdamW(mae.parameters(), lr=cfg.mae.lr,
                                  weight_decay=cfg.mae.weight_decay)

    # ── LR scheduler with warmup (MAE uses its own warmup_epochs=20) ──
    warmup_epochs = cfg.mae.warmup_epochs
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
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt['encoder'])
        if 'decoder' in ckpt:
            decoder.load_state_dict(ckpt['decoder'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        if "rng_state" not in ckpt:
            for _ in range(start_epoch):
                scheduler.step()
        print(f"Resuming from epoch {start_epoch}")

    trainer = MAETrainer(mae, optimizer, scheduler, use_amp=cfg.ssl_training.use_amp,
                         device=device, grad_clip_norm=cfg.ssl_training.grad_clip_norm,
                         grad_accum_steps=cfg.ssl_training.grad_accum_steps)
    if start_epoch:
        restore_runtime_state(ckpt, scheduler, trainer.scaler, loader)

    log = open(os.path.join(OUTPUT_DIR, 'train_log.csv'),
               'a' if start_epoch > 0 else 'w')
    if start_epoch == 0:
        log.write('epoch,loss,val_reconstruction_loss,lr,time_s,embed_std,avg_cosine_sim\n')

    print(f"\nTraining MAE for {total_epochs} epochs")
    print(f"  Mask ratio: {cfg.mae.mask_ratio}")
    print(f"  LR: {cfg.mae.lr}, Warmup: {warmup_epochs} epochs")
    print(f"  Grad clip norm: {cfg.ssl_training.grad_clip_norm}")

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        losses = [
            trainer.train_step(b[0], step_idx=i, total_steps=len(loader))
            for i, b in enumerate(loader)
        ]
        scheduler.step()
        avg = np.mean(losses)
        metrics = trainer.compute_collapse_metrics()
        val_loss = trainer.validation_loss(val_batch, seed=seed)
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        log.write(f"{epoch},{avg:.6f},{val_loss:.6f},{lr:.8f},{elapsed:.1f},"
                  f"{metrics['embed_std']:.6f},{metrics['avg_cosine_sim']:.6f}\n")
        log.flush()
        if epoch % 10 == 0 or epoch < 5 or epoch == total_epochs - 1:
            print(f"  Epoch {epoch:3d} | Loss: {avg:.4f} | Val: {val_loss:.4f} | LR: {lr:.6f} | "
                  f"{elapsed:.0f}s | std: {metrics['embed_std']:.4f} | "
                  f"cos: {metrics['avg_cosine_sim']:.4f}")
        if (epoch + 1) % cfg.ssl_training.checkpoint_every == 0:
            torch.save({'epoch': epoch, 'encoder': encoder.state_dict(),
                        'decoder': decoder.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        **checkpoint_runtime_state(scheduler, trainer.scaler, loader)},
                       os.path.join(OUTPUT_DIR, 'checkpoint.pt'))

    log.close()
    torch.save(encoder.state_dict(), os.path.join(OUTPUT_DIR, 'encoder.pt'))
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump({'paradigm': 'mae', 'backbone': 'vit_small_1d',
                   'seed': seed, 'sensitivity_arm': False,
                   'mask_ratio': cfg.mae.mask_ratio,
                   'warmup_epochs': warmup_epochs,
                   'grad_accum_steps': cfg.ssl_training.grad_accum_steps,
                   'grad_clip_norm': cfg.ssl_training.grad_clip_norm}, f, indent=2)
    write_artifact_snapshot(OUTPUT_DIR, cfg, seed=seed, filename='run_snapshot.json')
    print(f"\n✓ MAE pretraining complete! Saved to {OUTPUT_DIR}")


if __name__ == '__main__': main()
