#!/usr/bin/env python3
"""
Script 02d — Pretrain I-JEPA–adapted
====================================
Latent-predictive pretraining. Target encoder sees target patches + CLS
(not the full sequence); loss is L2. Disclosed as I-JEPA–adapted.
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
from ecg_ssl_utils.ssl.jepa import JEPAPredictor, JEPAModel, JEPATrainer


def main():
    cfg = get_config()
    seed = parse_pretrain_seed(cfg.ssl_training.pretrain_seeds[0])
    set_global_seed(seed)
    OUTPUT_DIR = f'/kaggle/working/ssl-jepa-vit-small-seed{seed}'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device} | seed: {seed}")

    signals = np.load(os.path.join(CLEAN_DIR, 'signals_train.npy'))
    print(f"Training signals: {signals.shape}")

    loader = make_deterministic_loader(
        TensorDataset(torch.tensor(signals, dtype=torch.float32)),
        cfg.ssl_training.batch_size, seed,
        workers=cfg.ssl_training.num_workers, pin_memory=True, drop_last=True,
    )

    encoder = ViTSmall1D(patch_size=cfg.backbone.patch_size, embed_dim=cfg.backbone.embed_dim,
                         depth=cfg.backbone.depth, num_heads=cfg.backbone.num_heads)
    predictor = JEPAPredictor(dim=cfg.backbone.embed_dim, depth=cfg.jepa.predictor_depth,
                              nheads=cfg.jepa.predictor_num_heads)
    jepa = JEPAModel(
        encoder, predictor, ema_momentum=cfg.jepa.ema_momentum_start,
        num_target_blocks=cfg.jepa.num_target_blocks,
        target_block_size_range=tuple(cfg.jepa.target_block_size_range),
    )

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=cfg.ssl_training.lr, weight_decay=cfg.jepa.weight_decay)

    # ── LR scheduler with warmup (JEPA uses its own warmup_epochs=20) ──
    warmup_epochs = cfg.jepa.warmup_epochs
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
        jepa.context_encoder.load_state_dict(ckpt['encoder'])
        if 'predictor' in ckpt:
            jepa.predictor.load_state_dict(ckpt['predictor'])
        if 'target_encoder' in ckpt:
            jepa.target_encoder.load_state_dict(ckpt['target_encoder'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        if "rng_state" not in ckpt:
            for _ in range(start_epoch):
                scheduler.step()
        print(f"Resuming from epoch {start_epoch}")

    trainer = JEPATrainer(jepa, optimizer, scheduler,
                          ema_start=cfg.jepa.ema_momentum_start,
                          ema_end=cfg.jepa.ema_momentum_end,
                          total_epochs=total_epochs,
                          use_amp=cfg.ssl_training.use_amp, device=device,
                          grad_clip_norm=cfg.ssl_training.grad_clip_norm,
                          grad_accum_steps=cfg.ssl_training.grad_accum_steps)
    if start_epoch:
        restore_runtime_state(ckpt, scheduler, trainer.scaler, loader)

    log = open(os.path.join(OUTPUT_DIR, 'train_log.csv'),
               'a' if start_epoch > 0 else 'w')
    if start_epoch == 0:
        log.write('epoch,loss,lr,time_s,embed_std,avg_cosine_sim,var_below_1e-4,cov_abs_mean\n')

    print(f"\nTraining JEPA for {total_epochs} epochs")
    print(f"  EMA momentum: {cfg.jepa.ema_momentum_start} → {cfg.jepa.ema_momentum_end}")
    print(f"  Warmup: {warmup_epochs} epochs")
    print(f"  Grad clip norm: {cfg.ssl_training.grad_clip_norm}")

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        losses = [
            trainer.train_step(b[0], epoch, step_idx=i, total_steps=len(loader))
            for i, b in enumerate(loader)
        ]
        scheduler.step()
        avg = np.mean(losses)
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        metrics = {'embed_std': float('nan'), 'avg_cosine_sim': float('nan'),
                   'var_below_1e-4': 0, 'cov_abs_mean': float('nan')}
        if epoch % 10 == 0 or epoch < 5 or epoch == total_epochs - 1:
            metrics = trainer.compute_collapse_metrics()
            print(f"  Epoch {epoch:3d} | Loss: {avg:.4f} | LR: {lr:.6f} | {elapsed:.0f}s | "
                  f"std: {metrics['embed_std']:.4f} | cos: {metrics['avg_cosine_sim']:.4f}")
            if metrics['embed_std'] < 1e-4 or metrics['avg_cosine_sim'] > 0.95:
                print("  WARNING: JEPA collapse diagnostics triggered")
        log.write(f"{epoch},{avg:.6f},{lr:.8f},{elapsed:.1f},{metrics['embed_std']:.6f},"
                  f"{metrics['avg_cosine_sim']:.6f},{metrics['var_below_1e-4']},"
                  f"{metrics['cov_abs_mean']:.6f}\n"); log.flush()
        if (epoch + 1) % cfg.ssl_training.checkpoint_every == 0:
            torch.save({'epoch': epoch,
                        'encoder': jepa.context_encoder.state_dict(),
                        'predictor': jepa.predictor.state_dict(),
                        'target_encoder': jepa.target_encoder.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        **checkpoint_runtime_state(scheduler, trainer.scaler, loader)},
                       os.path.join(OUTPUT_DIR, 'checkpoint.pt'))

    log.close()
    torch.save(jepa.context_encoder.state_dict(), os.path.join(OUTPUT_DIR, 'encoder.pt'))
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump({'paradigm': 'jepa_adapted', 'backbone': 'vit_small_1d',
                   'seed': seed, 'sensitivity_arm': False,
                   'ema_start': cfg.jepa.ema_momentum_start,
                   'warmup_epochs': warmup_epochs,
                   'num_target_blocks': cfg.jepa.num_target_blocks,
                   'grad_clip_norm': cfg.ssl_training.grad_clip_norm}, f, indent=2)
    write_artifact_snapshot(OUTPUT_DIR, cfg, seed=seed, filename='run_snapshot.json')
    print(f"\n✓ JEPA pretraining complete! Saved to {OUTPUT_DIR}")


if __name__ == '__main__': main()
