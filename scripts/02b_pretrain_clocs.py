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


def main():
    cfg = get_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    signals = np.load(os.path.join(CLEAN_DIR, 'signals_train.npy'))
    meta = pd.read_parquet(os.path.join(CLEAN_DIR, 'metadata.parquet'))
    train_meta = meta[meta['split'] == 'train'].reset_index(drop=True)
    patient_ids = torch.tensor(train_meta['patient_id'].values, dtype=torch.long)

    dataset = TensorDataset(torch.tensor(signals, dtype=torch.float32), patient_ids)
    loader = DataLoader(dataset, batch_size=cfg.ssl_training.batch_size,
                        shuffle=True, num_workers=cfg.ssl_training.num_workers,
                        pin_memory=True, drop_last=True)

    encoder = ViTSmall1D(patch_size=cfg.backbone.patch_size, embed_dim=cfg.backbone.embed_dim,
                         depth=cfg.backbone.depth, num_heads=cfg.backbone.num_heads)
    projector = MLPProjector(cfg.backbone.embed_dim, cfg.clocs.proj_hidden_dim,
                             cfg.clocs.proj_output_dim)
    params = list(encoder.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.ssl_training.lr,
                                  weight_decay=cfg.ssl_training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.ssl_training.epochs, cfg.ssl_training.min_lr)

    trainer = CLOCSTrainer(encoder, projector, optimizer, scheduler,
                           temperature=cfg.clocs.temperature,
                           lambda_temporal=cfg.clocs.lambda_temporal,
                           lambda_spatial=cfg.clocs.lambda_spatial,
                           lambda_patient=cfg.clocs.lambda_patient,
                           use_amp=cfg.ssl_training.use_amp, device=device)

    log = open(os.path.join(OUTPUT_DIR, 'train_log.csv'), 'w')
    log.write('epoch,loss_total,loss_temporal,loss_spatial,loss_patient,lr\n')

    for epoch in range(cfg.ssl_training.epochs):
        t0 = time.time()
        ep_losses = []
        for batch_sig, batch_pid in loader:
            losses = trainer.train_step(batch_sig, batch_pid)
            ep_losses.append(losses)
        scheduler.step()
        avg = {k: np.mean([l[k] for l in ep_losses]) for k in ep_losses[0]}
        lr = optimizer.param_groups[0]['lr']
        log.write(f"{epoch},{avg['total']:.6f},{avg['temporal']:.6f},"
                  f"{avg['spatial']:.6f},{avg['patient']:.6f},{lr:.8f}\n")
        log.flush()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Total: {avg['total']:.4f} | T: {avg['temporal']:.4f} "
                  f"| S: {avg['spatial']:.4f} | P: {avg['patient']:.4f} | {time.time()-t0:.0f}s")
        if (epoch+1) % cfg.ssl_training.checkpoint_every == 0:
            torch.save({'epoch':epoch,'encoder':encoder.state_dict()},
                       os.path.join(OUTPUT_DIR,'checkpoint.pt'))
    log.close()
    torch.save(encoder.state_dict(), os.path.join(OUTPUT_DIR, 'encoder.pt'))
    with open(os.path.join(OUTPUT_DIR,'config.json'),'w') as f:
        json.dump({'paradigm':'clocs','backbone':'vit_small_1d'}, f)
    print(f"✓ CLOCS pretraining complete!")

if __name__ == '__main__': main()
