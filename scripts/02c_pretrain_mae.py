#!/usr/bin/env python3
"""
Script 02c — Pretrain MAE
==========================
Masked Autoencoder on clean PTB-XL, 75% mask ratio.
Kaggle Output: /kaggle/working/ssl-mae-vit-small/ → 'ssl-mae-vit-small'
"""
import os, sys, json, time
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset

CLEAN_DIR = os.environ.get('CLEAN_DIR', '/kaggle/input/ptbxl-clean-processed')
UTILS_DIR = os.environ.get('UTILS_DIR', '/kaggle/input/ecg-ssl-utils')
OUTPUT_DIR = '/kaggle/working/ssl-mae-vit-small'
if UTILS_DIR not in sys.path: sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.models.vit_small_1d import ViTSmall1D
from ecg_ssl_utils.ssl.mae import MAEDecoder, MAEModel, MAETrainer

def main():
    cfg = get_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    signals = np.load(os.path.join(CLEAN_DIR, 'signals_train.npy'))
    loader = DataLoader(TensorDataset(torch.tensor(signals, dtype=torch.float32)),
                        batch_size=cfg.ssl_training.batch_size, shuffle=True,
                        num_workers=cfg.ssl_training.num_workers, pin_memory=True, drop_last=True)

    encoder = ViTSmall1D(patch_size=cfg.backbone.patch_size, embed_dim=cfg.backbone.embed_dim,
                         depth=cfg.backbone.depth, num_heads=cfg.backbone.num_heads)
    decoder = MAEDecoder(num_patches=encoder.num_patches, enc_dim=cfg.backbone.embed_dim,
                         dec_dim=cfg.mae.decoder_embed_dim, depth=cfg.mae.decoder_depth,
                         nheads=cfg.mae.decoder_num_heads, patch_size=cfg.backbone.patch_size)
    mae = MAEModel(encoder, decoder, mask_ratio=cfg.mae.mask_ratio)

    optimizer = torch.optim.AdamW(mae.parameters(), lr=cfg.mae.lr,
                                  weight_decay=cfg.mae.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.ssl_training.epochs, cfg.ssl_training.min_lr)
    trainer = MAETrainer(mae, optimizer, scheduler, use_amp=cfg.ssl_training.use_amp, device=device)

    log = open(os.path.join(OUTPUT_DIR, 'train_log.csv'), 'w')
    log.write('epoch,loss,lr\n')
    for epoch in range(cfg.ssl_training.epochs):
        t0 = time.time()
        losses = [trainer.train_step(b[0]) for b in loader]
        scheduler.step()
        avg = np.mean(losses)
        lr = optimizer.param_groups[0]['lr']
        log.write(f'{epoch},{avg:.6f},{lr:.8f}\n'); log.flush()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {avg:.4f} | {time.time()-t0:.0f}s")
        if (epoch+1) % cfg.ssl_training.checkpoint_every == 0:
            torch.save({'epoch':epoch,'encoder':encoder.state_dict()},
                       os.path.join(OUTPUT_DIR,'checkpoint.pt'))
    log.close()
    torch.save(encoder.state_dict(), os.path.join(OUTPUT_DIR, 'encoder.pt'))
    with open(os.path.join(OUTPUT_DIR,'config.json'),'w') as f:
        json.dump({'paradigm':'mae','mask_ratio':cfg.mae.mask_ratio}, f)
    print("✓ MAE pretraining complete!")

if __name__ == '__main__': main()
