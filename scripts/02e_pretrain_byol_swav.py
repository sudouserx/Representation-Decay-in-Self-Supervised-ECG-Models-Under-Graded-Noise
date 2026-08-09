#!/usr/bin/env python3
"""
Script 02e — Pretrain BYOL, SwAV & CLOCS-ResNet18 (Sensitivity Arms)
===================================================================
Pretrain additional SSL variants on clean PTB-XL as sensitivity comparisons.

Kaggle Output: 
/kaggle/working/ssl-byol-vit-small/
/kaggle/working/ssl-swav-vit-small/
/kaggle/working/ssl-clocs-resnet18/
"""
import os, sys, json, time
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset

CLEAN_DIR = os.environ.get('CLEAN_DIR', '/kaggle/input/ptbxl-clean-processed')
UTILS_DIR = os.environ.get('UTILS_DIR', '/kaggle/input/ecg-ssl-utils')
if UTILS_DIR not in sys.path: sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.models.vit_small_1d import ViTSmall1D
from ecg_ssl_utils.models.resnet18_1d import ResNet18_1D
from ecg_ssl_utils.models.projectors import BYOLProjector, MLPPredictor, MLPProjector, SwAVPrototypes
from ecg_ssl_utils.ssl.augmentations import ECGAugmentation
from ecg_ssl_utils.ssl.byol import BYOLTrainer
from ecg_ssl_utils.ssl.swav import SwAVTrainer
from ecg_ssl_utils.ssl.clocs import CLOCSTrainer
import pandas as pd


def train_byol(cfg, loader, device):
    out_dir = '/kaggle/working/ssl-byol-vit-small'
    os.makedirs(out_dir, exist_ok=True)
    
    encoder = ViTSmall1D(patch_size=cfg.backbone.patch_size, embed_dim=cfg.backbone.embed_dim,
                         depth=cfg.backbone.depth, num_heads=cfg.backbone.num_heads)
    projector = BYOLProjector(cfg.backbone.embed_dim, cfg.byol.projector_hidden_dim, cfg.byol.projector_output_dim)
    predictor = MLPPredictor(cfg.byol.projector_output_dim, cfg.byol.predictor_hidden_dim, cfg.byol.predictor_output_dim)
    
    params = list(encoder.parameters()) + list(projector.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.ssl_training.lr, weight_decay=cfg.ssl_training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.ssl_training.epochs, cfg.ssl_training.min_lr)
    
    trainer = BYOLTrainer(
        encoder, projector, predictor, ECGAugmentation(), optimizer, scheduler,
        ema_start=cfg.byol.ema_momentum_start, ema_end=cfg.byol.ema_momentum_end,
        total_epochs=cfg.ssl_training.epochs, use_amp=cfg.ssl_training.use_amp, device=device
    )
    
    log = open(os.path.join(out_dir, 'train_log.csv'), 'w')
    log.write('epoch,loss,lr\n')
    for epoch in range(cfg.ssl_training.epochs):
        losses = [trainer.train_step(b[0], epoch) for b in loader]
        scheduler.step()
        avg = np.mean(losses)
        lr = optimizer.param_groups[0]['lr']
        log.write(f'{epoch},{avg:.6f},{lr:.8f}\n')
        if epoch % 10 == 0: print(f"BYOL Epoch {epoch:3d} | Loss: {avg:.4f}")
    
    log.close()
    torch.save(trainer.online_encoder.state_dict(), os.path.join(out_dir, 'encoder.pt'))
    with open(os.path.join(out_dir,'config.json'),'w') as f:
        json.dump({'paradigm':'byol','backbone':'vit_small_1d'}, f)


def train_swav(cfg, loader, device):
    out_dir = '/kaggle/working/ssl-swav-vit-small'
    os.makedirs(out_dir, exist_ok=True)
    
    encoder = ViTSmall1D(patch_size=cfg.backbone.patch_size, embed_dim=cfg.backbone.embed_dim,
                         depth=cfg.backbone.depth, num_heads=cfg.backbone.num_heads)
    projector = MLPProjector(cfg.backbone.embed_dim, cfg.swav.proj_hidden_dim, cfg.swav.proj_output_dim)
    prototypes = SwAVPrototypes(cfg.swav.proj_output_dim, cfg.swav.num_prototypes)
    
    params = list(encoder.parameters()) + list(projector.parameters()) + list(prototypes.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.ssl_training.lr, weight_decay=cfg.ssl_training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.ssl_training.epochs, cfg.ssl_training.min_lr)
    
    trainer = SwAVTrainer(
        encoder, projector, prototypes, ECGAugmentation(), optimizer, scheduler,
        temperature=cfg.swav.temperature, sinkhorn_iters=cfg.swav.sinkhorn_iterations,
        sinkhorn_eps=cfg.swav.sinkhorn_epsilon, use_amp=cfg.ssl_training.use_amp, device=device
    )
    
    log = open(os.path.join(out_dir, 'train_log.csv'), 'w')
    log.write('epoch,loss,lr\n')
    for epoch in range(cfg.ssl_training.epochs):
        losses = [trainer.train_step(b[0]) for b in loader]
        scheduler.step()
        avg = np.mean(losses)
        lr = optimizer.param_groups[0]['lr']
        log.write(f'{epoch},{avg:.6f},{lr:.8f}\n')
        if epoch % 10 == 0: print(f"SwAV Epoch {epoch:3d} | Loss: {avg:.4f}")
    
    log.close()
    torch.save(encoder.state_dict(), os.path.join(out_dir, 'encoder.pt'))
    with open(os.path.join(out_dir,'config.json'),'w') as f:
        json.dump({'paradigm':'swav','backbone':'vit_small_1d'}, f)


def train_clocs_resnet(cfg, signals, patient_ids, device):
    out_dir = '/kaggle/working/ssl-clocs-resnet18'
    os.makedirs(out_dir, exist_ok=True)
    
    dataset = TensorDataset(torch.tensor(signals, dtype=torch.float32), patient_ids)
    loader = DataLoader(dataset, batch_size=cfg.ssl_training.batch_size, shuffle=True,
                        num_workers=cfg.ssl_training.num_workers, drop_last=True)
                        
    encoder = ResNet18_1D(in_channels=12, output_dim=cfg.backbone.embed_dim)
    projector = MLPProjector(cfg.backbone.embed_dim, cfg.clocs.proj_hidden_dim, cfg.clocs.proj_output_dim)
    
    params = list(encoder.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.ssl_training.lr, weight_decay=cfg.ssl_training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.ssl_training.epochs, cfg.ssl_training.min_lr)
    
    trainer = CLOCSTrainer(encoder, projector, optimizer, scheduler,
                           temperature=cfg.clocs.temperature, lambda_temporal=cfg.clocs.lambda_temporal,
                           lambda_spatial=cfg.clocs.lambda_spatial, lambda_patient=cfg.clocs.lambda_patient,
                           use_amp=cfg.ssl_training.use_amp, device=device)
                           
    log = open(os.path.join(out_dir, 'train_log.csv'), 'w')
    log.write('epoch,loss,lr\n')
    for epoch in range(cfg.ssl_training.epochs):
        ep_losses = [trainer.train_step(b[0], b[1])['total'] for b in loader]
        scheduler.step()
        avg = np.mean(ep_losses)
        lr = optimizer.param_groups[0]['lr']
        log.write(f'{epoch},{avg:.6f},{lr:.8f}\n')
        if epoch % 10 == 0: print(f"CLOCS-ResNet18 Epoch {epoch:3d} | Loss: {avg:.4f}")
    
    log.close()
    torch.save(encoder.state_dict(), os.path.join(out_dir, 'encoder.pt'))
    with open(os.path.join(out_dir,'config.json'),'w') as f:
        json.dump({'paradigm':'clocs','backbone':'resnet18_1d'}, f)


def main():
    cfg = get_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    signals = np.load(os.path.join(CLEAN_DIR, 'signals_train.npy'))
    loader = DataLoader(TensorDataset(torch.tensor(signals, dtype=torch.float32)),
                        batch_size=cfg.ssl_training.batch_size, shuffle=True,
                        num_workers=cfg.ssl_training.num_workers, drop_last=True)
    
    print("=== Training BYOL ===")
    train_byol(cfg, loader, device)
    
    print("=== Training SwAV ===")
    train_swav(cfg, loader, device)
    
    print("=== Training CLOCS-ResNet18 ===")
    meta = pd.read_parquet(os.path.join(CLEAN_DIR, 'metadata.parquet'))
    train_meta = meta[meta['split'] == 'train'].reset_index(drop=True)
    patient_ids = torch.tensor(train_meta['patient_id'].values, dtype=torch.long)
    train_clocs_resnet(cfg, signals, patient_ids, device)
    
    print("✓ All sensitivity arms trained!")

if __name__ == '__main__': main()
