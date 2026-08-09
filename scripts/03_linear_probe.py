#!/usr/bin/env python3
"""
Script 03 — Linear Probe Training
==================================
Train linear probe on clean representations from all frozen SSL encoders,
and apply temperature scaling.

Kaggle Inputs:  ptbxl-clean-processed, all ssl-* model outputs
Kaggle Output:  /kaggle/working/linear-probes-all/
"""
import os, sys, json, glob
import numpy as np, torch

CLEAN_DIR = os.environ.get('CLEAN_DIR', '/kaggle/input/ptbxl-clean-processed')
UTILS_DIR = os.environ.get('UTILS_DIR', '/kaggle/input/ecg-ssl-utils')
OUTPUT_DIR = '/kaggle/working/linear-probes-all'
if UTILS_DIR not in sys.path: sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.models.vit_small_1d import ViTSmall1D
from ecg_ssl_utils.models.resnet18_1d import ResNet18_1D
from ecg_ssl_utils.probe.linear_probe import train_probe, LinearProbe
from ecg_ssl_utils.probe.calibration import temperature_scaling
from ecg_ssl_utils.eval.ece import expected_calibration_error


def load_encoder(model_dir, cfg, device):
    with open(os.path.join(model_dir, 'config.json')) as f:
        m_cfg = json.load(f)
    if m_cfg['backbone'] == 'vit_small_1d':
        encoder = ViTSmall1D(patch_size=cfg.backbone.patch_size, embed_dim=cfg.backbone.embed_dim,
                             depth=cfg.backbone.depth, num_heads=cfg.backbone.num_heads)
    else:
        encoder = ResNet18_1D(in_channels=12, output_dim=cfg.backbone.embed_dim)
    
    encoder.load_state_dict(torch.load(os.path.join(model_dir, 'encoder.pt'), map_location=device))
    encoder.eval().to(device)
    for p in encoder.parameters(): p.requires_grad = False
    return encoder, m_cfg

def extract_features(encoder, signals, batch_size=256, device='cuda'):
    features = []
    for i in range(0, len(signals), batch_size):
        batch = torch.tensor(signals[i:i+batch_size], dtype=torch.float32).to(device)
        with torch.no_grad():
            features.append(encoder(batch).cpu().numpy())
    return np.vstack(features)


def main():
    cfg = get_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # In Kaggle, directories might be under /kaggle/input/
    model_dirs = glob.glob('/kaggle/input/ssl-*') + glob.glob('/kaggle/working/ssl-*')
    # Filter to actual model dirs
    model_dirs = [d for d in model_dirs if os.path.exists(os.path.join(d, 'encoder.pt'))]
    # Unique dirs (prefer /kaggle/input if duplicates exist)
    unique_models = {os.path.basename(d): d for d in model_dirs}
    
    signals_train = np.load(os.path.join(CLEAN_DIR, 'signals_train.npy'))
    labels_train = np.load(os.path.join(CLEAN_DIR, 'labels_train.npy'))
    signals_val = np.load(os.path.join(CLEAN_DIR, 'signals_val.npy'))
    labels_val = np.load(os.path.join(CLEAN_DIR, 'labels_val.npy'))
    signals_test = np.load(os.path.join(CLEAN_DIR, 'signals_test.npy'))
    
    results = {}
    
    for m_name, m_dir in unique_models.items():
        print(f"\nProcessing {m_name}...")
        encoder, m_cfg = load_encoder(m_dir, cfg, device)
        
        # 1. Extract features
        train_feat = extract_features(encoder, signals_train, device=device)
        val_feat = extract_features(encoder, signals_val, device=device)
        test_feat = extract_features(encoder, signals_test, device=device)
        
        # Save clean features for CKA reference later
        m_out_dir = os.path.join(OUTPUT_DIR, m_name)
        os.makedirs(m_out_dir, exist_ok=True)
        np.save(os.path.join(m_out_dir, 'clean_train_repr.npy'), train_feat)
        np.save(os.path.join(m_out_dir, 'clean_test_repr.npy'), test_feat)
        
        # 2. Train probe
        probe, val_auroc = train_probe(
            train_feat, labels_train, val_feat, labels_val,
            in_dim=cfg.backbone.embed_dim, n_classes=labels_train.shape[1],
            epochs=cfg.probe.epochs, batch_size=cfg.probe.batch_size,
            lr=cfg.probe.lr, patience=cfg.probe.patience, device=device
        )
        
        # 3. Temperature scaling
        probe.eval()
        with torch.no_grad():
            val_logits = probe(torch.tensor(val_feat, dtype=torch.float32).to(device)).cpu().numpy()
        T = temperature_scaling(val_logits, labels_val)
        probe.temperature.data = torch.tensor([T], device=device)
        print(f"  Val AUROC: {val_auroc:.4f} | Optimal Temp: {T:.4f}")
        
        # Compute ECE on calibrated val
        with torch.no_grad():
            val_probs = torch.sigmoid(torch.tensor(val_logits) / T).numpy()
        val_ece = expected_calibration_error(labels_val, val_probs, n_bins=cfg.probe.n_calibration_bins)
        print(f"  Calibrated Val ECE: {val_ece:.4f}")
        
        # Save probe
        torch.save(probe.state_dict(), os.path.join(m_out_dir, 'probe.pt'))
        results[m_name] = {'val_auroc': val_auroc, 'val_ece': val_ece, 'temperature': float(T)}
        
    with open(os.path.join(OUTPUT_DIR, 'probe_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✓ Linear probe training complete!")

if __name__ == '__main__': main()
