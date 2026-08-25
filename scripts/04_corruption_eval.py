#!/usr/bin/env python3
"""
Script 04 — Corruption Evaluation
=================================
Evaluate frozen encoders + linear probes on dynamically injected noisy ECGs.

Kaggle Inputs: ptbxl-clean-processed, noisy-ecg-bank, ssl-* models, linear-probes-all
Kaggle Output: /kaggle/working/corruption-eval-results/
"""
import os, sys, json
import numpy as np, pandas as pd, torch

CLEAN_DIR = os.environ.get('CLEAN_DIR', '/kaggle/input/ptbxl-clean-processed')
NOISE_DIR = os.environ.get('NOISE_DIR', '/kaggle/input/noisy-ecg-bank')
PROBE_DIR = os.environ.get('PROBE_DIR', '/kaggle/input/linear-probes-all')
UTILS_DIR = os.environ.get('UTILS_DIR', '/kaggle/input/ecg-ssl-utils')
OUTPUT_DIR = '/kaggle/working/corruption-eval-results'
if UTILS_DIR not in sys.path: sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.noise.injection import inject_noise
from ecg_ssl_utils.probe.linear_probe import LinearProbe
from ecg_ssl_utils.models.vit_small_1d import ViTSmall1D
from ecg_ssl_utils.models.resnet18_1d import ResNet18_1D

def get_model_dirs():
    import glob
    model_dirs = glob.glob('/kaggle/input/ssl-*') + glob.glob('/kaggle/working/ssl-*')
    model_dirs = [d for d in model_dirs if os.path.exists(os.path.join(d, 'encoder.pt'))]
    return {os.path.basename(d): d for d in model_dirs}

def load_noise_bank():
    bank = {}
    templates_dir = os.path.join(NOISE_DIR, 'templates')
    for f in os.listdir(templates_dir):
        if f.endswith('.npy'):
            bank[f[:-4]] = np.load(os.path.join(templates_dir, f))
    return bank

def load_pipeline(m_name, m_dir, cfg, device):
    with open(os.path.join(m_dir, 'config.json')) as f:
        m_cfg = json.load(f)
    if m_cfg['backbone'] == 'vit_small_1d':
        encoder = ViTSmall1D(patch_size=cfg.backbone.patch_size, embed_dim=cfg.backbone.embed_dim,
                             depth=cfg.backbone.depth, num_heads=cfg.backbone.num_heads)
    else:
        encoder = ResNet18_1D(in_channels=12, output_dim=cfg.backbone.embed_dim)
    
    encoder.load_state_dict(torch.load(os.path.join(m_dir, 'encoder.pt'), map_location=device))
    encoder.eval().to(device)
    
    # Load probe
    probe_path = os.path.join(PROBE_DIR, m_name, 'probe.pt')
    probe = LinearProbe(cfg.backbone.embed_dim, cfg.data.n_classes)
    probe.load_state_dict(torch.load(probe_path, map_location=device))
    probe.eval().to(device)
    
    return encoder, probe

def process_condition(clean_signals, manifest_subset, noise_bank, encoder, probe, device, batch_size=256):
    N = len(clean_signals)
    reps, probs = [], []
    
    for i in range(0, N, batch_size):
        batch_clean = clean_signals[i:i+batch_size]
        batch_manifest = manifest_subset.iloc[i:i+batch_size]
        
        batch_noisy = []
        for j, (_, row) in enumerate(batch_manifest.iterrows()):
            mixed_types = row['mixture_types'].split(',') if row['is_mixed'] and row['mixture_types'] else None
            mixed_weights = [float(w) for w in row['mixture_weights'].split(',')] if row['is_mixed'] and row['mixture_weights'] else None
            noisy = inject_noise(
                batch_clean[j], noise_bank, row['noise_type'], row['snr_db'], 
                row['seed'], row['record_id'], mixed_types, mixed_weights
            )
            batch_noisy.append(noisy)
            
        x = torch.tensor(np.stack(batch_noisy), dtype=torch.float32).to(device)
        with torch.no_grad():
            h = encoder(x)
            p = probe.predict_proba(h)
            reps.append(h.cpu().numpy().astype(np.float16))  # save space
            probs.append(p.cpu().numpy().astype(np.float32))
            
    return np.vstack(reps), np.vstack(probs)


def main():
    cfg = get_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    test_signals = np.load(os.path.join(CLEAN_DIR, 'signals_test.npy'))
    manifest = pd.read_parquet(os.path.join(NOISE_DIR, 'noise_manifest.parquet'))
    noise_bank = load_noise_bank()
    models = get_model_dirs()
    
    # Group by noise condition
    conditions = manifest.groupby(['noise_type', 'snr_db', 'seed'])
    
    for m_name, m_dir in models.items():
        print(f"\nEvaluating {m_name}")
        encoder, probe = load_pipeline(m_name, m_dir, cfg, device)
        m_out_dir = os.path.join(OUTPUT_DIR, m_name)
        os.makedirs(m_out_dir, exist_ok=True)
        
        for (ntype, snr, seed), group in conditions:
            print(f"  Condition: {ntype} @ {snr}dB (seed {seed})")
            
            # Ensure group matches test_signals order exactly
            assert len(group) == len(test_signals)
            
            reps, probs = process_condition(test_signals, group, noise_bank, encoder, probe, device)
            
            cond_dir = os.path.join(m_out_dir, f"{ntype}_{snr}db_{seed}")
            os.makedirs(cond_dir, exist_ok=True)
            # Use np.savez_compressed to save space
            np.savez_compressed(os.path.join(cond_dir, 'results.npz'), representations=reps, predictions=probs)

    print("\n✓ Corruption evaluation complete!")

if __name__ == '__main__': main()
