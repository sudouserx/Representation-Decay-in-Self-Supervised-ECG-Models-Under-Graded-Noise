#!/usr/bin/env python3
"""
Script 00 — Data Preparation
=============================
Load PTB-XL, preprocess, create patient-level splits, save as clean dataset.

Kaggle Inputs:  PTB-XL dataset (e.g., 'khyeh0719/ptb-xl-dataset')
Kaggle Output:  /kaggle/working/ptbxl_clean/ → publish as 'ptbxl-clean-processed'
Est. Runtime:   ~30 min (CPU or GPU)
"""

import os, sys, json
import numpy as np
import pandas as pd

# ─── Path setup for Kaggle ─────────────────────────────────────
# Adjust these paths based on your Kaggle dataset structure
PTBXL_DIR = os.environ.get('PTBXL_DIR', '/kaggle/input/ptb-xl-dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3')
UTILS_DIR = os.environ.get('UTILS_DIR', '/kaggle/input/ecg-ssl-utils')
OUTPUT_DIR = '/kaggle/working/ptbxl_clean'

if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.data.ptbxl_loader import load_ptbxl, get_patient_split
from ecg_ssl_utils.data.preprocessing import (
    bandpass_filter, compute_norm_stats, normalize_signals, save_norm_stats
)
from ecg_ssl_utils.data.label_encoder import (
    encode_scp_labels, get_label_map, save_label_map,
    encode_superclass_labels, SUPERCLASS_NAMES,
)


def main():
    cfg = get_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load PTB-XL
    print("=" * 60)
    print("STEP 1: Loading PTB-XL dataset")
    print("=" * 60)
    signals, metadata = load_ptbxl(
        PTBXL_DIR,
        sampling_rate=cfg.data.sampling_rate,
        signal_length=cfg.data.signal_length,
    )
    print(f"  Signals shape: {signals.shape}")
    print(f"  Metadata shape: {metadata.shape}")

    # 2. Patient-level split
    print("\n" + "=" * 60)
    print("STEP 2: Creating patient-level split")
    print("=" * 60)
    splits = get_patient_split(
        metadata,
        train_folds=cfg.data.train_folds,
        val_folds=cfg.data.val_folds,
        test_folds=cfg.data.test_folds,
    )

    # Add split column
    metadata['split'] = 'train'
    metadata.loc[splits['val'], 'split'] = 'val'
    metadata.loc[splits['test'], 'split'] = 'test'

    # 3. Bandpass filter
    print("\n" + "=" * 60)
    print("STEP 3: Bandpass filtering (0.5–45 Hz)")
    print("=" * 60)
    signals = bandpass_filter(
        signals,
        low=cfg.data.bandpass_low,
        high=cfg.data.bandpass_high,
        fs=cfg.data.sampling_rate,
        order=cfg.data.filter_order,
    )
    print(f"  Filtered signals shape: {signals.shape}")

    # 4. Normalization stats from training set
    print("\n" + "=" * 60)
    print("STEP 4: Computing normalization stats (train set)")
    print("=" * 60)
    train_signals = signals[splits['train']]
    norm_stats = compute_norm_stats(train_signals)
    print(f"  Per-lead mean: {norm_stats['mean']}")
    print(f"  Per-lead std:  {norm_stats['std']}")

    # 5. Normalize all signals
    signals = normalize_signals(signals, norm_stats['mean'], norm_stats['std'])

    # 6. Encode labels
    print("\n" + "=" * 60)
    print("STEP 5: Encoding SCP labels")
    print("=" * 60)
    label_map = get_label_map(metadata)
    labels = encode_scp_labels(metadata, label_map)
    print(f"  Label map size: {len(label_map)} classes")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Avg labels per sample: {labels.sum(axis=1).mean():.2f}")

    # 6b. Encode superclass labels (5-class: NORM, MI, STTC, CD, HYP)
    print("\n" + "=" * 60)
    print("STEP 5b: Encoding superclass labels")
    print("=" * 60)
    scp_statements_path = os.path.join(PTBXL_DIR, 'scp_statements.csv')
    if os.path.exists(scp_statements_path):
        superclass_labels = encode_superclass_labels(metadata, scp_statements_path)
        print(f"  Superclass names: {SUPERCLASS_NAMES}")
        print(f"  Superclass labels shape: {superclass_labels.shape}")
        for i, name in enumerate(SUPERCLASS_NAMES):
            print(f"    {name}: {int(superclass_labels[:, i].sum())} positive")
    else:
        print(f"  WARNING: scp_statements.csv not found at {scp_statements_path}")
        print(f"  Superclass labels will not be generated.")
        superclass_labels = None

    # 7. Save everything
    print("\n" + "=" * 60)
    print("STEP 6: Saving artifacts")
    print("=" * 60)

    for split_name, idx in splits.items():
        np.save(os.path.join(OUTPUT_DIR, f'signals_{split_name}.npy'), signals[idx])
        np.save(os.path.join(OUTPUT_DIR, f'labels_{split_name}.npy'), labels[idx])
        if superclass_labels is not None:
            np.save(os.path.join(OUTPUT_DIR, f'superclass_labels_{split_name}.npy'),
                    superclass_labels[idx])
        print(f"  {split_name}: signals {signals[idx].shape}, labels {labels[idx].shape}"
              + (f", superclass {superclass_labels[idx].shape}" if superclass_labels is not None else ""))

    # Metadata
    metadata.to_parquet(os.path.join(OUTPUT_DIR, 'metadata.parquet'), index=False)
    save_norm_stats(norm_stats, os.path.join(OUTPUT_DIR, 'norm_stats.json'))
    save_label_map(label_map, os.path.join(OUTPUT_DIR, 'label_map.json'))

    # Config snapshot
    config_snapshot = {
        'sampling_rate': cfg.data.sampling_rate,
        'signal_length': cfg.data.signal_length,
        'n_leads': cfg.data.n_leads,
        'n_classes': len(label_map),
        'bandpass_low': cfg.data.bandpass_low,
        'bandpass_high': cfg.data.bandpass_high,
        'filter_order': cfg.data.filter_order,
        'train_folds': cfg.data.train_folds,
        'val_folds': cfg.data.val_folds,
        'test_folds': cfg.data.test_folds,
        'n_train': len(splits['train']),
        'n_val': len(splits['val']),
        'n_test': len(splits['test']),
    }
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(config_snapshot, f, indent=2)

    print("\n✓ Data preparation complete!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Files: {os.listdir(OUTPUT_DIR)}")


if __name__ == '__main__':
    main()
