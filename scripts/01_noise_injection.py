#!/usr/bin/env python3
"""
Script 01 — Noise Injection
============================
Build noise bank from MIT-BIH NSTDB + synthetic sources.
Save noise templates and injection manifest (not pre-materialized noisy signals).

Kaggle Inputs:  ptbxl-clean-processed, MIT-BIH NSTDB
Kaggle Output:  /kaggle/working/noisy_ecg_bank/ → publish as 'noisy-ecg-bank'
Est. Runtime:   ~2-4 h (CPU)
"""

import os, sys, json
import numpy as np
import pandas as pd
from itertools import product

CLEAN_DIR = os.environ.get('CLEAN_DIR', '/kaggle/input/ptbxl-clean-processed')
NSTDB_DIR = os.environ.get('NSTDB_DIR', '/kaggle/input/mit-bih-noise-stress-test-database-1.0.0')
UTILS_DIR = os.environ.get('UTILS_DIR', '/kaggle/input/ecg-ssl-utils')
OUTPUT_DIR = '/kaggle/working/noisy_ecg_bank'

if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.noise.nstdb_loader import load_nstdb_noise
from ecg_ssl_utils.noise.synthetic_noise import generate_noise_bank
from ecg_ssl_utils.noise.injection import inject_noise, compute_snr


def main():
    cfg = get_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load NSTDB noise records
    print("=" * 60)
    print("STEP 1: Loading MIT-BIH NSTDB noise records")
    print("=" * 60)
    nstdb_bank = load_nstdb_noise(
        NSTDB_DIR, n_leads_target=12, target_fs=500,
        n_templates=20, template_duration_s=10.0, seed=42
    )
    for k, v in nstdb_bank.items():
        print(f"  {k}: {v.shape}")

    # 2. Generate synthetic noise templates
    print("\n" + "=" * 60)
    print("STEP 2: Generating synthetic noise templates")
    print("=" * 60)
    synth_bank = generate_noise_bank(
        n_templates=20, duration_s=10.0, fs=500, n_leads=12, seed=42
    )
    for k, v in synth_bank.items():
        print(f"  {k}: {v.shape}")

    # 3. Merge into single noise bank
    noise_bank = {**nstdb_bank, **synth_bank}

    # 4. Save noise templates
    print("\n" + "=" * 60)
    print("STEP 3: Saving noise templates")
    print("=" * 60)
    templates_dir = os.path.join(OUTPUT_DIR, 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    for noise_type, templates in noise_bank.items():
        np.save(os.path.join(templates_dir, f'{noise_type}.npy'), templates)
        print(f"  Saved {noise_type}: {templates.shape}")

    # 5. Build injection manifest
    print("\n" + "=" * 60)
    print("STEP 4: Building injection manifest")
    print("=" * 60)

    # Load metadata to get test record IDs
    metadata = pd.read_parquet(os.path.join(CLEAN_DIR, 'metadata.parquet'))
    test_meta = metadata[metadata['split'] == 'test']

    manifest_rows = []

    # Single-noise conditions
    for noise_type in cfg.noise.noise_types_single:
        for snr_db in cfg.noise.snr_grid:
            for seed in cfg.noise.seeds:
                for _, row in test_meta.iterrows():
                    manifest_rows.append({
                        'record_id': row['ecg_id'],
                        'patient_id': row['patient_id'],
                        'noise_type': noise_type,
                        'snr_db': snr_db,
                        'seed': seed,
                        'is_mixed': False,
                        'mixture_name': '',
                        'mixture_types': '',
                        'mixture_weights': '',
                    })

    # Mixed-noise conditions
    for mix_cfg in cfg.noise.mixed_noise_configs:
        for snr_db in cfg.noise.snr_grid:
            for seed in cfg.noise.seeds:
                for _, row in test_meta.iterrows():
                    manifest_rows.append({
                        'record_id': row['ecg_id'],
                        'patient_id': row['patient_id'],
                        'noise_type': mix_cfg['name'],
                        'snr_db': snr_db,
                        'seed': seed,
                        'is_mixed': True,
                        'mixture_name': mix_cfg['name'],
                        'mixture_types': ','.join(mix_cfg['types']),
                        'mixture_weights': ','.join(map(str, mix_cfg['weights'])),
                    })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_parquet(os.path.join(OUTPUT_DIR, 'noise_manifest.parquet'), index=False)

    n_conditions = manifest.groupby(['noise_type', 'snr_db', 'seed']).ngroups
    print(f"  Total manifest rows: {len(manifest)}")
    print(f"  Unique conditions: {n_conditions}")
    print(f"  Noise types: {manifest['noise_type'].unique().tolist()}")

    # 6. Validate injection on a few samples
    print("\n" + "=" * 60)
    print("STEP 5: Validation — checking SNR accuracy")
    print("=" * 60)
    test_signals = np.load(os.path.join(CLEAN_DIR, 'signals_test.npy'))

    for noise_type in ['bw', 'powerline', 'electrode_pop']:
        for target_snr in [24, 0, -6]:
            clean = test_signals[0]
            noisy = inject_noise(
                clean, noise_bank, noise_type, target_snr,
                seed=42, record_id=0,
            )
            actual_snr = compute_snr(clean, noisy)
            diff = abs(actual_snr - target_snr)
            status = "✓" if diff < 1.0 else "✗"
            print(f"  {status} {noise_type}@{target_snr}dB → actual: {actual_snr:.2f}dB (Δ={diff:.2f})")

    # 7. Save injection function as module
    print("\n" + "=" * 60)
    print("STEP 6: Saving config")
    print("=" * 60)
    noise_config = {
        'snr_grid': cfg.noise.snr_grid,
        'seeds': cfg.noise.seeds,
        'noise_types_single': cfg.noise.noise_types_single,
        'mixed_noise_configs': cfg.noise.mixed_noise_configs,
        'n_templates': 20,
        'template_duration_s': 10.0,
    }
    with open(os.path.join(OUTPUT_DIR, 'noise_config.json'), 'w') as f:
        json.dump(noise_config, f, indent=2)

    print(f"\n✓ Noise injection pipeline complete!")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Files: {os.listdir(OUTPUT_DIR)}")


if __name__ == '__main__':
    main()
