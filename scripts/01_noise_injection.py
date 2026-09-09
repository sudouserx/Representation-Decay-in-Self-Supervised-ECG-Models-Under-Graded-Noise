#!/usr/bin/env python3
"""
Script 01 — Noise Injection
============================
Build noise bank from MIT-BIH NSTDB + physically motivated synthetic sources.
Save noise templates, injection manifest, and PSD validation plots.

Kaggle Inputs:  ptbxl-clean-processed, MIT-BIH NSTDB
Kaggle Output:  /kaggle/working/noisy_ecg_bank/ → publish as 'noisy-ecg-bank'
Est. Runtime:   ~2-4 h (CPU)
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

CLEAN_DIR = os.environ.get("CLEAN_DIR", "/kaggle/input/ptbxl-clean-processed")
NSTDB_DIR = os.environ.get("NSTDB_DIR", "/kaggle/input/mit-bih-noise-stress-test-database-1.0.0")
UTILS_DIR = os.environ.get("UTILS_DIR", "/kaggle/input/ecg-ssl-utils")
OUTPUT_DIR = "/kaggle/working/noisy_ecg_bank"

if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.artifact import write_artifact_snapshot
from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.noise.injection import compute_snr, inject_noise
from ecg_ssl_utils.noise.nstdb_loader import load_nstdb_noise
from ecg_ssl_utils.noise.synthetic_noise import generate_noise_bank


def save_psd_plots(noise_bank, fs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for noise_type, templates in noise_bank.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for t in templates[: min(5, len(templates))]:
            f, pxx = welch(t.mean(axis=0), fs=fs, nperseg=min(1024, t.shape[-1]))
            ax.semilogy(f, pxx, alpha=0.6)
        ax.axvspan(0.05, 45.0, color="tab:green", alpha=0.12, label="ECG band 0.05–45 Hz")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.set_title(f"PSD — {noise_type} (physically motivated synthetic / empirical replay)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"psd_{noise_type}.png"), dpi=120)
        plt.close(fig)


def main():
    cfg = get_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading MIT-BIH NSTDB noise records (hard-fail if missing)")
    print("=" * 60)
    nstdb_bank = load_nstdb_noise(
        NSTDB_DIR, n_leads_target=12, target_fs=500,
        n_templates=20, template_duration_s=10.0, seed=42,
    )
    for k, v in nstdb_bank.items():
        print(f"  {k}: {v.shape}")

    print("\n" + "=" * 60)
    print("STEP 2: Generating synthetic noise templates")
    print("=" * 60)
    synth_bank = generate_noise_bank(
        n_templates=20, duration_s=10.0, fs=500, n_leads=12, seed=42
    )
    for k, v in synth_bank.items():
        print(f"  {k}: {v.shape}")

    noise_bank = {**nstdb_bank, **synth_bank}

    print("\n" + "=" * 60)
    print("STEP 3: Saving noise templates + PSD plots")
    print("=" * 60)
    templates_dir = os.path.join(OUTPUT_DIR, "templates")
    os.makedirs(templates_dir, exist_ok=True)
    for noise_type, templates in noise_bank.items():
        np.save(os.path.join(templates_dir, f"{noise_type}.npy"), templates)
        print(f"  Saved {noise_type}: {templates.shape}")
    save_psd_plots(noise_bank, cfg.data.sampling_rate, os.path.join(OUTPUT_DIR, "psd"))

    print("\n" + "=" * 60)
    print("STEP 4: Building injection manifest (post-exclusion test IDs)")
    print("=" * 60)
    metadata = pd.read_parquet(os.path.join(CLEAN_DIR, "metadata.parquet"))
    test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)

    manifest_rows = []
    for noise_type in cfg.noise.noise_types_single:
        for snr_db in cfg.noise.snr_grid:
            for seed in cfg.noise.seeds:
                for _, row in test_meta.iterrows():
                    manifest_rows.append({
                        "record_id": row["ecg_id"],
                        "patient_id": row["patient_id"],
                        "noise_type": noise_type,
                        "snr_db": snr_db,
                        "seed": seed,
                        "is_mixed": False,
                        "mixture_name": "",
                        "mixture_types": "",
                        "mixture_weights": "",
                    })

    for mix_cfg in cfg.noise.mixed_noise_configs:
        for snr_db in cfg.noise.snr_grid:
            for seed in cfg.noise.seeds:
                for _, row in test_meta.iterrows():
                    manifest_rows.append({
                        "record_id": row["ecg_id"],
                        "patient_id": row["patient_id"],
                        "noise_type": mix_cfg["name"],
                        "snr_db": snr_db,
                        "seed": seed,
                        "is_mixed": True,
                        "mixture_name": mix_cfg["name"],
                        "mixture_types": ",".join(mix_cfg["types"]),
                        "mixture_weights": ",".join(map(str, mix_cfg["weights"])),
                    })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_parquet(os.path.join(OUTPUT_DIR, "noise_manifest.parquet"), index=False)
    n_conditions = manifest.groupby(["noise_type", "snr_db", "seed"]).ngroups
    print(f"  Total manifest rows: {len(manifest)}")
    print(f"  Unique conditions: {n_conditions}")
    print(f"  Test records: {len(test_meta)}")

    print("\n" + "=" * 60)
    print("STEP 5: SNR accuracy on raw (mV) test signals")
    print("=" * 60)
    raw_path = os.path.join(CLEAN_DIR, "signals_test_raw.npy")
    proc_path = os.path.join(CLEAN_DIR, "signals_test.npy")
    test_signals = np.load(raw_path if os.path.exists(raw_path) else proc_path)
    snr_convention = "raw_mV" if os.path.exists(raw_path) else "zscored"

    for noise_type in ["bw", "powerline", "electrode_pop"]:
        for target_snr in [24, 0, -6]:
            clean = test_signals[0]
            noisy = inject_noise(
                clean, noise_bank, noise_type, target_snr, seed=42, record_id=0,
            )
            actual_snr = compute_snr(clean, noisy)
            diff = abs(actual_snr - target_snr)
            status = "OK" if diff < 1.0 else "FAIL"
            print(f"  {status} {noise_type}@{target_snr}dB → actual: {actual_snr:.2f}dB (Δ={diff:.2f})")

    write_artifact_snapshot(
        OUTPUT_DIR,
        cfg,
        extra={
            "n_templates": 20,
            "template_duration_s": 10.0,
            "snr_convention": snr_convention,
            "n_conditions": int(n_conditions),
            "n_test_records": int(len(test_meta)),
        },
        filename="noise_config.json",
    )

    print("\n✓ Noise injection pipeline complete!")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Files: {os.listdir(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
