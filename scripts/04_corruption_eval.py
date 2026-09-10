#!/usr/bin/env python3
"""
Script 04 — Corruption Evaluation
=================================
Inject noise on raw (mV) test ECGs, apply the identical training front-end
(band-pass + train-only z-score), then evaluate frozen encoder + probe.

Kaggle Inputs: ptbxl-clean-processed, noisy-ecg-bank, ssl-* models, linear-probes-all
Kaggle Output: /kaggle/working/corruption-eval-results/
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.signal import welch

CLEAN_DIR = os.environ.get("CLEAN_DIR", "/kaggle/input/ptbxl-clean-processed")
NOISE_DIR = os.environ.get("NOISE_DIR", "/kaggle/input/noisy-ecg-bank")
PROBE_DIR = os.environ.get("PROBE_DIR", "/kaggle/input/linear-probes-all")
UTILS_DIR = os.environ.get("UTILS_DIR", "/kaggle/input/ecg-ssl-utils")
OUTPUT_DIR = "/kaggle/working/corruption-eval-results"
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.artifact import file_sha256, write_artifact_snapshot
from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.data.preprocessing import bandpass_filter, load_norm_stats, normalize_signals
from ecg_ssl_utils.models.resnet18_1d import ResNet18_1D
from ecg_ssl_utils.models.vit_small_1d import ViTSmall1D
from ecg_ssl_utils.noise.injection import compute_snr, inject_noise
from ecg_ssl_utils.probe.linear_probe import LinearProbe


def get_model_dirs():
    import glob
    model_dirs = glob.glob("/kaggle/input/ssl-*") + glob.glob("/kaggle/working/ssl-*")
    model_dirs = [d for d in model_dirs if os.path.exists(os.path.join(d, "encoder.pt"))]
    return {os.path.basename(d): d for d in model_dirs}


def load_noise_bank():
    bank = {}
    templates_dir = os.path.join(NOISE_DIR, "templates")
    for f in os.listdir(templates_dir):
        if f.endswith(".npy"):
            bank[f[:-4]] = np.load(os.path.join(templates_dir, f))
    return bank


def load_pipeline(m_name, m_dir, cfg, device, probe_n_classes=5):
    with open(os.path.join(m_dir, "config.json")) as f:
        m_cfg = json.load(f)
    backbone = m_cfg.get("backbone")
    if backbone is None:
        backbone = m_cfg.get("extra", {}).get("backbone", "vit_small_1d")
    if backbone == "vit_small_1d":
        encoder = ViTSmall1D(
            patch_size=cfg.backbone.patch_size, embed_dim=cfg.backbone.embed_dim,
            depth=cfg.backbone.depth, num_heads=cfg.backbone.num_heads,
        )
    else:
        encoder = ResNet18_1D(in_channels=12, output_dim=cfg.backbone.embed_dim)

    encoder.load_state_dict(torch.load(os.path.join(m_dir, "encoder.pt"), map_location=device))
    probe = LinearProbe(cfg.backbone.embed_dim, probe_n_classes)
    probe.load_state_dict(torch.load(os.path.join(PROBE_DIR, m_name, "probe.pt"), map_location=device))
    encoder.eval().to(device)
    probe.eval().to(device)
    return encoder, probe


def hash_parameters(model):
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.data.cpu().numpy().tobytes())
    return h.hexdigest()


def preprocess_noisy(noisy, cfg, norm_stats):
    filtered = bandpass_filter(
        noisy,
        low=cfg.data.bandpass_low,
        high=cfg.data.bandpass_high,
        fs=cfg.data.sampling_rate,
        order=cfg.data.filter_order,
    )
    normalized = normalize_signals(filtered, norm_stats["mean"], norm_stats["std"])
    return filtered, normalized


def process_condition(raw_signals, clean_filtered, manifest_subset, noise_bank, encoder, probe,
                      device, cfg, norm_stats, batch_size=256):
    N = len(raw_signals)
    reps, probs = [], []
    snr_pre, snr_post = [], []
    psd_pre, psd_post, psd_freq = [], [], None
    for i in range(0, N, batch_size):
        batch_raw = raw_signals[i:i + batch_size]
        batch_manifest = manifest_subset.iloc[i:i + batch_size]
        batch_proc = []
        for j, (_, row) in enumerate(batch_manifest.iterrows()):
            mixed_types = row["mixture_types"].split(",") if row["is_mixed"] and row["mixture_types"] else None
            mixed_weights = (
                [float(w) for w in row["mixture_weights"].split(",")]
                if row["is_mixed"] and row["mixture_weights"] else None
            )
            noisy = inject_noise(
                batch_raw[j], noise_bank, row["noise_type"], row["snr_db"],
                row["seed"], int(row["record_id"]), mixed_types, mixed_weights,
            )
            filtered, normalized = preprocess_noisy(noisy, cfg, norm_stats)
            batch_proc.append(normalized)
            snr_pre.append(compute_snr(batch_raw[j], noisy))
            snr_post.append(compute_snr(clean_filtered[i + j], filtered))
            if i + j < 32:
                psd_freq, p_pre = welch(
                    (noisy - batch_raw[j]).mean(axis=0),
                    fs=cfg.data.sampling_rate, nperseg=1024,
                )
                _, p_post = welch(
                    (filtered - clean_filtered[i + j]).mean(axis=0),
                    fs=cfg.data.sampling_rate, nperseg=1024,
                )
                psd_pre.append(p_pre)
                psd_post.append(p_post)
        x = torch.tensor(np.stack(batch_proc), dtype=torch.float32).to(device)
        with torch.no_grad():
            h = encoder(x)
            p = probe.predict_proba(h)
            reps.append(h.cpu().numpy().astype(np.float32))
            probs.append(p.cpu().numpy().astype(np.float32))
    return (
        np.vstack(reps), np.vstack(probs),
        np.asarray(snr_pre, dtype=np.float32),
        np.asarray(snr_post, dtype=np.float32),
        np.asarray(psd_freq, dtype=np.float32),
        np.mean(psd_pre, axis=0).astype(np.float32),
        np.mean(psd_post, axis=0).astype(np.float32),
    )


def main():
    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    raw_path = os.path.join(CLEAN_DIR, "signals_test_raw.npy")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"{raw_path} missing. Re-run script 00 to save unfiltered test signals."
        )
    test_signals = np.load(raw_path)
    clean_filtered = bandpass_filter(
        test_signals,
        low=cfg.data.bandpass_low,
        high=cfg.data.bandpass_high,
        fs=cfg.data.sampling_rate,
        order=cfg.data.filter_order,
    )
    norm_stats = load_norm_stats(os.path.join(CLEAN_DIR, "norm_stats.json"))
    meta = pd.read_parquet(os.path.join(CLEAN_DIR, "metadata.parquet"))
    test_meta = meta[meta["split"] == "test"].reset_index(drop=True)
    expected_ids = test_meta["ecg_id"].values

    manifest = pd.read_parquet(os.path.join(NOISE_DIR, "noise_manifest.parquet"))
    noise_bank = load_noise_bank()
    models = get_model_dirs()
    manifest_path = os.path.join(PROBE_DIR, "model_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError("model_manifest.json missing; rerun script 03")
    model_manifest = json.load(open(manifest_path))
    cohort_path = os.path.join(CLEAN_DIR, "cohort_definition.json")
    if file_sha256(cohort_path) != model_manifest.get("cohort_definition_sha256"):
        raise RuntimeError("clean-data cohort does not match the probe manifest")
    expected_models = {row["name"]: row for row in model_manifest["models"]}
    missing = sorted(set(expected_models) - set(models))
    if missing:
        raise RuntimeError(f"models from probe manifest are not mounted: {missing}")
    models = {name: models[name] for name in expected_models}
    for name, directory in models.items():
        actual = file_sha256(os.path.join(directory, "encoder.pt"))
        if actual != expected_models[name]["encoder_sha256"]:
            raise RuntimeError(f"encoder hash mismatch for {name}")
    conditions = manifest.groupby(["noise_type", "snr_db", "seed"])

    probe_metrics_path = os.path.join(PROBE_DIR, "probe_metrics.json")
    probe_metrics = json.load(open(probe_metrics_path)) if os.path.exists(probe_metrics_path) else {}

    for m_name, m_dir in models.items():
        print(f"\nEvaluating {m_name}")
        probe_n_classes = probe_metrics.get(m_name, {}).get("n_classes", cfg.data.n_superclasses)
        encoder, probe = load_pipeline(m_name, m_dir, cfg, device, probe_n_classes)
        m_out_dir = os.path.join(OUTPUT_DIR, m_name)
        os.makedirs(m_out_dir, exist_ok=True)

        for (ntype, snr, seed), group in conditions:
            group = group.reset_index(drop=True)
            print(f"  Condition: {ntype} @ {snr}dB (seed {seed})")
            assert len(group) == len(test_signals), (
                f"Manifest rows {len(group)} != test signals {len(test_signals)}"
            )
            assert np.array_equal(group["record_id"].values, expected_ids), (
                "record_id order does not match test metadata ecg_id order"
            )

            enc_hash_before = hash_parameters(encoder)
            probe_hash_before = hash_parameters(probe)
            reps, probs, snr_pre, snr_post, psd_hz, psd_pre, psd_post = process_condition(
                test_signals, clean_filtered, group, noise_bank, encoder, probe,
                device, cfg, norm_stats,
            )
            assert enc_hash_before == hash_parameters(encoder), "Encoder weights mutated"
            assert probe_hash_before == hash_parameters(probe), "Probe weights mutated"

            cond_dir = os.path.join(m_out_dir, f"{ntype}_{snr}db_{seed}")
            os.makedirs(cond_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(cond_dir, "results.npz"),
                representations=reps,
                predictions=probs,
                snr_pre_filter_db=snr_pre,
                snr_post_filter_db=snr_post,
                noise_psd_hz=psd_hz,
                noise_psd_pre_filter=psd_pre,
                noise_psd_post_filter=psd_post,
            )

    write_artifact_snapshot(
        OUTPUT_DIR, cfg,
        extra={"injection": "raw_then_identical_frontend", "repr_dtype": "float32"},
    )
    print("\n✓ Corruption evaluation complete!")


if __name__ == "__main__":
    main()
