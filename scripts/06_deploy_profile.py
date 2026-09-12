#!/usr/bin/env python3
"""
Script 06 — Server-proxy Deployment Profiling
=============================================
Export encoder+probe to ONNX, INT8-quantize with train-set calibration,
parity-gate vs FP32, and profile CPU ONNX Runtime on a real ECG tensor.

Kaggle Inputs: ssl-* models, linear-probes-all, ptbxl-clean-processed
Kaggle Output: /kaggle/working/deployment-profiles/
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from dataclasses import asdict

UTILS_DIR = os.environ.get("UTILS_DIR", "/kaggle/input/ecg-ssl-utils")
CLEAN_DIR = os.environ.get("CLEAN_DIR", "/kaggle/input/ptbxl-clean-processed")
PROBE_DIR = os.environ.get("PROBE_DIR", "/kaggle/input/linear-probes-all")
OUTPUT_DIR = "/kaggle/working/deployment-profiles"
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.artifact import discover_ssl_model_dirs, write_artifact_snapshot
from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.deploy.onnx_export import ClassifierWrapper, export_to_onnx
from ecg_ssl_utils.deploy.profiler import (
    ProviderUnavailableError,
    probe_usable_providers,
    profile_model,
)
from ecg_ssl_utils.deploy.quantization import quantization_parity, quantize_model
from ecg_ssl_utils.models.resnet18_1d import ResNet18_1D
from ecg_ssl_utils.models.vit_small_1d import ViTSmall1D
from ecg_ssl_utils.probe.linear_probe import LinearProbe


def _backbone_from_cfg(m_cfg):
    if "backbone" in m_cfg and isinstance(m_cfg["backbone"], str):
        return m_cfg["backbone"]
    extra = m_cfg.get("extra") or {}
    if "backbone" in extra:
        return extra["backbone"]
    nested = m_cfg.get("config", {}).get("backbone", {})
    if isinstance(nested, dict):
        return nested.get("arch", "vit_small_1d")
    return "vit_small_1d"


def load_wrapper(model_dir, m_name, cfg, n_classes):
    with open(os.path.join(model_dir, "config.json")) as f:
        m_cfg = json.load(f)
    backbone = _backbone_from_cfg(m_cfg)
    if backbone == "vit_small_1d":
        encoder = ViTSmall1D(
            patch_size=cfg.backbone.patch_size, embed_dim=cfg.backbone.embed_dim,
            depth=cfg.backbone.depth, num_heads=cfg.backbone.num_heads,
        )
    else:
        encoder = ResNet18_1D(in_channels=12, output_dim=cfg.backbone.embed_dim)
    encoder.load_state_dict(torch.load(os.path.join(model_dir, "encoder.pt"), map_location="cpu"))
    probe = LinearProbe(cfg.backbone.embed_dim, n_classes)
    probe_path = os.path.join(PROBE_DIR, m_name, "probe.pt")
    if os.path.exists(probe_path):
        probe.load_state_dict(torch.load(probe_path, map_location="cpu"))
    else:
        print(
            f"WARNING: missing probe weights at {probe_path}; "
            "exporting an untrained LinearProbe (results are not valid)."
        )
    encoder.eval()
    probe.eval()
    return ClassifierWrapper(encoder, probe)


def _shuffled_rows(arr, n, seed):
    n = min(int(n), len(arr))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(arr), size=n, replace=False)
    return idx, [arr[i:i + 1].astype(np.float32) for i in idx]


def _load_val_labels(n_needed):
    for name in ("superclass_labels_val.npy", "labels_val.npy"):
        path = os.path.join(CLEAN_DIR, name)
        if os.path.exists(path):
            y = np.load(path)
            if len(y) < n_needed:
                print(f"WARNING: {name} has {len(y)} rows < {n_needed} parity samples")
            return y
    print("WARNING: no val labels found; skipping AUROC/ECE quantization parity")
    return None


def main():
    cfg = get_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    unique_models = discover_ssl_model_dirs()

    print("Binding CPU ONNX Runtime (GPU inference profiling is out of scope)...")
    providers = probe_usable_providers(cfg.deploy.providers)
    print(f"Profiling provider: {providers}")

    calib_path = os.path.join(CLEAN_DIR, "signals_train.npy")
    if not os.path.exists(calib_path):
        raise FileNotFoundError("signals_train.npy required for quantization calibration (never test).")
    train_signals = np.load(calib_path)
    _, calib_data = _shuffled_rows(
        train_signals, cfg.deploy.calibration_samples, cfg.deploy.calibration_seed,
    )

    val_path = os.path.join(CLEAN_DIR, "signals_val.npy")
    val_signals = np.load(val_path) if os.path.exists(val_path) else train_signals
    parity_idx, parity_samples = _shuffled_rows(
        val_signals, cfg.deploy.parity_samples, cfg.deploy.calibration_seed,
    )
    timing_input = val_signals[int(parity_idx[0]):int(parity_idx[0]) + 1].astype(np.float32)

    y_val = _load_val_labels(len(val_signals))
    y_parity = y_val[parity_idx] if y_val is not None else None

    probe_metrics = {}
    pm = os.path.join(PROBE_DIR, "probe_metrics.json")
    if os.path.exists(pm):
        probe_metrics = json.load(open(pm))

    results = []
    for m_name, m_dir in unique_models.items():
        print(f"\nProfiling {m_name}")
        n_classes = probe_metrics.get(m_name, {}).get("n_classes", cfg.data.n_superclasses)
        wrapper = load_wrapper(m_dir, m_name, cfg, n_classes)

        fp32_path = os.path.join(OUTPUT_DIR, f"{m_name}_fp32.onnx")
        export_to_onnx(wrapper, fp32_path, opset=cfg.deploy.opset_version)

        quant_paths = {"fp32": fp32_path}
        for mode in cfg.deploy.quantization_modes:
            if mode == "fp32":
                continue
            quant_path = os.path.join(OUTPUT_DIR, f"{m_name}_{mode}.onnx")
            try:
                quantize_model(fp32_path, quant_path, mode=mode, calibration_data=calib_data)
                quant_paths[mode] = quant_path
            except Exception as e:
                print(f"  Quantization ({mode}) failed for {m_name}: {e}")

        for precision, model_path in quant_paths.items():
            parity = {
                "parity_cosine": None,
                "parity_max_abs_err": None,
                "parity_failed": None,
                "parity_auroc_fp32": None,
                "parity_auroc_quant": None,
                "parity_delta_auroc": None,
                "parity_ece_fp32": None,
                "parity_ece_quant": None,
                "parity_delta_ece": None,
            }
            if precision != "fp32":
                try:
                    parity = quantization_parity(
                        fp32_path, model_path, parity_samples,
                        provider="CPUExecutionProvider",
                        y_true=y_parity,
                        min_positives=cfg.eval.min_class_positives,
                    )
                    cosine = parity["parity_cosine"]
                    parity["parity_failed"] = bool(cosine < cfg.deploy.parity_min_cosine)
                    gate = "FAIL" if parity["parity_failed"] else "PASS"
                    extra = ""
                    if parity.get("parity_delta_auroc") is not None:
                        extra = (
                            f"  ΔAUROC={parity['parity_delta_auroc']:.4f}"
                            f"  ΔECE={parity['parity_delta_ece']:.4f}"
                        )
                    print(
                        f"  Parity {precision}: cosine={cosine:.6f} "
                        f"[{gate} vs {cfg.deploy.parity_min_cosine}]{extra}"
                    )
                except Exception as e:
                    print(f"  Parity check failed ({precision}): {e}")
            for provider in providers:
                try:
                    prof = profile_model(
                        model_path, m_name, precision, provider,
                        warmup=cfg.deploy.warmup_runs,
                        n_runs=cfg.deploy.benchmark_runs,
                        input_tensor=timing_input,
                    )
                    d = asdict(prof)
                    d.update(parity)
                    results.append(d)
                except ProviderUnavailableError as e:
                    print(f"  Skipping {m_name}/{precision} on {provider}: {e}")
                except Exception as e:
                    print(f"  Profiling failed for {m_name}/{precision} on {provider}: {e}")

    df = pd.DataFrame(results)
    df.to_parquet(os.path.join(OUTPUT_DIR, "deployment_profiles.parquet"), index=False)
    write_artifact_snapshot(
        OUTPUT_DIR, cfg,
        extra={
            "calib": "signals_train.npy (shuffled, never test)",
            "scope": "encoder+probe, CPU ORT only (no GPU inference profiling)",
            "providers_requested": list(cfg.deploy.providers),
            "providers_usable": providers,
        },
    )
    print("\n✓ Deployment profiling complete!")


if __name__ == "__main__":
    main()
