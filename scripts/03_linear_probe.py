#!/usr/bin/env python3
"""
Script 03 — Linear Probe Training
==================================
Train linear probe on clean representations from all frozen SSL encoders,
fit temperature on raw validation logits (applied exactly once), and tune
per-class F1 thresholds on validation.

Kaggle Inputs:  ptbxl-clean-processed, all ssl-* model outputs
Kaggle Output:  /kaggle/working/linear-probes-all/
"""
import json
import os
import sys

import numpy as np
import torch

CLEAN_DIR = os.environ.get("CLEAN_DIR", "/kaggle/input/ptbxl-clean-processed")
UTILS_DIR = os.environ.get("UTILS_DIR", "/kaggle/input/ecg-ssl-utils")
OUTPUT_DIR = "/kaggle/working/linear-probes-all"
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.artifact import (
    build_model_manifest, file_sha256, write_artifact_snapshot,
)
from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.eval.ece import expected_calibration_error
from ecg_ssl_utils.eval.f1 import per_class_f1
from ecg_ssl_utils.models.resnet18_1d import ResNet18_1D
from ecg_ssl_utils.models.vit_small_1d import ViTSmall1D
from ecg_ssl_utils.probe.calibration import temperature_scaling
from ecg_ssl_utils.probe.linear_probe import LinearProbe, train_probe


def load_encoder(model_dir, cfg, device):
    with open(os.path.join(model_dir, "config.json")) as f:
        m_cfg = json.load(f)
    backbone = m_cfg.get("backbone") or m_cfg.get("config", {}).get("backbone", {}).get("arch", "vit_small_1d")
    if isinstance(m_cfg.get("config"), dict) and "backbone" not in m_cfg:
        extra = m_cfg.get("extra") or m_cfg
        backbone = extra.get("backbone", backbone)
    if backbone == "vit_small_1d":
        encoder = ViTSmall1D(
            patch_size=cfg.backbone.patch_size, embed_dim=cfg.backbone.embed_dim,
            depth=cfg.backbone.depth, num_heads=cfg.backbone.num_heads,
        )
    else:
        encoder = ResNet18_1D(in_channels=12, output_dim=cfg.backbone.embed_dim)

    encoder.load_state_dict(torch.load(os.path.join(model_dir, "encoder.pt"), map_location=device))
    encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder, m_cfg


def extract_features(encoder, signals, batch_size=256, device="cuda"):
    features = []
    for i in range(0, len(signals), batch_size):
        batch = torch.tensor(signals[i:i + batch_size], dtype=torch.float32).to(device)
        with torch.no_grad():
            features.append(encoder(batch).cpu().numpy())
    return np.vstack(features)


def _load_labels(clean_dir):
    sc_train = os.path.join(clean_dir, "superclass_labels_train.npy")
    sc_val = os.path.join(clean_dir, "superclass_labels_val.npy")
    sc_test = os.path.join(clean_dir, "superclass_labels_test.npy")
    if os.path.exists(sc_train) and os.path.exists(sc_val):
        labels_train = np.load(sc_train)
        labels_val = np.load(sc_val)
        labels_test = np.load(sc_test) if os.path.exists(sc_test) else None
        return labels_train, labels_val, labels_test, labels_train.shape[1], "superclass"
    labels_train = np.load(os.path.join(clean_dir, "labels_train.npy"))
    labels_val = np.load(os.path.join(clean_dir, "labels_val.npy"))
    labels_test = np.load(os.path.join(clean_dir, "labels_test.npy"))
    return labels_train, labels_val, labels_test, labels_train.shape[1], "scp_codes"


def raw_logits(probe, feat, device):
    """Linear logits with temperature forced to 1 (no scaling)."""
    with torch.no_grad():
        return probe.fc(torch.tensor(feat, dtype=torch.float32).to(device)).cpu().numpy()


def predict_probs(logits, T):
    """Apply temperature exactly once, then sigmoid."""
    return 1.0 / (1.0 + np.exp(-logits / T))


def tune_thresholds(y_true, y_prob, grid=None):
    if grid is None:
        grid = np.arange(0.05, 1.0, 0.05)
    n_classes = y_true.shape[1]
    thresholds = np.full(n_classes, 0.5, dtype=np.float32)
    for c in range(n_classes):
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            f1 = per_class_f1(y_true[:, c:c + 1], y_prob[:, c:c + 1], threshold=float(t))
            if f1["macro"] > best_f1:
                best_f1, best_t = f1["macro"], float(t)
        thresholds[c] = best_t
    return thresholds


def discover_models():
    import glob
    model_dirs = glob.glob("/kaggle/input/ssl-*") + glob.glob("/kaggle/working/ssl-*")
    model_dirs = [d for d in model_dirs if os.path.exists(os.path.join(d, "encoder.pt"))]
    return {os.path.basename(d): d for d in model_dirs}


def main():
    cfg = get_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    unique_models = discover_models()
    model_manifest = build_model_manifest(
        unique_models, cfg.ssl_training.pretrain_seeds,
    )
    cohort_path = os.path.join(CLEAN_DIR, "cohort_definition.json")
    if not os.path.exists(cohort_path):
        raise FileNotFoundError(
            "cohort_definition.json missing; rebuild data with script 00",
        )
    model_manifest["cohort_definition_sha256"] = file_sha256(cohort_path)
    with open(os.path.join(OUTPUT_DIR, "model_manifest.json"), "w") as f:
        json.dump(model_manifest, f, indent=2)
    signals_train = np.load(os.path.join(CLEAN_DIR, "signals_train.npy"))
    signals_val = np.load(os.path.join(CLEAN_DIR, "signals_val.npy"))
    signals_test = np.load(os.path.join(CLEAN_DIR, "signals_test.npy"))
    labels_train, labels_val, labels_test, n_classes, label_type = _load_labels(CLEAN_DIR)
    class_names = ["NORM", "MI", "STTC", "CD", "HYP"] if n_classes == 5 else None

    results = {}
    for m_name, m_dir in unique_models.items():
        print(f"\nProcessing {m_name}...")
        encoder, m_cfg = load_encoder(m_dir, cfg, device)

        train_feat = extract_features(encoder, signals_train, device=device)
        val_feat = extract_features(encoder, signals_val, device=device)
        test_feat = extract_features(encoder, signals_test, device=device)

        m_out_dir = os.path.join(OUTPUT_DIR, m_name)
        os.makedirs(m_out_dir, exist_ok=True)
        np.save(os.path.join(m_out_dir, "clean_train_repr.npy"), train_feat)
        np.save(os.path.join(m_out_dir, "clean_test_repr.npy"), test_feat.astype(np.float32))

        probe, val_auroc = train_probe(
            train_feat, labels_train, val_feat, labels_val,
            in_dim=cfg.backbone.embed_dim, n_classes=n_classes,
            epochs=cfg.probe.epochs, batch_size=cfg.probe.batch_size,
            lr=cfg.probe.lr, patience=cfg.probe.patience, device=device,
            seed=cfg.eval.bootstrap_seed,
        )

        probe.eval()
        val_logits_raw = raw_logits(probe, val_feat, device)
        T = float(temperature_scaling(val_logits_raw, labels_val))
        probe.temperature.data = torch.tensor([T], device=device)

        val_probs = predict_probs(val_logits_raw, T)
        test_logits = raw_logits(probe, test_feat, device)
        test_probs = predict_probs(test_logits, T)

        thresholds = tune_thresholds(labels_val, val_probs)
        val_ece = expected_calibration_error(labels_val, val_probs, n_bins=cfg.probe.n_calibration_bins)
        print(f"  Val AUROC: {val_auroc:.4f} | T: {T:.4f} | Val ECE: {val_ece:.4f}")
        print(f"  Val-tuned thresholds: {thresholds.tolist()}")

        np.save(os.path.join(m_out_dir, "clean_test_logits.npy"), test_logits)
        np.save(os.path.join(m_out_dir, "clean_test_predictions.npy"), test_probs)
        with open(os.path.join(m_out_dir, "temperature.json"), "w") as f:
            json.dump({"T": T}, f, indent=2)
        with open(os.path.join(m_out_dir, "class_thresholds.json"), "w") as f:
            json.dump(
                {
                    "thresholds": [float(t) for t in thresholds],
                    "class_names": class_names,
                },
                f,
                indent=2,
            )
        torch.save(probe.state_dict(), os.path.join(m_out_dir, "probe.pt"))

        pretrain_seed = None
        if "seed" in m_name:
            try:
                pretrain_seed = int(m_name.split("seed")[-1].split("-")[0])
            except ValueError:
                pretrain_seed = m_cfg.get("seed")
        else:
            pretrain_seed = m_cfg.get("seed")

        results[m_name] = {
            "val_auroc": float(val_auroc),
            "val_ece": float(val_ece),
            "temperature": T,
            "n_classes": n_classes,
            "label_type": label_type,
            "thresholds": [float(t) for t in thresholds],
            "pretrain_seed": pretrain_seed,
            "sensitivity_arm": bool(m_cfg.get("sensitivity_arm", False)),
        }

    with open(os.path.join(OUTPUT_DIR, "probe_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_artifact_snapshot(OUTPUT_DIR, cfg, extra={"n_models": len(results)})
    print("\n✓ Linear probe training complete!")


if __name__ == "__main__":
    main()
