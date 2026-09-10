#!/usr/bin/env python3
"""
Script 00 — Data Preparation
=============================
Load PTB-XL, save raw splits, bandpass (0.05–45 Hz sosfiltfilt), normalize
with train-only stats, and encode the official diagnostic-superclass labels.
The former strict likelihood >50 construction is retained as a sensitivity arm.

Kaggle Inputs:  PTB-XL dataset (e.g., 'khyeh0719/ptb-xl-dataset')
Kaggle Output:  /kaggle/working/ptbxl_clean/ → publish as 'ptbxl-clean-processed'
Est. Runtime:   ~30 min (CPU or GPU)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

PTBXL_DIR = os.environ.get(
    "PTBXL_DIR",
    "/kaggle/input/ptb-xl-dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
)
UTILS_DIR = os.environ.get("UTILS_DIR", "/kaggle/input/ecg-ssl-utils")
OUTPUT_DIR = "/kaggle/working/ptbxl_clean"

if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.artifact import write_artifact_snapshot
from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.data.label_encoder import (
    SUPERCLASS_NAMES,
    drop_empty_superclass_rows,
    encode_scp_labels,
    encode_superclass_labels,
    get_label_map,
    save_label_map,
)
from ecg_ssl_utils.data.preprocessing import (
    bandpass_filter,
    compute_norm_stats,
    normalize_signals,
    save_norm_stats,
)
from ecg_ssl_utils.data.ptbxl_loader import get_patient_split, load_ptbxl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-dataset",
        dest="strict_dataset",
        action="store_true",
        default=None,
        help="Assert full PTB-XL 1.0.3 record/patient counts (default: config).",
    )
    parser.add_argument(
        "--no-strict-dataset",
        dest="strict_dataset",
        action="store_false",
        help="Allow a subset (skip N/patient-count assertions).",
    )
    args = parser.parse_args()

    cfg = get_config()
    strict = cfg.data.strict_dataset if args.strict_dataset is None else args.strict_dataset
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    n_patients = int(metadata["patient_id"].nunique())
    print(f"  Unique patients: {n_patients}")

    if strict:
        assert signals.shape[0] == cfg.data.expected_n_records, (
            f"Expected {cfg.data.expected_n_records} records, got {signals.shape[0]}. "
            "Pass --no-strict-dataset for intentional subsets."
        )
        assert n_patients == cfg.data.expected_n_patients, (
            f"Expected {cfg.data.expected_n_patients} patients, got {n_patients}."
        )
        print("  ✓ Dataset-size assertions passed.")

    print("\n" + "=" * 60)
    print("STEP 2: Encoding labels")
    print("=" * 60)
    label_map = get_label_map(metadata)
    labels = encode_scp_labels(metadata, label_map)
    print(f"  SCP labels: {labels.shape}, avg {labels.sum(axis=1).mean():.2f}/record")

    scp_statements_path = os.path.join(PTBXL_DIR, "scp_statements.csv")
    superclass_official = None
    superclass_sensitivity = None
    if os.path.exists(scp_statements_path):
        superclass_official = encode_superclass_labels(
            metadata, scp_statements_path, threshold=None
        )
        superclass_sensitivity = encode_superclass_labels(
            metadata, scp_statements_path,
            threshold=cfg.data.sensitivity_label_threshold,
        )
        print(f"  Superclass names: {SUPERCLASS_NAMES}")
        for i, name in enumerate(SUPERCLASS_NAMES):
            print(
                f"    {name}: official={int(superclass_official[:, i].sum())} "
                f"sensitivity(>{cfg.data.sensitivity_label_threshold})="
                f"{int(superclass_sensitivity[:, i].sum())}"
            )
    else:
        print(f"  WARNING: scp_statements.csv not found at {scp_statements_path}")

    if (
        cfg.data.exclude_no_superclass
        and superclass_official is not None
    ):
        keep = drop_empty_superclass_rows(superclass_official)
        n_drop = int((~keep).sum())
        legacy_empty = ~drop_empty_superclass_rows(superclass_sensitivity)
        cohort_report = {
            "definition": "official PTB-XL diagnostic superclass aggregation",
            "n_input": int(len(metadata)),
            "n_official_empty": n_drop,
            "n_legacy_gt50_empty": int(legacy_empty.sum()),
            "legacy_threshold": cfg.data.sensitivity_label_threshold,
            "by_fold": {
                str(fold): {
                    "n": int((metadata["strat_fold"] == fold).sum()),
                    "official_empty": int(((metadata["strat_fold"] == fold) & ~keep).sum()),
                    "legacy_gt50_empty": int(((metadata["strat_fold"] == fold) & legacy_empty).sum()),
                }
                for fold in sorted(metadata["strat_fold"].unique())
            },
            "class_counts": {
                name: {
                    "official": int(superclass_official[:, i].sum()),
                    "legacy_gt50": int(superclass_sensitivity[:, i].sum()),
                }
                for i, name in enumerate(SUPERCLASS_NAMES)
            },
        }
        if "sex" in metadata:
            cohort_report["by_sex"] = {
                str(sex): {
                    "n": int((metadata["sex"] == sex).sum()),
                    "official_empty": int(((metadata["sex"] == sex) & ~keep).sum()),
                    "legacy_gt50_empty": int(((metadata["sex"] == sex) & legacy_empty).sum()),
                }
                for sex in sorted(metadata["sex"].dropna().unique())
            }
        if "age" in metadata:
            age_group = pd.cut(
                metadata["age"], bins=[0, 40, 60, 80, np.inf],
                labels=["0-40", "40-60", "60-80", "80+"],
                include_lowest=True,
            )
            cohort_report["by_age"] = {
                str(group): {
                    "n": int((age_group == group).sum()),
                    "official_empty": int(((age_group == group) & ~keep).sum()),
                    "legacy_gt50_empty": int(((age_group == group) & legacy_empty).sum()),
                }
                for group in age_group.cat.categories
            }
        with open(os.path.join(OUTPUT_DIR, "cohort_definition.json"), "w") as f:
            import json
            json.dump(cohort_report, f, indent=2)
        print(f"  Excluding {n_drop} records with no official diagnostic superclass.")
        signals = signals[keep]
        metadata = metadata.loc[keep].reset_index(drop=True)
        labels = labels[keep]
        superclass_official = superclass_official[keep]
        superclass_sensitivity = superclass_sensitivity[keep]
        print(f"  Remaining records: {len(metadata)}")

    print("\n" + "=" * 60)
    print("STEP 3: Patient-level split")
    print("=" * 60)
    splits = get_patient_split(
        metadata,
        train_folds=cfg.data.train_folds,
        val_folds=cfg.data.val_folds,
        test_folds=cfg.data.test_folds,
    )
    metadata["split"] = "train"
    metadata.loc[splits["val"], "split"] = "val"
    metadata.loc[splits["test"], "split"] = "test"
    for name, idx in splits.items():
        print(f"  {name}: {len(idx)} records, {metadata.loc[idx, 'patient_id'].nunique()} patients")

    print("\n" + "=" * 60)
    print("STEP 4: Saving unfiltered raw splits")
    print("=" * 60)
    for split_name, idx in splits.items():
        raw_path = os.path.join(OUTPUT_DIR, f"signals_{split_name}_raw.npy")
        np.save(raw_path, signals[idx])
        print(f"  {split_name} raw: {signals[idx].shape} → {raw_path}")

    print("\n" + "=" * 60)
    print(f"STEP 5: Bandpass filtering ({cfg.data.bandpass_low}–{cfg.data.bandpass_high} Hz, sosfiltfilt)")
    print("=" * 60)
    filtered = bandpass_filter(
        signals,
        low=cfg.data.bandpass_low,
        high=cfg.data.bandpass_high,
        fs=cfg.data.sampling_rate,
        order=cfg.data.filter_order,
    )

    print("\n" + "=" * 60)
    print("STEP 6: Train-only normalization stats")
    print("=" * 60)
    train_signals = filtered[splits["train"]]
    norm_stats = compute_norm_stats(train_signals)
    print(f"  Per-lead mean: {norm_stats['mean']}")
    print(f"  Per-lead std:  {norm_stats['std']}")
    filtered = normalize_signals(filtered, norm_stats["mean"], norm_stats["std"])

    print("\n" + "=" * 60)
    print("STEP 7: Saving artifacts")
    print("=" * 60)
    for split_name, idx in splits.items():
        np.save(os.path.join(OUTPUT_DIR, f"signals_{split_name}.npy"), filtered[idx])
        np.save(os.path.join(OUTPUT_DIR, f"labels_{split_name}.npy"), labels[idx])
        if superclass_official is not None:
            np.save(
                os.path.join(OUTPUT_DIR, f"superclass_labels_{split_name}.npy"),
                superclass_official[idx],
            )
            np.save(
                os.path.join(OUTPUT_DIR, f"superclass_labels_raw_{split_name}.npy"),
                superclass_official[idx],
            )
            np.save(
                os.path.join(OUTPUT_DIR, f"superclass_labels_likelihood_gt50_{split_name}.npy"),
                superclass_sensitivity[idx],
            )
        extra = f", superclass {superclass_official[idx].shape}" if superclass_official is not None else ""
        print(f"  {split_name}: signals {filtered[idx].shape}, labels {labels[idx].shape}{extra}")

    metadata.to_parquet(os.path.join(OUTPUT_DIR, "metadata.parquet"), index=False)
    save_norm_stats(norm_stats, os.path.join(OUTPUT_DIR, "norm_stats.json"))
    save_label_map(label_map, os.path.join(OUTPUT_DIR, "label_map.json"))

    write_artifact_snapshot(
        OUTPUT_DIR,
        cfg,
        extra={
            "n_train": int(len(splits["train"])),
            "n_val": int(len(splits["val"])),
            "n_test": int(len(splits["test"])),
            "n_records_after_exclusion": int(len(metadata)),
            "n_patients_after_exclusion": int(metadata["patient_id"].nunique()),
            "label_definition": "official_all_diagnostic_statements",
            "sensitivity_label_threshold_strict_gt": cfg.data.sensitivity_label_threshold,
            "ptbxl_version": cfg.data.ptbxl_version,
            "filter": "sosfiltfilt",
        },
    )

    print("\n✓ Data preparation complete!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Files: {os.listdir(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
