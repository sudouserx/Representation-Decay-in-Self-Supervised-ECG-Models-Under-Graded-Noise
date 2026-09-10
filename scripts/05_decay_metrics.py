#!/usr/bin/env python3
"""
Script 05 — Decay Metrics
=========================
Paired clean-vs-noisy AUROC, PR-AUC, F1 (val-tuned thresholds), ECE, Brier,
CKA, effective rank. Patient-clustered bootstrap CIs and paired tests operate
on noise-seed-averaged predictions. Geometry is computed for each noise seed
and then summarized; dependent noise realizations are never combined as if
they were independent. DeLong is retained as appendix only.

Kaggle Inputs:  ptbxl-clean-processed, corruption-eval-results, linear-probes-all
Kaggle Output:  /kaggle/working/decay-metrics-results/
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

CLEAN_DIR = os.environ.get("CLEAN_DIR", "/kaggle/input/ptbxl-clean-processed")
EVAL_DIR = os.environ.get("EVAL_DIR", "/kaggle/input/corruption-eval-results")
PROBE_DIR = os.environ.get("PROBE_DIR", "/kaggle/input/linear-probes-all")
UTILS_DIR = os.environ.get("UTILS_DIR", "/kaggle/input/ecg-ssl-utils")
OUTPUT_DIR = "/kaggle/working/decay-metrics-results"
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.artifact import write_artifact_snapshot
from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.eval.auroc import macro_auroc, subgroup_auroc
from ecg_ssl_utils.eval.bootstrap import (
    patient_bootstrap_ci,
    patient_bootstrap_pvalue,
)
from ecg_ssl_utils.eval.cka import linear_cka
from ecg_ssl_utils.eval.delong import delong_test
from ecg_ssl_utils.eval.ece import expected_calibration_error
from ecg_ssl_utils.eval.effective_rank import effective_rank
from ecg_ssl_utils.eval.f1 import per_class_f1
from ecg_ssl_utils.eval.prauc import per_class_brier, per_class_pr_auc


PRIMARY_PREFIXES = (
    "ssl-simclr-", "ssl-clocs-vit", "ssl-mae-", "ssl-jepa-",
)


def _parse_encoder_name(enc):
    pretrain_seed = None
    if "seed" in enc:
        try:
            pretrain_seed = int(enc.split("seed")[-1].split("-")[0])
        except ValueError:
            pretrain_seed = None
    sensitivity = not any(enc.startswith(p) for p in PRIMARY_PREFIXES)
    if enc.startswith("ssl-supervised"):
        sensitivity = False
    if enc.startswith("ssl-byol") or enc.startswith("ssl-swav") or "resnet" in enc:
        sensitivity = True
    return pretrain_seed, sensitivity


def _load_labels(clean_dir):
    sc_test = os.path.join(clean_dir, "superclass_labels_test.npy")
    if os.path.exists(sc_test):
        return np.load(sc_test), ["NORM", "MI", "STTC", "CD", "HYP"]
    return np.load(os.path.join(clean_dir, "labels_test.npy")), None


def _load_thresholds(enc, n_classes):
    path = os.path.join(PROBE_DIR, enc, "class_thresholds.json")
    if os.path.exists(path):
        data = json.load(open(path))
        thr = np.asarray(data.get("thresholds", [0.5] * n_classes), dtype=float)
        if thr.size == n_classes:
            return thr
    return np.full(n_classes, 0.5)


def _metric_bundle(labels, preds, reps, clean_reps, class_names, thresholds, cfg, pids):
    n_classes = labels.shape[1]
    auroc = macro_auroc(labels, preds)
    ece = expected_calibration_error(labels, preds, n_bins=cfg.eval.ece_bins)
    ece_em = expected_calibration_error(
        labels, preds, n_bins=cfg.eval.ece_bins, binning="equal_mass",
    ) if cfg.eval.ece_equal_mass else float("nan")
    f1 = per_class_f1(labels, preds, threshold=thresholds, class_names=class_names)
    f1_fixed = per_class_f1(
        labels, preds, threshold=np.full(n_classes, 0.5),
        class_names=class_names,
    )
    prauc = per_class_pr_auc(labels, preds, min_positives=cfg.eval.min_class_positives, class_names=class_names)
    brier = per_class_brier(labels, preds, class_names=class_names)
    cka, collapsed = linear_cka(clean_reps, reps, var_guard=cfg.eval.collapse_var_guard, return_collapse_flag=True)
    erank = effective_rank(reps, n_components=64, seed=cfg.eval.bootstrap_seed)
    return {
        "auroc": auroc,
        "ece": ece,
        "ece_equal_mass": ece_em,
        "f1_macro": f1["macro"],
        "f1_macro_fixed_0_5": f1_fixed["macro"],
        "prauc_macro": prauc["macro"],
        "brier_macro": brier["macro"],
        "cka": cka,
        "collapse_suspected": bool(collapsed),
        "erank": erank,
        **{f"f1_{k}": v for k, v in f1.items() if k != "macro"},
        **{f"prauc_{k}": v for k, v in prauc.items() if k != "macro"},
        **{f"brier_{k}": v for k, v in brier.items() if k != "macro"},
    }


def compute_clean_reference(enc, clean_reps, clean_preds, labels, pids, cfg, class_names, thresholds):
    pretrain_seed, sensitivity = _parse_encoder_name(enc)
    bundle = _metric_bundle(labels, clean_preds, clean_reps, clean_reps, class_names, thresholds, cfg, pids)
    _, auroc_lo, auroc_hi = patient_bootstrap_ci(
        lambda idx: macro_auroc(labels[idx], clean_preds[idx]),
        pids, n_bootstrap=cfg.eval.bootstrap_n, seed=cfg.eval.bootstrap_seed,
        point_estimate=bundle["auroc"],
    )
    _, ece_lo, ece_hi = patient_bootstrap_ci(
        lambda idx: expected_calibration_error(labels[idx], clean_preds[idx], n_bins=cfg.eval.ece_bins),
        pids, n_bootstrap=cfg.eval.bootstrap_n, seed=cfg.eval.bootstrap_seed,
        point_estimate=bundle["ece"],
    )
    def _er(idx):
        sub = clean_reps[idx]
        if len(sub) > cfg.eval.er_bootstrap_subsample:
            rng = np.random.RandomState(cfg.eval.bootstrap_seed)
            sub = sub[rng.choice(len(sub), cfg.eval.er_bootstrap_subsample, replace=False)]
        return effective_rank(sub, n_components=64, seed=cfg.eval.bootstrap_seed)
    _, er_lo, er_hi = patient_bootstrap_ci(
        _er, pids, n_bootstrap=cfg.eval.er_bootstrap_n, seed=cfg.eval.bootstrap_seed,
        point_estimate=bundle["erank"],
    )
    return {
        "encoder": enc,
        "pretrain_seed": pretrain_seed,
        "sensitivity_arm": sensitivity,
        **bundle,
        "auroc_lo": auroc_lo, "auroc_hi": auroc_hi,
        "ece_lo": ece_lo, "ece_hi": ece_hi,
        "erank_lo": er_lo, "erank_hi": er_hi,
    }


def _per_class_auroc(y, p, c):
    from sklearn.metrics import roc_auc_score
    if y[:, c].sum() < 5 or (1 - y[:, c]).sum() < 1:
        return float("nan")
    return float(roc_auc_score(y[:, c], p[:, c]))


def _bh_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    try:
        from statsmodels.stats.multitest import multipletests
        _, adj, _, _ = multipletests(p, method="fdr_bh")
        return adj
    except Exception:
        order = np.argsort(p)
        n = len(p)
        adj = np.empty(n)
        running = 1.0
        for rank, idx in enumerate(order[::-1], start=1):
            i = n - rank
            running = min(running, p[idx] * n / (i + 1))
            adj[idx] = min(running, 1.0)
        return adj


def main():
    cfg = get_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    labels_test, class_names = _load_labels(CLEAN_DIR)
    n_classes = labels_test.shape[1]
    meta = pd.read_parquet(os.path.join(CLEAN_DIR, "metadata.parquet"))
    test_meta = meta[meta["split"] == "test"].reset_index(drop=True)
    pids = test_meta["patient_id"].values

    import glob
    encoders = [os.path.basename(d) for d in glob.glob(os.path.join(EVAL_DIR, "ssl-*"))]

    print("Computing clean reference metrics...")
    clean_refs = []
    clean_store = {}
    for enc in encoders:
        clean_reps = np.load(os.path.join(PROBE_DIR, enc, "clean_test_repr.npy")).astype(np.float32)
        pred_path = os.path.join(PROBE_DIR, enc, "clean_test_predictions.npy")
        if not os.path.exists(pred_path):
            print(f"  WARNING: missing clean predictions for {enc}")
            continue
        clean_preds = np.load(pred_path)
        thresholds = _load_thresholds(enc, n_classes)
        ref = compute_clean_reference(
            enc, clean_reps, clean_preds, labels_test, pids, cfg, class_names, thresholds,
        )
        if "age" in test_meta.columns:
            ref["subgroup_age"] = json.dumps(subgroup_auroc(labels_test, clean_preds, test_meta["age"].values))
        if "sex" in test_meta.columns:
            ref["subgroup_sex"] = json.dumps(subgroup_auroc(labels_test, clean_preds, test_meta["sex"].values))
        clean_refs.append(ref)
        clean_store[enc] = (clean_reps, clean_preds, thresholds)
        print(f"  {enc}: AUROC={ref['auroc']:.4f} ECE={ref['ece']:.4f} ER={ref['erank']:.1f}")

    pd.DataFrame(clean_refs).to_parquet(os.path.join(OUTPUT_DIR, "clean_reference.parquet"), index=False)

    print("\nLoading per-seed noisy predictions...")
    per_seed_rows = []
    grouped = defaultdict(list)
    for enc in encoders:
        if enc not in clean_store:
            continue
        enc_dir = os.path.join(EVAL_DIR, enc)
        for cond_dir in os.listdir(enc_dir):
            cond_path = os.path.join(enc_dir, cond_dir)
            if not os.path.isdir(cond_path):
                continue
            parts = cond_dir.split("_")
            seed = int(parts[-1])
            snr = float(parts[-2].replace("db", ""))
            ntype = "_".join(parts[:-2])
            data = np.load(os.path.join(cond_path, "results.npz"))
            grouped[(enc, ntype, snr)].append((seed, data["representations"].astype(np.float32), data["predictions"]))

    print("Aggregating noise seeds (average probabilities, then metrics + CI)...")
    agg_rows = []
    test_rows = []
    for (enc, ntype, snr), items in grouped.items():
        clean_reps, clean_preds, thresholds = clean_store[enc]
        pretrain_seed, sensitivity = _parse_encoder_name(enc)
        items = sorted(items, key=lambda x: x[0])
        pred_stack = np.stack([p for _, _, p in items], axis=0)
        rep_stack = np.stack([r for _, r, _ in items], axis=0)
        avg_preds = pred_stack.mean(axis=0)
        seed_geometry = []

        for seed, reps, preds in items:
            bundle = _metric_bundle(labels_test, preds, reps, clean_reps, class_names, thresholds, cfg, pids)
            row = {
                "encoder": enc, "pretrain_seed": pretrain_seed, "sensitivity_arm": sensitivity,
                "noise_type": ntype, "snr_db": snr, "noise_seed": seed, **bundle,
            }
            per_seed_rows.append(row)
            seed_geometry.append((bundle["cka"], bundle["erank"]))

        # Prediction metrics are computed after averaging repeated corruption
        # realizations. Geometry remains per-realization and is summarized.
        bundle = _metric_bundle(
            labels_test, avg_preds, rep_stack[0], clean_reps,
            class_names, thresholds, cfg, pids,
        )
        bundle["cka"] = float(np.nanmean([g[0] for g in seed_geometry]))
        bundle["erank"] = float(np.nanmean([g[1] for g in seed_geometry]))
        bundle["cka_noise_seed_sd"] = float(np.nanstd([g[0] for g in seed_geometry]))
        bundle["erank_noise_seed_sd"] = float(np.nanstd([g[1] for g in seed_geometry]))

        for c in range(n_classes):
            c_name = class_names[c] if class_names else str(c)
            def _delta(idx, _c=c):
                return (
                    _per_class_auroc(labels_test[idx], clean_preds[idx], _c)
                    - _per_class_auroc(labels_test[idx], avg_preds[idx], _c)
                )
            try:
                dlt, pval = patient_bootstrap_pvalue(
                    _delta, pids, n_bootstrap=cfg.eval.bootstrap_n,
                    seed=cfg.eval.bootstrap_seed,
                )
            except Exception:
                dlt, pval = float("nan"), 1.0
            delong_p = np.nan
            try:
                if labels_test[:, c].sum() >= cfg.eval.min_class_positives:
                    _, delong_p, _, _ = delong_test(
                        labels_test[:, c], clean_preds[:, c], avg_preds[:, c],
                    )
            except Exception:
                pass
            test_rows.append({
                "encoder": enc, "pretrain_seed": pretrain_seed,
                "noise_type": ntype, "snr_db": snr, "class": c_name,
                "delta_auroc": dlt, "bootstrap_p": pval,
                "delong_p": delong_p, "n_noise_seeds": len(items),
            })
        _, auc_lo, auc_hi = patient_bootstrap_ci(
            lambda idx: macro_auroc(labels_test[idx], avg_preds[idx]),
            pids, n_bootstrap=cfg.eval.bootstrap_n, seed=cfg.eval.bootstrap_seed,
            point_estimate=bundle["auroc"],
        )
        _, ece_lo, ece_hi = patient_bootstrap_ci(
            lambda idx: expected_calibration_error(labels_test[idx], avg_preds[idx], n_bins=cfg.eval.ece_bins),
            pids, n_bootstrap=cfg.eval.bootstrap_n, seed=cfg.eval.bootstrap_seed,
            point_estimate=bundle["ece"],
        )
        _, cka_lo, cka_hi = patient_bootstrap_ci(
            lambda idx: np.nanmean([
                linear_cka(clean_reps[idx], reps[idx], var_guard=cfg.eval.collapse_var_guard)
                for reps in rep_stack
            ]),
            pids, n_bootstrap=cfg.eval.bootstrap_n, seed=cfg.eval.bootstrap_seed,
            point_estimate=bundle["cka"] if not np.isnan(bundle["cka"]) else 0.0,
        )

        def _er(idx):
            vals = []
            for reps in rep_stack:
                sub = reps[idx]
                if len(sub) > cfg.eval.er_bootstrap_subsample:
                    rng = np.random.RandomState(cfg.eval.bootstrap_seed)
                    sub = sub[rng.choice(len(sub), cfg.eval.er_bootstrap_subsample, replace=False)]
                vals.append(effective_rank(sub, n_components=64, seed=cfg.eval.bootstrap_seed))
            return float(np.nanmean(vals))

        _, er_lo, er_hi = patient_bootstrap_ci(
            _er, pids, n_bootstrap=cfg.eval.er_bootstrap_n, seed=cfg.eval.bootstrap_seed,
            point_estimate=bundle["erank"],
        )
        seed_aurocs = [macro_auroc(labels_test, p) for _, _, p in items]
        agg_rows.append({
            "encoder": enc, "pretrain_seed": pretrain_seed, "sensitivity_arm": sensitivity,
            "noise_type": ntype, "snr_db": snr, **bundle,
            "auroc_lo": auc_lo, "auroc_hi": auc_hi,
            "ece_lo": ece_lo, "ece_hi": ece_hi,
            "cka_lo": cka_lo, "cka_hi": cka_hi,
            "erank_lo": er_lo, "erank_hi": er_hi,
            "auroc_noise_seed_sd": float(np.std(seed_aurocs)),
            "n_noise_seeds": len(items),
        })

    df_seed = pd.DataFrame(per_seed_rows)
    df_agg = pd.DataFrame(agg_rows)
    df_tests = pd.DataFrame(test_rows)

    if not df_tests.empty:
        # Prespecified primary family: empirical NSTDB noise at one SNR.
        # Other rows remain available as explicitly exploratory tests.
        primary_mask = (
            df_tests["noise_type"].isin(cfg.eval.primary_noise_types)
            & np.isclose(df_tests["snr_db"], cfg.eval.primary_snr_db)
        )
        df_tests["primary_family"] = primary_mask
        df_tests["bootstrap_p_bh"] = np.nan
        if primary_mask.any():
            df_tests.loc[primary_mask, "bootstrap_p_bh"] = _bh_adjust(
                df_tests.loc[primary_mask, "bootstrap_p"].fillna(1.0).values
            )
        family_size = int(primary_mask.sum())
        df_tests.to_parquet(
            os.path.join(OUTPUT_DIR, "hypothesis_tests.parquet"), index=False,
        )
        family_note = {
            "family": {
                "noise_types": cfg.eval.primary_noise_types,
                "snr_db": cfg.eval.primary_snr_db,
                "dimensions": "encoder × noise-type × class",
            },
            "n_tests": family_size,
            "method": "paired patient-clustered bootstrap on noise-seed-averaged predictions; BH-FDR within primary family",
            "delong": "appendix only; independence assumption violated by multi-record patients",
            "exploratory": "all other SNR/noise rows are saved without multiplicity-adjusted claims",
        }
        with open(os.path.join(OUTPUT_DIR, "bh_family.json"), "w") as f:
            json.dump(family_note, f, indent=2)

    df_seed.to_parquet(os.path.join(OUTPUT_DIR, "metric_curves_per_seed.parquet"), index=False)
    df_agg.to_parquet(os.path.join(OUTPUT_DIR, "metric_curves.parquet"), index=False)

    if not df_agg.empty and "pretrain_seed" in df_agg.columns:
        prim = df_agg[~df_agg["sensitivity_arm"]].copy()
        if prim["pretrain_seed"].notna().any():
            grp = ["encoder", "noise_type", "snr_db"]
            metric_cols = [c for c in ["auroc", "ece", "cka", "erank", "f1_macro", "prauc_macro", "brier_macro"] if c in prim.columns]
            # collapse seed-tagged encoder names to paradigm stem
            prim["paradigm"] = prim["encoder"].str.replace(r"-seed\d+", "", regex=True)
            summary = prim.groupby(["paradigm", "noise_type", "snr_db"])[metric_cols].agg(["mean", "std"]).reset_index()
            summary.columns = ["_".join([c for c in col if c]).strip("_") for col in summary.columns]
            summary.to_parquet(os.path.join(OUTPUT_DIR, "pretrain_seed_summary.parquet"), index=False)

    write_artifact_snapshot(
        OUTPUT_DIR, cfg,
        extra={"stats": "patient_clustered_bootstrap_seed_averaged_bh"},
    )
    print("✓ Decay metrics computed!")


if __name__ == "__main__":
    main()
