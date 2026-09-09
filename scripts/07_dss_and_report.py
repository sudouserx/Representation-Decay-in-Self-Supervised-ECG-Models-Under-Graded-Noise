#!/usr/bin/env python3
"""
Script 07 — Robustness Index & Reports
======================================
Secondary robustness index (pre-registered weights, two-sided ΔER),
Kendall-τ weight stability, decision-support filters (gates do not zero
the index), and report artifacts.

Kaggle Inputs: decay-metrics-results, deployment-profiles
Kaggle Output: /kaggle/working/ecg-ssl-robustness-report/
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

UTILS_DIR = os.environ.get("UTILS_DIR", "/kaggle/input/ecg-ssl-utils")
DECAY_DIR = os.environ.get("DECAY_DIR", "/kaggle/input/decay-metrics-results")
DEPLOY_DIR = os.environ.get("DEPLOY_DIR", "/kaggle/input/deployment-profiles")
OUTPUT_DIR = "/kaggle/working/ecg-ssl-robustness-report"
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.artifact import write_artifact_snapshot
from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.report.config_guidelines import generate_config_guidelines
from ecg_ssl_utils.report.decay_curves import plot_decay_curves
from ecg_ssl_utils.report.flowchart import generate_decision_flowchart
from ecg_ssl_utils.report.leaderboard import generate_leaderboard
from ecg_ssl_utils.report.risk_report import generate_risk_report
from ecg_ssl_utils.score.decision_support import DecisionSupportEngine
from ecg_ssl_utils.score.normalization import reference_anchored_normalize
from ecg_ssl_utils.score.robustness_score import compute_robustness_score
from ecg_ssl_utils.score.weight_stability import weight_rank_stability


def main():
    cfg = get_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "decay_atlas"), exist_ok=True)

    decay_path = os.path.join(DECAY_DIR, "metric_curves.parquet")
    clean_ref_path = os.path.join(DECAY_DIR, "clean_reference.parquet")
    deploy_path = os.path.join(DEPLOY_DIR, "deployment_profiles.parquet")

    if not os.path.exists(decay_path):
        print("Waiting for previous stages to complete (decay metrics missing).")
        return

    decay_df = pd.read_parquet(decay_path)
    clean_ref_df = pd.read_parquet(clean_ref_path) if os.path.exists(clean_ref_path) else pd.DataFrame()
    deploy_df = pd.read_parquet(deploy_path) if os.path.exists(deploy_path) else pd.DataFrame()

    clean_baselines = {}
    if not clean_ref_df.empty:
        for _, row in clean_ref_df.iterrows():
            clean_baselines[row["encoder"]] = row

    dss_rows = []
    for _, row in decay_df.iterrows():
        enc = row["encoder"]
        if enc in clean_baselines:
            baseline = clean_baselines[enc]
            delta_cka = 1.0 - row["cka"]
            clean_er = baseline["erank"]
            delta_er = abs(1.0 - (row["erank"] / clean_er)) if clean_er > 0 else 0.0
            delta_auroc = max(0.0, baseline["auroc"] - row["auroc"])
            delta_ece = max(0.0, row["ece"] - baseline["ece"])
        else:
            delta_cka = 1.0 - row["cka"]
            delta_er = 0.0
            delta_auroc = 0.0
            delta_ece = row["ece"]

        dss_rows.append({
            "model_id": enc,
            "pretrain_seed": row.get("pretrain_seed", None),
            "sensitivity_arm": row.get("sensitivity_arm", False),
            "noise_type": row["noise_type"],
            "snr_db": row["snr_db"],
            "delta_cka": delta_cka,
            "delta_erank": delta_er,
            "delta_ece": delta_ece,
            "delta_auroc": delta_auroc,
            "ece": row["ece"],
            "auroc": row["auroc"],
            "f1_macro": row.get("f1_macro", float("nan")),
            "cka": row["cka"],
            "erank": row["erank"],
        })

    dss_df = pd.DataFrame(dss_rows)
    anchors = {}
    for col in ["delta_cka", "delta_erank", "delta_ece", "delta_auroc"]:
        dss_df[f"{col}_norm"], anchors[col] = reference_anchored_normalize(
            dss_df[col].values, return_anchors=True,
        )

    scores = []
    for _, row in dss_df.iterrows():
        score, _ = compute_robustness_score(
            row["delta_cka_norm"], row["delta_erank_norm"],
            row["delta_ece_norm"], row["delta_auroc_norm"],
            weights=cfg.dss.weights,
        )
        scores.append(score)
    dss_df["robustness_index"] = scores
    dss_df["robustness_score"] = scores  # alias for existing report helpers
    dss_df.to_parquet(os.path.join(OUTPUT_DIR, "robustness_results.parquet"), index=False)
    with open(os.path.join(OUTPUT_DIR, "normalization_anchors.json"), "w") as f:
        json.dump(anchors, f, indent=2)

    raw_dims = dss_df[[
        "model_id", "noise_type", "snr_db",
        "delta_auroc", "delta_ece", "delta_cka", "delta_erank",
        "auroc", "ece", "cka", "erank", "robustness_index",
    ]]
    raw_dims.to_parquet(os.path.join(OUTPUT_DIR, "raw_dimensions.parquet"), index=False)

    validity = {}
    if len(dss_df) > 3:
        for col in ["delta_cka", "delta_erank", "delta_ece", "robustness_index"]:
            rho, p = spearmanr(dss_df[col], dss_df["delta_auroc"], nan_policy="omit")
            validity[col] = {"spearman_vs_delta_auroc": float(rho) if rho == rho else None, "p": float(p) if p == p else None}
    with open(os.path.join(OUTPUT_DIR, "score_validity.json"), "w") as f:
        json.dump(validity, f, indent=2)

    min_snr = dss_df["snr_db"].min()
    min_snr_df = dss_df[dss_df["snr_db"] == min_snr]
    stability = weight_rank_stability(min_snr_df, cfg.dss.weights, n_samples=cfg.dss.stability_samples)
    with open(os.path.join(OUTPUT_DIR, "weight_stability.json"), "w") as f:
        json.dump(stability, f, indent=2)
    print(f"  Weight stability: τ={stability['mean_kendall_tau']:.3f} top1={stability['frac_top1_agreement']:.3f}")

    min_snr_mean = min_snr_df.groupby("model_id").mean(numeric_only=True).reset_index()
    min_snr_mean["dss"] = min_snr_mean["robustness_score"]
    min_snr_min = min_snr_df.groupby("model_id")["robustness_index"].min().rename("robustness_index_min").reset_index()
    min_snr_mean = min_snr_mean.merge(min_snr_min, on="model_id", how="left")
    min_snr_mean.to_parquet(os.path.join(OUTPUT_DIR, "min_snr_mean.parquet"), index=False)

    if not deploy_df.empty:
        engine = DecisionSupportEngine(
            min_robustness=0.50,
            max_latency_ms=100.0,
            max_memory_mb=512.0,
            auroc_gate=cfg.dss.auroc_gate,
            ece_gate=cfg.dss.ece_gate,
            parity_min_cosine=cfg.deploy.parity_min_cosine,
        )
        ds_results = engine.evaluate(min_snr_mean, deploy_df, stability["per_model_stability"])
        generate_config_guidelines(
            ds_results,
            noise_condition=f"min-SNR mean across noise types ({min_snr} dB)",
            constraints={"min_robustness": 0.50, "max_latency_ms": 100.0, "max_memory_mb": 512.0},
            output_dir=OUTPUT_DIR,
        )

    if not deploy_df.empty:
        latency = deploy_df[
            (deploy_df["precision"] == "fp32")
            & (deploy_df["provider"] == "CPUExecutionProvider")
        ]
        if not latency.empty:
            cols = [c for c in ["model_id", "latency_p50", "memory_mb"] if c in latency.columns]
            min_snr_mean = min_snr_mean.merge(latency[cols], on="model_id", how="left")

    generate_leaderboard(min_snr_mean, os.path.join(OUTPUT_DIR, "leaderboard.html"))

    encs = dss_df["model_id"].unique()
    for ntype in dss_df["noise_type"].unique():
        data = {}
        snr_vals = None
        for enc in encs:
            subset = decay_df[
                (decay_df["encoder"] == enc) & (decay_df["noise_type"] == ntype)
            ].sort_values("snr_db", ascending=False)
            if len(subset) == 0:
                continue
            snr_vals = subset["snr_db"].values
            data[enc] = {
                "mean": subset["auroc"].values,
                "ci_lo": subset["auroc_lo"].values if "auroc_lo" in subset.columns else subset["auroc"].values,
                "ci_hi": subset["auroc_hi"].values if "auroc_hi" in subset.columns else subset["auroc"].values,
            }
        if data and snr_vals is not None:
            plot_decay_curves(
                data, "AUROC", ntype, snr_vals, list(data.keys()),
                save_path=os.path.join(OUTPUT_DIR, "decay_atlas", f"auroc_vs_snr_{ntype}.png"),
            )

    generate_decision_flowchart(min_snr_mean, os.path.join(OUTPUT_DIR, "decision_flowchart.md"))
    generate_risk_report(decay_df, clean_ref_df, os.path.join(OUTPUT_DIR, "risk_reports"))
    write_artifact_snapshot(OUTPUT_DIR, cfg, extra={"index": "secondary_robustness_index"})
    print("✓ Report generation complete!")


if __name__ == "__main__":
    main()
