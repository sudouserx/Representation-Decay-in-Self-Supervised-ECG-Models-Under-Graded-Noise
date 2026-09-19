#!/usr/bin/env python3
"""
Script 07 — Robustness Index & Reports
======================================
Secondary robustness index (pre-registered weights, two-sided ΔER),
Kendall-τ weight stability, decision-support filters (gates do not zero
the index), and report artifacts.

Important seed/reporting invariant:
    Script 05 is expected to provide all three primary pretraining seeds
    (42, 123, 456). Seed-specific model IDs are retained throughout this
    script. The report layer is audited so a helper that silently filters
    to seed 42 cannot produce the final manuscript-facing reports.

Kaggle Inputs: decay-metrics-results, deployment-profiles
Kaggle Output: /kaggle/working/ecg-ssl-robustness-report/
"""
import html
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

UTILS_DIR = os.environ.get("UTILS_DIR", "/kaggle/input/ecg-ssl-utils")
DECAY_DIR = os.environ.get("DECAY_DIR", "/kaggle/input/decay-metrics-results")
DEPLOY_DIR = os.environ.get("DEPLOY_DIR", "/kaggle/input/deployment-profiles")
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR", "/kaggle/working/ecg-ssl-robustness-report"
)

PRIMARY_PRETRAIN_SEEDS = (42, 123, 456)
PRIMARY_METHOD_PREFIXES = (
    "ssl-simclr-vit-small",
    "ssl-clocs-vit-small",
    "ssl-mae-vit-small",
    "ssl-jepa-vit-small",
    "ssl-supervised-vit-small",
)
SENSITIVITY_MODEL_IDS = {
    "ssl-byol-vit-small-seed42",
    "ssl-swav-vit-small-seed42",
    "ssl-clocs-resnet18-seed42",
}
SENSITIVITY_SEEDS = {42}
SNRS_PER_ENCODER = 6
NOISE_TYPES_PER_ENCODER = 9
CELLS_PER_ENCODER = SNRS_PER_ENCODER * NOISE_TYPES_PER_ENCODER

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


def _seed_from_model_id(model_id: str) -> int | None:
    match = re.search(r"-seed(\d+)$", str(model_id))
    return int(match.group(1)) if match else None


def _is_primary_model(model_id: str) -> bool:
    return any(
        str(model_id).startswith(f"{prefix}-seed")
        for prefix in PRIMARY_METHOD_PREFIXES
    )


def _validate_reporting_inputs(
    decay_df: pd.DataFrame,
    clean_ref_df: pd.DataFrame,
    deploy_df: pd.DataFrame,
    output_dir: str,
) -> list[str]:
    """Fail fast on incomplete seed/model coverage before any report is written."""
    required_decay = {
        "encoder",
        "pretrain_seed",
        "noise_type",
        "snr_db",
        "auroc",
        "ece",
        "cka",
        "erank",
    }
    missing_decay = sorted(required_decay.difference(decay_df.columns))
    if missing_decay:
        raise RuntimeError(
            f"metric_curves.parquet is missing required columns: {missing_decay}"
        )

    if clean_ref_df.empty:
        raise RuntimeError("clean_reference.parquet is empty or missing.")

    required_clean = {"encoder", "pretrain_seed", "auroc", "ece", "erank"}
    missing_clean = sorted(required_clean.difference(clean_ref_df.columns))
    if missing_clean:
        raise RuntimeError(
            f"clean_reference.parquet is missing required columns: {missing_clean}"
        )

    encoders = sorted(decay_df["encoder"].astype(str).unique())
    clean_encoders = sorted(clean_ref_df["encoder"].astype(str).unique())

    if len(encoders) != 18:
        raise RuntimeError(
            "Incomplete encoder coverage in metric_curves.parquet: "
            f"expected 18 seed-specific encoders, found {len(encoders)}. "
            f"Found: {encoders}"
        )

    expected_counts = decay_df.groupby("encoder").size()
    bad_counts = expected_counts[expected_counts != CELLS_PER_ENCODER]
    if not bad_counts.empty:
        raise RuntimeError(
            "Incomplete decay grid detected. Every encoder must have "
            f"{CELLS_PER_ENCODER} aggregated cells (9 noise types × 6 SNR). "
            f"Bad encoders: {bad_counts.to_dict()}"
        )

    if encoders != clean_encoders:
        missing_clean = sorted(set(encoders) - set(clean_encoders))
        extra_clean = sorted(set(clean_encoders) - set(encoders))
        raise RuntimeError(
            "Clean-reference coverage does not match decay encoders. "
            f"missing_clean={missing_clean}, extra_clean={extra_clean}"
        )

    duplicate_clean = clean_ref_df["encoder"].duplicated(keep=False)
    if duplicate_clean.any():
        dupes = sorted(clean_ref_df.loc[duplicate_clean, "encoder"].astype(str).unique())
        raise RuntimeError(
            "clean_reference.parquet contains duplicate encoder baselines: "
            f"{dupes}"
        )

    actual_seed_counts = (
        decay_df["pretrain_seed"].astype(int).value_counts().sort_index().to_dict()
    )
    clean_seed_counts = (
        clean_ref_df["pretrain_seed"].astype(int).value_counts().sort_index().to_dict()
    )

    expected_primary_ids = {
        f"{prefix}-seed{seed}"
        for prefix in PRIMARY_METHOD_PREFIXES
        for seed in PRIMARY_PRETRAIN_SEEDS
    }
    expected_all_ids = expected_primary_ids | SENSITIVITY_MODEL_IDS
    actual_ids = set(encoders)

    missing_models = sorted(expected_all_ids - actual_ids)
    unexpected_models = sorted(actual_ids - expected_all_ids)
    if missing_models or unexpected_models:
        raise RuntimeError(
            "Seed/model coverage mismatch. "
            f"missing={missing_models}, unexpected={unexpected_models}"
        )

    primary_seed_sets = {}
    for prefix in PRIMARY_METHOD_PREFIXES:
        prefix_ids = [
            model_id
            for model_id in encoders
            if model_id.startswith(f"{prefix}-seed")
        ]
        seeds = sorted(_seed_from_model_id(model_id) for model_id in prefix_ids)
        primary_seed_sets[prefix] = seeds
        if set(seeds) != set(PRIMARY_PRETRAIN_SEEDS):
            raise RuntimeError(
                f"Primary method {prefix} does not contain exactly seeds "
                f"{PRIMARY_PRETRAIN_SEEDS}: found {seeds}"
            )

    sensitivity_seeds = {
        model_id: _seed_from_model_id(model_id) for model_id in SENSITIVITY_MODEL_IDS
    }
    if set(sensitivity_seeds.values()) != SENSITIVITY_SEEDS:
        raise RuntimeError(
            f"Sensitivity arms must remain one-seed arms at seed 42; found {sensitivity_seeds}"
        )

    if len(clean_ref_df) != len(encoders):
        raise RuntimeError(
            f"Expected {len(encoders)} clean references, found {len(clean_ref_df)}"
        )

    if deploy_df.empty:
        deployment_status = "empty"
    elif "model_id" not in deploy_df.columns:
        deployment_status = "missing model_id column"
    else:
        deployment_models = set(deploy_df["model_id"].astype(str).unique())
        missing_deploy = sorted(actual_ids - deployment_models)
        deployment_status = {
            "models": len(deployment_models),
            "missing_models": missing_deploy,
        }
        if missing_deploy:
            print(
                "WARNING: deployment profile coverage is incomplete; "
                f"missing {len(missing_deploy)} models: {missing_deploy}"
            )

    validation = {
        "status": "PASS",
        "decay_rows": int(len(decay_df)),
        "unique_encoders": int(len(encoders)),
        "clean_reference_rows": int(len(clean_ref_df)),
        "pretrain_seed_counts_decay": {
            str(k): int(v) for k, v in actual_seed_counts.items()
        },
        "pretrain_seed_counts_clean_reference": {
            str(k): int(v) for k, v in clean_seed_counts.items()
        },
        "primary_seed_sets": primary_seed_sets,
        "sensitivity_models": sorted(SENSITIVITY_MODEL_IDS),
        "deployment": deployment_status,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "report_input_validation.json"), "w") as f:
        json.dump(validation, f, indent=2)

    print("==================================================")
    print("REPORT INPUT VALIDATION")
    print("==================================================")
    print(f"Decay rows:                  {len(decay_df)}")
    print(f"Unique encoders:              {len(encoders)}")
    print(f"Pretraining seeds:            {sorted(actual_seed_counts)}")
    print(f"Clean references:             {len(clean_ref_df)}")
    print("Primary seed grid:            COMPLETE")
    print("Decay grid per encoder:       COMPLETE (9 × 6)")
    print("Expected model set:           COMPLETE (15 primary + 3 sensitivity)")
    print("==================================================")

    return encoders


def _build_min_snr_summary(dss_df: pd.DataFrame, min_snr: float) -> pd.DataFrame:
    """Aggregate only true numeric metrics and explicitly preserve seed metadata."""
    min_snr_df = dss_df[dss_df["snr_db"] == min_snr].copy()

    numeric_cols = [
        "delta_cka",
        "delta_erank",
        "delta_ece",
        "delta_auroc",
        "ece",
        "auroc",
        "f1_macro",
        "cka",
        "erank",
        "robustness_index",
        "robustness_score",
    ]
    numeric_cols = [c for c in numeric_cols if c in min_snr_df.columns]

    summary = (
        min_snr_df.groupby("model_id", as_index=False)[numeric_cols]
        .mean(numeric_only=True)
    )

    metadata_cols = [c for c in ["pretrain_seed", "sensitivity_arm"] if c in dss_df.columns]
    metadata = dss_df[["model_id", *metadata_cols]].drop_duplicates("model_id")
    summary = summary.merge(metadata, on="model_id", how="left", validate="one_to_one")

    summary["dss"] = summary["robustness_score"]

    min_scores = (
        min_snr_df.groupby("model_id", as_index=False)["robustness_index"]
        .min()
        .rename(columns={"robustness_index": "robustness_index_min"})
    )
    summary = summary.merge(min_scores, on="model_id", how="left", validate="one_to_one")

    order = ["model_id"]
    for col in ["pretrain_seed", "sensitivity_arm"]:
        if col in summary.columns:
            order.append(col)
    summary = summary.sort_values(order).reset_index(drop=True)
    return summary


def _html_table(df: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    rows = []
    header = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    rows.append(f"<tr>{header}</tr>")

    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row.get(col, "")
            if pd.isna(value):
                text = ""
            elif isinstance(value, (float, np.floating)):
                text = f"{float(value):.{digits}f}"
            else:
                text = str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return "<table><thead>" + rows[0] + "</thead><tbody>" + "".join(rows[1:]) + "</tbody></table>"


def _write_seed_complete_leaderboard(
    summary: pd.DataFrame,
    output_path: str,
    reason: str,
) -> None:
    """Deterministic fallback that explicitly preserves all seed-specific model IDs."""
    primary = summary[summary["model_id"].map(_is_primary_model)].sort_values(
        ["model_id"]
    )
    sensitivity = summary[~summary["model_id"].map(_is_primary_model)].sort_values(
        ["model_id"]
    )

    columns = [
        "model_id",
        "pretrain_seed",
        "sensitivity_arm",
        "auroc",
        "ece",
        "robustness_score",
        "robustness_index_min",
        "latency_p50",
        "memory_mb",
    ]
    columns = [c for c in columns if c in summary.columns]

    css = """
    body { font-family: Arial, sans-serif; margin: 32px; color: #222; }
    h1, h2 { margin-bottom: 8px; }
    .meta { padding: 12px; background: #f4f4f4; border: 1px solid #ddd; margin: 12px 0 24px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 28px; }
    th, td { border: 1px solid #ddd; padding: 7px 9px; text-align: left; font-size: 13px; }
    th { background: #eee; }
    .note { color: #555; font-size: 13px; }
    """
    doc = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Seed-complete robustness leaderboard</title><style>{css}</style></head><body>",
        "<h1>Robustness Leaderboard — Seed-complete fallback</h1>",
        "<div class='meta'>",
        f"<b>Models:</b> {len(summary)} &nbsp; "
        f"<b>Primary:</b> {len(primary)} &nbsp; "
        f"<b>Sensitivity:</b> {len(sensitivity)}<br>",
        "Pretraining seeds retained explicitly: 42, 123, 456. "
        "Sensitivity arms remain separate and are not pooled with primary models.",
        "</div>",
        f"<p class='note'>Generated because the existing leaderboard helper did not expose the complete model set. {html.escape(reason)}</p>",
        "<h2>Primary models — seed-specific rows</h2>",
        _html_table(primary, columns),
        "<h2>Sensitivity arms — seed-specific rows</h2>",
        _html_table(sensitivity, columns),
        "</body></html>",
    ]
    Path(output_path).write_text("".join(doc), encoding="utf-8")


def _write_seed_complete_risk_report(
    dss_df: pd.DataFrame,
    output_path: str,
    reason: str,
) -> None:
    """Deterministic fallback risk report covering every seed-specific encoder."""
    risk_dir = os.path.dirname(output_path)
    os.makedirs(risk_dir, exist_ok=True)

    css = """
    body { font-family: Arial, sans-serif; margin: 28px; color: #222; }
    h1, h2 { margin-bottom: 8px; }
    .meta { padding: 12px; background: #f4f4f4; border: 1px solid #ddd; margin: 12px 0 22px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 28px; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; font-size: 12px; }
    th { background: #eee; }
    .note { color: #555; font-size: 13px; }
    """

    columns = [
        "model_id",
        "pretrain_seed",
        "sensitivity_arm",
        "snr_db",
        "auroc",
        "ece",
        "delta_auroc",
        "delta_ece",
        "cka",
        "erank",
        "robustness_index",
    ]
    columns = [c for c in columns if c in dss_df.columns]

    pieces = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Seed-complete risk report</title><style>{css}</style></head><body>",
        "<h1>Risk Report — Seed-complete fallback</h1>",
        "<div class='meta'>",
        f"<b>Rows:</b> {len(dss_df)} &nbsp; "
        f"<b>Encoders:</b> {dss_df['model_id'].nunique()} &nbsp; "
        f"<b>Noise types:</b> {dss_df['noise_type'].nunique()} &nbsp; "
        f"<b>SNR levels:</b> {dss_df['snr_db'].nunique()}<br>",
        "All seed-specific encoder IDs are retained. Primary seeds are 42, 123, 456; "
        "one-seed sensitivity arms are flagged separately.",
        "</div>",
        f"<p class='note'>Generated because the existing risk-report helper did not expose the complete model set. {html.escape(reason)}</p>",
    ]

    for noise_type in sorted(dss_df["noise_type"].astype(str).unique()):
        subset = dss_df[dss_df["noise_type"].astype(str) == noise_type].copy()
        subset = subset.sort_values(["model_id", "snr_db"], ascending=[True, False])
        pieces.append(f"<h2>{html.escape(noise_type)}</h2>")
        pieces.append(_html_table(subset, columns))

    pieces.append("</body></html>")
    Path(output_path).write_text("".join(pieces), encoding="utf-8")


def _html_contains_all_models(path: str, model_ids: list[str]) -> tuple[bool, list[str]]:
    if not os.path.exists(path):
        return False, model_ids
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    missing = [model_id for model_id in model_ids if model_id not in text]
    return len(missing) == 0, missing


def _directory_html_contains_all_models(directory: str, model_ids: list[str]) -> tuple[bool, list[str]]:
    html_files = list(Path(directory).glob("*.html")) if os.path.isdir(directory) else []
    combined = []
    for path in html_files:
        combined.append(path.read_text(encoding="utf-8", errors="ignore"))
    text = "\n".join(combined)
    missing = [model_id for model_id in model_ids if model_id not in text]
    return len(missing) == 0, missing


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
    clean_ref_df = (
        pd.read_parquet(clean_ref_path)
        if os.path.exists(clean_ref_path)
        else pd.DataFrame()
    )
    deploy_df = (
        pd.read_parquet(deploy_path)
        if os.path.exists(deploy_path)
        else pd.DataFrame()
    )

    encs = _validate_reporting_inputs(
        decay_df, clean_ref_df, deploy_df, OUTPUT_DIR
    )

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
            # This branch should now be unreachable because validation requires exact
            # clean-reference coverage. Keep it for defensive compatibility.
            delta_cka = 1.0 - row["cka"]
            delta_er = 0.0
            delta_auroc = 0.0
            delta_ece = row["ece"]

        dss_rows.append({
            "model_id": enc,
            "pretrain_seed": row.get("pretrain_seed", _seed_from_model_id(enc)),
            "sensitivity_arm": row.get("sensitivity_arm", enc in SENSITIVITY_MODEL_IDS),
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
            row["delta_cka_norm"],
            row["delta_erank_norm"],
            row["delta_ece_norm"],
            row["delta_auroc_norm"],
            weights=cfg.dss.weights,
        )
        scores.append(score)
    dss_df["robustness_index"] = scores
    dss_df["robustness_score"] = scores

    # Explicitly preserve seed metadata in the principal intermediate artifact.
    dss_df.to_parquet(
        os.path.join(OUTPUT_DIR, "robustness_results.parquet"), index=False
    )
    with open(os.path.join(OUTPUT_DIR, "normalization_anchors.json"), "w") as f:
        json.dump(anchors, f, indent=2)

    raw_cols = [
        "model_id",
        "pretrain_seed",
        "sensitivity_arm",
        "noise_type",
        "snr_db",
        "delta_auroc",
        "delta_ece",
        "delta_cka",
        "delta_erank",
        "auroc",
        "ece",
        "cka",
        "erank",
        "robustness_index",
    ]
    raw_dims = dss_df[[c for c in raw_cols if c in dss_df.columns]]
    raw_dims.to_parquet(
        os.path.join(OUTPUT_DIR, "raw_dimensions.parquet"), index=False
    )

    validity = {}
    if len(dss_df) > 3:
        for col in ["delta_cka", "delta_erank", "delta_ece", "robustness_index"]:
            rho, p = spearmanr(
                dss_df[col], dss_df["delta_auroc"], nan_policy="omit"
            )
            validity[col] = {
                "spearman_vs_delta_auroc": float(rho) if rho == rho else None,
                "p": float(p) if p == p else None,
            }
    with open(os.path.join(OUTPUT_DIR, "score_validity.json"), "w") as f:
        json.dump(validity, f, indent=2)

    min_snr = dss_df["snr_db"].min()
    min_snr_df = dss_df[dss_df["snr_db"] == min_snr]
    stability = weight_rank_stability(
        min_snr_df, cfg.dss.weights, n_samples=cfg.dss.stability_samples
    )
    with open(os.path.join(OUTPUT_DIR, "weight_stability.json"), "w") as f:
        json.dump(stability, f, indent=2)
    print(
        f"  Weight stability: τ={stability['mean_kendall_tau']:.3f} "
        f"top1={stability['frac_top1_agreement']:.3f}"
    )

    min_snr_mean = _build_min_snr_summary(dss_df, min_snr)
    min_snr_mean.to_parquet(
        os.path.join(OUTPUT_DIR, "min_snr_mean.parquet"), index=False
    )

    if not deploy_df.empty:
        engine = DecisionSupportEngine(
            min_robustness=0.50,
            max_latency_ms=100.0,
            max_memory_mb=512.0,
            auroc_gate=cfg.dss.auroc_gate,
            ece_gate=cfg.dss.ece_gate,
            parity_min_cosine=cfg.deploy.parity_min_cosine,
        )
        ds_results = engine.evaluate(
            min_snr_mean,
            deploy_df,
            stability["per_model_stability"],
        )
        generate_config_guidelines(
            ds_results,
            noise_condition=f"min-SNR mean across noise types ({min_snr} dB)",
            constraints={
                "min_robustness": 0.50,
                "max_latency_ms": 100.0,
                "max_memory_mb": 512.0,
            },
            output_dir=OUTPUT_DIR,
        )

    if not deploy_df.empty and {"precision", "provider"}.issubset(deploy_df.columns):
        latency = deploy_df[
            (deploy_df["precision"] == "fp32")
            & (deploy_df["provider"] == "CPUExecutionProvider")
        ]
        if not latency.empty:
            cols = [
                c for c in ["model_id", "latency_p50", "memory_mb"]
                if c in latency.columns
            ]
            min_snr_mean = min_snr_mean.merge(
                latency[cols], on="model_id", how="left", validate="one_to_one"
            )

    # First use the repository helper so existing formatting is preserved.
    leaderboard_path = os.path.join(OUTPUT_DIR, "leaderboard.html")
    generate_leaderboard(min_snr_mean, leaderboard_path)
    leaderboard_ok, leaderboard_missing = _html_contains_all_models(
        leaderboard_path, encs
    )
    if not leaderboard_ok:
        reason = (
            "Missing model IDs in helper output: "
            f"{leaderboard_missing}"
        )
        print("WARNING: leaderboard helper omitted seed-specific models.")
        print(f"         {reason}")
        _write_seed_complete_leaderboard(
            min_snr_mean,
            leaderboard_path,
            reason,
        )
    else:
        print("✓ Leaderboard contains all seed-specific encoders.")

    # Decay curves are built directly from the validated seed-complete dataframe.
    for ntype in dss_df["noise_type"].unique():
        data = {}
        snr_vals = None
        for enc in encs:
            subset = decay_df[
                (decay_df["encoder"] == enc)
                & (decay_df["noise_type"] == ntype)
            ].sort_values("snr_db", ascending=False)
            if len(subset) == 0:
                continue
            current_snr = subset["snr_db"].values
            if snr_vals is None:
                snr_vals = current_snr
            data[enc] = {
                "mean": subset["auroc"].values,
                "ci_lo": (
                    subset["auroc_lo"].values
                    if "auroc_lo" in subset.columns
                    else subset["auroc"].values
                ),
                "ci_hi": (
                    subset["auroc_hi"].values
                    if "auroc_hi" in subset.columns
                    else subset["auroc"].values
                ),
            }
        if data and snr_vals is not None:
            plot_decay_curves(
                data,
                "AUROC",
                ntype,
                snr_vals,
                list(data.keys()),
                save_path=os.path.join(
                    OUTPUT_DIR,
                    "decay_atlas",
                    f"auroc_vs_snr_{ntype}.png",
                ),
            )

    generate_decision_flowchart(
        min_snr_mean, os.path.join(OUTPUT_DIR, "decision_flowchart.md")
    )

    # Run the existing risk-report helper, then audit its output. If it silently
    # drops seed 123/456, replace only the final HTML with a seed-complete report.
    risk_dir = os.path.join(OUTPUT_DIR, "risk_reports")
    os.makedirs(risk_dir, exist_ok=True)
    generate_risk_report(decay_df, clean_ref_df, risk_dir)
    risk_ok, risk_missing = _directory_html_contains_all_models(risk_dir, encs)
    if not risk_ok:
        reason = f"Missing model IDs in helper output: {risk_missing}"
        print("WARNING: risk-report helper omitted seed-specific models.")
        print(f"         {reason}")
        _write_seed_complete_risk_report(
            dss_df,
            os.path.join(risk_dir, "risk_report.html"),
            reason,
        )
    else:
        print("✓ Risk report contains all seed-specific encoders.")

    write_artifact_snapshot(
        OUTPUT_DIR,
        cfg,
        extra={
            "index": "secondary_robustness_index",
            "report_seed_policy": {
                "primary_pretrain_seeds": list(PRIMARY_PRETRAIN_SEEDS),
                "sensitivity_seeds": sorted(SENSITIVITY_SEEDS),
                "validated_encoder_count": len(encs),
            },
        },
    )
    print("✓ Report generation complete!")


if __name__ == "__main__":
    main()