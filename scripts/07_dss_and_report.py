#!/usr/bin/env python3
"""
Script 07 — Robustness Score Computation & Report Generation
=============================================================
Compute Robustness Score (performance decay + calibration degradation +
representation decay), Sobol sensitivity analysis, and generate decision
artifacts including per-noise-type risk reports.

Deployment cost (latency/memory) is used as a constraint filter in the
decision-support artifact, NOT as a component of the robustness score.

Kaggle Inputs: decay-metrics-results, deployment-profiles
Kaggle Output: /kaggle/working/ecg-ssl-robustness-report/
"""
import os, sys, json
import numpy as np, pandas as pd

UTILS_DIR = os.environ.get('UTILS_DIR', '/kaggle/input/ecg-ssl-utils')
DECAY_DIR = os.environ.get('DECAY_DIR', '/kaggle/input/decay-metrics-results')
DEPLOY_DIR = os.environ.get('DEPLOY_DIR', '/kaggle/input/deployment-profiles')
OUTPUT_DIR = '/kaggle/working/ecg-ssl-robustness-report'
if UTILS_DIR not in sys.path: sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.score.robustness_score import compute_robustness_score
from ecg_ssl_utils.score.normalization import reference_anchored_normalize
from ecg_ssl_utils.score.sobol import sobol_sensitivity
from ecg_ssl_utils.report.decay_curves import plot_decay_curves
from ecg_ssl_utils.report.leaderboard import generate_leaderboard
from ecg_ssl_utils.report.flowchart import generate_decision_flowchart
from ecg_ssl_utils.report.risk_report import generate_risk_report

def main():
    cfg = get_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'decay_atlas'), exist_ok=True)
    
    decay_path = os.path.join(DECAY_DIR, 'metric_curves.parquet')
    clean_ref_path = os.path.join(DECAY_DIR, 'clean_reference.parquet')
    deploy_path = os.path.join(DEPLOY_DIR, 'deployment_profiles.parquet')
    
    if not os.path.exists(decay_path):
        print("Waiting for previous stages to complete (decay metrics missing).")
        return
        
    decay_df = pd.read_parquet(decay_path)
    clean_ref_df = pd.read_parquet(clean_ref_path) if os.path.exists(clean_ref_path) else pd.DataFrame()
    deploy_df = pd.read_parquet(deploy_path) if os.path.exists(deploy_path) else pd.DataFrame()
    
    # ── 1. Compute deltas relative to clean reference ──
    # Build clean baselines lookup
    clean_baselines = {}
    if not clean_ref_df.empty:
        for _, row in clean_ref_df.iterrows():
            clean_baselines[row['encoder']] = row
    
    dss_rows = []
    for _, row in decay_df.iterrows():
        enc = row['encoder']
        
        # Get clean baseline for this encoder
        if enc in clean_baselines:
            baseline = clean_baselines[enc]
            # ΔCKA = 1 − CKA(clean, noisy) — already stored as raw CKA in decay_df
            delta_cka = 1.0 - row['cka']
            # ΔEffective Rank = 1 − ER(noisy) / ER(clean)
            clean_er = baseline['erank']
            delta_er = 1.0 - (row['erank'] / clean_er) if clean_er > 0 else 0.0
            # ΔAUROC = clean_auroc - noisy_auroc (higher = more decay)
            delta_auroc = max(0.0, baseline['auroc'] - row['auroc'])
            # ΔECE = noisy_ece - clean_ece (higher = worse calibration)
            delta_ece = max(0.0, row['ece'] - baseline['ece'])
        else:
            # Fallback: no clean reference, use raw values
            delta_cka = 1.0 - row['cka']
            delta_er = 0.0
            delta_auroc = 0.0
            delta_ece = row['ece']
            
        dss_rows.append({
            'model_id': enc,
            'noise_type': row['noise_type'],
            'snr_db': row['snr_db'],
            'delta_cka': delta_cka,
            'delta_erank': delta_er,
            'delta_ece': delta_ece,
            'delta_auroc': delta_auroc,
            'ece': row['ece'],
            'auroc': row['auroc'],
            'f1_macro': row.get('f1_macro', float('nan')),
        })
        
    dss_df = pd.DataFrame(dss_rows)
    
    # ── 2. Normalize deltas globally ──
    # METHODOLOGY NOTE:
    # Global min-max normalization is anchored to the observed worst/best case in the 
    # current experimental grid. If a future run contains different corruption extremes, 
    # all normalized scores will shift. This transformation is valid for within-experiment 
    # ranking, but limits absolute comparability across independent studies.
    for col in ['delta_cka', 'delta_erank', 'delta_ece', 'delta_auroc']:
        dss_df[f'{col}_norm'] = reference_anchored_normalize(dss_df[col].values)
        
    # ── 3. Compute Robustness Score (no latency) ──
    scores = []
    for _, row in dss_df.iterrows():
        score, comps = compute_robustness_score(
            row['delta_cka_norm'],
            row['delta_erank_norm'],
            row['delta_ece_norm'],
            row['delta_auroc_norm'],
            weights=cfg.dss.weights,
        )
        scores.append(score)
    dss_df['robustness_score'] = scores
    dss_df.to_parquet(os.path.join(OUTPUT_DIR, 'robustness_results.parquet'), index=False)
    
    # ── 4. Sobol Sensitivity Analysis ──
    # Evaluate ranking stability across scoring weights
    worst_case = dss_df[dss_df['snr_db'] == dss_df['snr_db'].min()]
    if len(worst_case) > 0:
        def eval_ranking(weights):
            """Compute a ranking-based metric for given weights."""
            temp_scores = []
            for _, r in worst_case.iterrows():
                s, _ = compute_robustness_score(
                    r['delta_cka_norm'], r['delta_erank_norm'],
                    r['delta_ece_norm'], r['delta_auroc_norm'],
                    weights=weights,
                )
                temp_scores.append(s)
            return np.std(temp_scores)  # variance as stability proxy

        sensitivity = sobol_sensitivity(eval_ranking, n_samples=cfg.dss.sobol_samples)
        with open(os.path.join(OUTPUT_DIR, 'sobol_sensitivity.json'), 'w') as f:
            json.dump(sensitivity, f, indent=2)
        print(f"  Sobol sensitivity: {sensitivity}")
    
    # ── 5. Report Artifacts ──

    # 5a. Decision Support Engine & Guidelines (R1 & R2)
    from ecg_ssl_utils.score.decision_support import DecisionSupportEngine
    from ecg_ssl_utils.report.config_guidelines import generate_config_guidelines
    
    worst_agg = dss_df[dss_df['snr_db'] == dss_df['snr_db'].min()].groupby('model_id').mean(numeric_only=True).reset_index()
    # Add 'dss' column alias for leaderboard compatibility
    worst_agg['dss'] = worst_agg['robustness_score']
    
    if not deploy_df.empty:
        engine = DecisionSupportEngine(
            min_robustness=0.50,
            max_latency_ms=100.0,
            max_memory_mb=512.0,
            auroc_gate=cfg.dss.auroc_gate,
            ece_gate=cfg.dss.ece_gate
        )
        
        # We need mock Sobol stability map for the engine if it exists
        sobol_map = {m: 1.0 for m in worst_agg['model_id']}  # Mock for now, could parse sensitivity
        ds_results = engine.evaluate(worst_agg, deploy_df, sobol_map)
        
        generate_config_guidelines(
            ds_results, 
            noise_condition=f"Worst-case SNR ({dss_df['snr_db'].min()} dB)",
            constraints={'min_robustness': 0.50, 'max_latency_ms': 100.0, 'max_memory_mb': 512.0},
            output_dir=OUTPUT_DIR
        )

    # 5b. Leaderboard — use deployment constraints as filters
    # Merge deployment info if available (as filter, not score component)
    if not deploy_df.empty:
        latency = deploy_df[
            (deploy_df['precision'] == 'fp32') &
            (deploy_df['provider'] == 'CPUExecutionProvider')
        ]
        if not latency.empty:
            lat_cols = latency[['model_id', 'latency_p50', 'memory_mb']].rename(
                columns={'model_id': 'model_id'}
            )
            worst_agg = worst_agg.merge(lat_cols, on='model_id', how='left')

    generate_leaderboard(worst_agg, deploy_df, os.path.join(OUTPUT_DIR, 'leaderboard.html'))
    
    # 5b. Decay Curves
    encs = dss_df['model_id'].unique()
    for ntype in dss_df['noise_type'].unique():
        data = {}
        for enc in encs:
            subset = decay_df[
                (decay_df['encoder'] == enc) & (decay_df['noise_type'] == ntype)
            ].sort_values('snr_db', ascending=False)
            if len(subset) == 0:
                continue
            data[enc] = {
                'mean': subset['auroc'].values,
                'ci_lo': subset['auroc_lo'].values if 'auroc_lo' in subset.columns else subset['auroc'].values,
                'ci_hi': subset['auroc_hi'].values if 'auroc_hi' in subset.columns else subset['auroc'].values,
            }
        if data:
            plot_decay_curves(
                data, 'AUROC', ntype, subset['snr_db'].values, list(data.keys()),
                save_path=os.path.join(OUTPUT_DIR, 'decay_atlas', f'auroc_vs_snr_{ntype}.png'),
            )
                           
    # 5c. Decision Flowchart
    generate_decision_flowchart(worst_agg, os.path.join(OUTPUT_DIR, 'decision_flowchart.md'))

    # 5d. Per-Noise-Type Risk Report (R3)
    generate_risk_report(decay_df, clean_ref_df, os.path.join(OUTPUT_DIR, 'risk_reports'))
    
    print("✓ Report generation complete!")

if __name__ == '__main__': main()
