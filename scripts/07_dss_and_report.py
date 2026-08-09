#!/usr/bin/env python3
"""
Script 07 — DSS Computation & Report Generation
===============================================
Compute Deployment Safety Score, Sobol sensitivity, and generate decision artifacts.

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
from ecg_ssl_utils.score.dss import compute_dss
from ecg_ssl_utils.score.normalization import reference_anchored_normalize
from ecg_ssl_utils.score.sobol import sobol_sensitivity
from ecg_ssl_utils.report.decay_curves import plot_decay_curves
from ecg_ssl_utils.report.leaderboard import generate_leaderboard
from ecg_ssl_utils.report.flowchart import generate_decision_flowchart

def main():
    cfg = get_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'decay_atlas'), exist_ok=True)
    
    # In practice these would point to previous Kaggle outputs or local files
    decay_path = os.path.join(DECAY_DIR, 'metric_curves.parquet')
    deploy_path = os.path.join(DEPLOY_DIR, 'deployment_profiles.parquet')
    
    if not os.path.exists(decay_path) or not os.path.exists(deploy_path):
        print("Waiting for previous stages to complete (paths missing).")
        return
        
    decay_df = pd.read_parquet(decay_path)
    deploy_df = pd.read_parquet(deploy_path)
    
    # 1. Merge and Normalize
    # Use FP32 CPU latency for baseline
    latency = deploy_df[(deploy_df['precision']=='fp32') & (deploy_df['provider']=='CPUExecutionProvider')]
    latency = latency[['model_id', 'latency_p50', 'memory_mb']].rename(columns={'model_id': 'encoder'})
    
    merged = decay_df.merge(latency, on='encoder', how='left')
    
    # Compute Delta CKA and Delta ERank (assuming 24dB is baseline clean)
    clean_refs = merged[merged['snr_db'] == 24].set_index(['encoder', 'noise_type'])
    
    dss_rows = []
    for (enc, ntype, snr), group in merged.groupby(['encoder', 'noise_type', 'snr_db']):
        row = group.iloc[0]
        # Get baseline for delta
        try:
            baseline = clean_refs.loc[(enc, ntype)]
            delta_cka = 1.0 - row['cka'] # CKA is 1 for identical, lower is worse
            delta_er = 1.0 - (row['erank'] / baseline['erank'])
        except KeyError:
            delta_cka, delta_er = 0.0, 0.0
            
        dss_rows.append({
            'model_id': enc, 'noise_type': ntype, 'snr_db': snr,
            'delta_cka': delta_cka, 'delta_erank': delta_er,
            'ece': row['ece'], 'auroc': row['auroc'],
            'latency_p50': row['latency_p50'], 'memory_mb': row['memory_mb']
        })
        
    dss_df = pd.DataFrame(dss_rows)
    
    # Normalize globally
    for col in ['delta_cka', 'delta_erank', 'ece', 'latency_p50']:
        dss_df[f'{col}_norm'] = reference_anchored_normalize(dss_df[col].values)
        
    # 2. Compute DSS
    scores = []
    for _, row in dss_df.iterrows():
        dss, comps, passed = compute_dss(
            row['delta_cka_norm'], row['delta_erank_norm'], row['ece_norm'], row['latency_p50_norm'],
            auroc=row['auroc'], weights=cfg.dss.weights,
            ece_gate=cfg.dss.ece_gate, auroc_gate=cfg.dss.auroc_gate
        )
        scores.append(dss)
    dss_df['dss'] = scores
    dss_df.to_parquet(os.path.join(OUTPUT_DIR, 'dss_results.parquet'), index=False)
    
    # 3. Report Artifacts
    # Leaderboard (averaged over conditions for simplicity here, or worst-case)
    worst_case = dss_df[dss_df['snr_db'] == -6].groupby('model_id').mean().reset_index()
    generate_leaderboard(worst_case, deploy_df, os.path.join(OUTPUT_DIR, 'leaderboard.html'))
    
    # Decay Curves
    encs = dss_df['model_id'].unique()
    for ntype in dss_df['noise_type'].unique():
        data = {}
        for enc in encs:
            subset = decay_df[(decay_df['encoder']==enc) & (decay_df['noise_type']==ntype)].sort_values('snr_db', ascending=False)
            data[enc] = {'mean': subset['auroc'].values, 'ci_lo': subset['auroc_lo'].values, 'ci_hi': subset['auroc_hi'].values}
        plot_decay_curves(data, 'AUROC', ntype, subset['snr_db'].values, encs,
                          save_path=os.path.join(OUTPUT_DIR, 'decay_atlas', f'auroc_vs_snr_{ntype}.png'))
                          
    # Flowchart
    generate_decision_flowchart(worst_case, os.path.join(OUTPUT_DIR, 'decision_flowchart.md'))
    
    print("✓ Report generation complete!")

if __name__ == '__main__': main()
