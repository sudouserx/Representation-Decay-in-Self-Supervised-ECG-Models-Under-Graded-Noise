#!/usr/bin/env python3
"""
Script 05 — Decay Metrics
=========================
Compute CKA, Effective Rank, ECE, AUROC (overall and subgroup), and DeLong tests.

Kaggle Inputs:  ptbxl-clean-processed, corruption-eval-results, linear-probes-all
Kaggle Output:  /kaggle/working/decay-metrics-results/
"""
import os, sys, json, glob
import numpy as np, pandas as pd
from collections import defaultdict
import multiprocessing as mp

CLEAN_DIR = os.environ.get('CLEAN_DIR', '/kaggle/input/ptbxl-clean-processed')
EVAL_DIR = os.environ.get('EVAL_DIR', '/kaggle/input/corruption-eval-results')
PROBE_DIR = os.environ.get('PROBE_DIR', '/kaggle/input/linear-probes-all')
UTILS_DIR = os.environ.get('UTILS_DIR', '/kaggle/input/ecg-ssl-utils')
OUTPUT_DIR = '/kaggle/working/decay-metrics-results'
if UTILS_DIR not in sys.path: sys.path.insert(0, UTILS_DIR)

from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.eval.cka import linear_cka
from ecg_ssl_utils.eval.effective_rank import effective_rank
from ecg_ssl_utils.eval.ece import expected_calibration_error
from ecg_ssl_utils.eval.auroc import macro_auroc, subgroup_auroc
from ecg_ssl_utils.eval.bootstrap import patient_bootstrap_ci
from ecg_ssl_utils.eval.delong import delong_test

def _compute_metrics(idx, reps, probs, clean_reps, labels):
    cka = linear_cka(clean_reps[idx], reps[idx])
    erank = effective_rank(reps[idx])
    ece = expected_calibration_error(labels[idx], probs[idx])
    auroc = macro_auroc(labels[idx], probs[idx])
    return cka, erank, ece, auroc

def process_condition(args):
    enc, ntype, snr, seed, rep_path, clean_reps, labels, pids = args
    data = np.load(rep_path)
    reps = data['representations'].astype(np.float32)
    probs = data['predictions']
    
    idx_all = np.arange(len(labels))
    cka, erank, ece, auroc = _compute_metrics(idx_all, reps, probs, clean_reps, labels)
    
    # Bootstrap
    _, cka_lo, cka_hi = patient_bootstrap_ci(
        lambda idx: linear_cka(clean_reps[idx], reps[idx]),
        pids, n_bootstrap=100
    )
    _, er_lo, er_hi = patient_bootstrap_ci(
        lambda idx: effective_rank(reps[idx]),
        pids, n_bootstrap=100
    )
    _, ece_lo, ece_hi = patient_bootstrap_ci(
        lambda idx: expected_calibration_error(labels[idx], probs[idx]),
        pids, n_bootstrap=100
    )
    _, auc_lo, auc_hi = patient_bootstrap_ci(
        lambda idx: macro_auroc(labels[idx], probs[idx]),
        pids, n_bootstrap=100
    )
    
    return {
        'encoder': enc, 'noise_type': ntype, 'snr_db': snr, 'seed': seed,
        'cka': cka, 'cka_lo': cka_lo, 'cka_hi': cka_hi,
        'erank': erank, 'erank_lo': er_lo, 'erank_hi': er_hi,
        'ece': ece, 'ece_lo': ece_lo, 'ece_hi': ece_hi,
        'auroc': auroc, 'auroc_lo': auc_lo, 'auroc_hi': auc_hi,
    }

def main():
    cfg = get_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    labels_test = np.load(os.path.join(CLEAN_DIR, 'labels_test.npy'))
    meta = pd.read_parquet(os.path.join(CLEAN_DIR, 'metadata.parquet'))
    test_meta = meta[meta['split'] == 'test'].reset_index(drop=True)
    pids = test_meta['patient_id'].values
    
    encoders = [os.path.basename(d) for d in glob.glob(os.path.join(EVAL_DIR, 'ssl-*'))]
    
    tasks = []
    for enc in encoders:
        clean_reps = np.load(os.path.join(PROBE_DIR, enc, 'clean_test_repr.npy'))
        enc_dir = os.path.join(EVAL_DIR, enc)
        for cond_dir in os.listdir(enc_dir):
            if not os.path.isdir(os.path.join(enc_dir, cond_dir)): continue
            parts = cond_dir.split('_')
            seed = int(parts[-1])
            snr = float(parts[-2].replace('db',''))
            ntype = '_'.join(parts[:-2])
            rep_path = os.path.join(enc_dir, cond_dir, 'results.npz')
            tasks.append((enc, ntype, snr, seed, rep_path, clean_reps, labels_test, pids))
            
    print(f"Processing {len(tasks)} conditions...")
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_condition, tasks)
        
    df = pd.DataFrame(results)
    # Average over seeds
    metrics_cols = ['cka', 'cka_lo', 'cka_hi', 'erank', 'erank_lo', 'erank_hi', 
                    'ece', 'ece_lo', 'ece_hi', 'auroc', 'auroc_lo', 'auroc_hi']
    df_agg = df.groupby(['encoder', 'noise_type', 'snr_db'])[metrics_cols].mean().reset_index()
    
    df_agg.to_parquet(os.path.join(OUTPUT_DIR, 'metric_curves.parquet'), index=False)
    print("✓ Decay metrics computed!")

if __name__ == '__main__': main()
