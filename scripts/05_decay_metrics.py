#!/usr/bin/env python3
"""
Script 05 — Decay Metrics
=========================
Compute CKA, Effective Rank, ECE, AUROC, per-class F1, and DeLong tests.
All metrics are computed as paired clean-vs-noisy comparisons per the methodology.

Kaggle Inputs:  ptbxl-clean-processed, corruption-eval-results, linear-probes-all
Kaggle Output:  /kaggle/working/decay-metrics-results/
"""
import os, sys, json, glob
import numpy as np, pandas as pd
from collections import defaultdict
import multiprocessing as mp
from scipy.stats import false_discovery_rate

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
from ecg_ssl_utils.eval.f1 import per_class_f1
from ecg_ssl_utils.eval.bootstrap import patient_bootstrap_ci
from ecg_ssl_utils.eval.delong import delong_test


def _load_labels(clean_dir):
    """Load superclass labels if available, otherwise fall back to SCP labels."""
    sc_test = os.path.join(clean_dir, 'superclass_labels_test.npy')
    if os.path.exists(sc_test):
        labels = np.load(sc_test)
        class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        print(f"  Using superclass labels ({labels.shape[1]} classes)")
    else:
        labels = np.load(os.path.join(clean_dir, 'labels_test.npy'))
        class_names = None
        print(f"  Using SCP code labels ({labels.shape[1]} classes)")
    return labels, class_names


def compute_clean_reference(enc, clean_reps, clean_preds, labels, pids, cfg):
    """Compute clean baseline metrics for a single encoder."""
    idx_all = np.arange(len(labels))
    clean_auroc = macro_auroc(labels, clean_preds)
    clean_ece = expected_calibration_error(labels, clean_preds, n_bins=cfg.eval.ece_bins)
    clean_erank = effective_rank(clean_reps)
    clean_f1 = per_class_f1(labels, clean_preds)

    # Bootstrap CIs for clean metrics
    _, auroc_lo, auroc_hi = patient_bootstrap_ci(
        lambda idx: macro_auroc(labels[idx], clean_preds[idx]),
        pids, n_bootstrap=cfg.eval.bootstrap_n,
    )
    _, ece_lo, ece_hi = patient_bootstrap_ci(
        lambda idx: expected_calibration_error(labels[idx], clean_preds[idx]),
        pids, n_bootstrap=cfg.eval.bootstrap_n,
    )
    _, er_lo, er_hi = patient_bootstrap_ci(
        lambda idx: effective_rank(clean_reps[idx]),
        pids, n_bootstrap=cfg.eval.bootstrap_n,
    )

    return {
        'encoder': enc,
        'auroc': clean_auroc, 'auroc_lo': auroc_lo, 'auroc_hi': auroc_hi,
        'ece': clean_ece, 'ece_lo': ece_lo, 'ece_hi': ece_hi,
        'erank': clean_erank, 'erank_lo': er_lo, 'erank_hi': er_hi,
        'f1_macro': clean_f1['macro'],
        **{f'f1_{k}': v for k, v in clean_f1.items() if k != 'macro'},
    }


def process_condition(args):
    """Process a single (encoder, noise_type, snr, seed) condition."""
    enc, ntype, snr, seed, rep_path, clean_reps, clean_preds, labels, pids, cfg, class_names = args
    data = np.load(rep_path)
    reps = data['representations'].astype(np.float32)
    probs = data['predictions']

    idx_all = np.arange(len(labels))

    # Core metrics
    cka = linear_cka(clean_reps, reps)
    erank = effective_rank(reps)
    ece = expected_calibration_error(labels, probs, n_bins=cfg.eval.ece_bins)
    auroc = macro_auroc(labels, probs)
    f1 = per_class_f1(labels, probs, class_names=class_names)

    # Bootstrap CIs (using methodology-specified 1000 resamples)
    _, cka_lo, cka_hi = patient_bootstrap_ci(
        lambda idx: linear_cka(clean_reps[idx], reps[idx]),
        pids, n_bootstrap=cfg.eval.bootstrap_n,
    )
    _, er_lo, er_hi = patient_bootstrap_ci(
        lambda idx: effective_rank(reps[idx]),
        pids, n_bootstrap=cfg.eval.bootstrap_n,
    )
    _, ece_lo, ece_hi = patient_bootstrap_ci(
        lambda idx: expected_calibration_error(labels[idx], probs[idx]),
        pids, n_bootstrap=cfg.eval.bootstrap_n,
    )
    _, auc_lo, auc_hi = patient_bootstrap_ci(
        lambda idx: macro_auroc(labels[idx], probs[idx]),
        pids, n_bootstrap=cfg.eval.bootstrap_n,
    )

    # DeLong test: clean vs noisy AUROC per class (paired comparison)
    # METHODOLOGY NOTE: These p-values are reported per-condition without 
    # multiple-comparison adjustment. They serve as descriptive indicators of 
    # statistically detectable AUROC degradation, not as controlled family-wise 
    # error rates. Benjamini-Hochberg correction is applied across conditions later.
    delong_results = {}
    n_classes = labels.shape[1]
    for c in range(n_classes):
        if labels[:, c].sum() >= cfg.eval.min_class_positives and (1 - labels[:, c]).sum() >= 1:
            try:
                z, p, auc_clean, auc_noisy = delong_test(
                    labels[:, c], clean_preds[:, c], probs[:, c]
                )
                c_name = class_names[c] if class_names and c < len(class_names) else str(c)
                delong_results[c_name] = {'z': z, 'p': p, 'auc_clean': auc_clean, 'auc_noisy': auc_noisy}
            except Exception:
                pass

    # Aggregate DeLong: report min p-value and whether any class is significant
    delong_min_p = min([d['p'] for d in delong_results.values()]) if delong_results else 1.0
    delong_n_sig = sum(1 for d in delong_results.values() if d['p'] < cfg.eval.delong_alpha)

    result = {
        'encoder': enc, 'noise_type': ntype, 'snr_db': snr, 'seed': seed,
        'cka': cka, 'cka_lo': cka_lo, 'cka_hi': cka_hi,
        'erank': erank, 'erank_lo': er_lo, 'erank_hi': er_hi,
        'ece': ece, 'ece_lo': ece_lo, 'ece_hi': ece_hi,
        'auroc': auroc, 'auroc_lo': auc_lo, 'auroc_hi': auc_hi,
        'f1_macro': f1['macro'],
        'delong_min_p': delong_min_p,
        'delong_n_significant': delong_n_sig,
    }
    # Add per-class F1
    for k, v in f1.items():
        if k != 'macro':
            result[f'f1_{k}'] = v

    return result


def main():
    cfg = get_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    labels_test, class_names = _load_labels(CLEAN_DIR)
    meta = pd.read_parquet(os.path.join(CLEAN_DIR, 'metadata.parquet'))
    test_meta = meta[meta['split'] == 'test'].reset_index(drop=True)
    pids = test_meta['patient_id'].values

    encoders = [os.path.basename(d) for d in glob.glob(os.path.join(EVAL_DIR, 'ssl-*'))]

    # ── Compute clean reference metrics for each encoder ──
    print("Computing clean reference metrics...")
    clean_refs = []
    for enc in encoders:
        clean_reps = np.load(os.path.join(PROBE_DIR, enc, 'clean_test_repr.npy'))
        clean_preds_path = os.path.join(PROBE_DIR, enc, 'clean_test_predictions.npy')
        if os.path.exists(clean_preds_path):
            clean_preds = np.load(clean_preds_path)
        else:
            print(f"  WARNING: clean_test_predictions.npy not found for {enc}, skipping clean ref")
            continue
        ref = compute_clean_reference(enc, clean_reps, clean_preds, labels_test, pids, cfg)
        clean_refs.append(ref)
        print(f"  {enc}: AUROC={ref['auroc']:.4f}, ECE={ref['ece']:.4f}, "
              f"ER={ref['erank']:.1f}, F1={ref['f1_macro']:.4f}")

    clean_ref_df = pd.DataFrame(clean_refs)
    clean_ref_df.to_parquet(os.path.join(OUTPUT_DIR, 'clean_reference.parquet'), index=False)

    # ── Compute noisy condition metrics ──
    print("\nComputing noisy condition metrics...")
    tasks = []
    for enc in encoders:
        clean_reps = np.load(os.path.join(PROBE_DIR, enc, 'clean_test_repr.npy'))
        clean_preds_path = os.path.join(PROBE_DIR, enc, 'clean_test_predictions.npy')
        if not os.path.exists(clean_preds_path):
            continue
        clean_preds = np.load(clean_preds_path)

        enc_dir = os.path.join(EVAL_DIR, enc)
        for cond_dir in os.listdir(enc_dir):
            if not os.path.isdir(os.path.join(enc_dir, cond_dir)):
                continue
            parts = cond_dir.split('_')
            seed = int(parts[-1])
            snr = float(parts[-2].replace('db', ''))
            ntype = '_'.join(parts[:-2])
            rep_path = os.path.join(enc_dir, cond_dir, 'results.npz')
            tasks.append((enc, ntype, snr, seed, rep_path,
                          clean_reps, clean_preds, labels_test, pids, cfg, class_names))

    print(f"Processing {len(tasks)} conditions...")
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_condition, tasks)

    df = pd.DataFrame(results)

    # Average over seeds
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    group_cols = ['encoder', 'noise_type', 'snr_db']
    agg_cols = [c for c in numeric_cols if c not in group_cols and c != 'seed']
    df_agg = df.groupby(group_cols)[agg_cols].mean().reset_index()

    # Apply BH correction for DeLong p-values globally across all aggregated conditions
    if 'delong_min_p' in df_agg.columns:
        p_values = df_agg['delong_min_p'].values
        # Using scipy.stats.false_discovery_rate (or multipletests from statsmodels if available)
        # We will use multipletests if statsmodels is present, otherwise fallback
        try:
            from statsmodels.stats.multitest import multipletests
            _, bh_pvals, _, _ = multipletests(p_values, method='fdr_bh')
        except ImportError:
            # Fallback to simple BH implementation if statsmodels is missing
            order = np.argsort(p_values)
            n_tests = len(p_values)
            bh_pvals = np.zeros(n_tests)
            for i, p_idx in enumerate(order):
                bh_pvals[p_idx] = p_values[p_idx] * n_tests / (i + 1)
            bh_pvals = np.minimum(bh_pvals, 1.0)
            
        df_agg['delong_min_p_bh_adj'] = bh_pvals

    df.to_parquet(os.path.join(OUTPUT_DIR, 'metric_curves_per_seed.parquet'), index=False)
    df_agg.to_parquet(os.path.join(OUTPUT_DIR, 'metric_curves.parquet'), index=False)
    print("✓ Decay metrics computed!")

if __name__ == '__main__': main()
