"""Patient-level bootstrap confidence intervals and paired tests."""
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.stats import norm


def _patient_index_map(patient_ids: np.ndarray):
    unique_patients = np.unique(patient_ids)
    p2idx = {}
    for i, pid in enumerate(patient_ids):
        p2idx.setdefault(pid, []).append(i)
    return unique_patients, p2idx


def _resample_indices(rng, unique_patients, p2idx):
    sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
    idx = []
    for pid in sampled:
        idx.extend(p2idx[pid])
    return np.array(idx)


def patient_bootstrap_ci(
    metric_fn: Callable,
    patient_ids: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    point_estimate: Optional[float] = None,
    **metric_kwargs,
) -> Tuple[float, float, float]:
    """
    Bootstrap CI by resampling patients (not individual ECGs).

    *point* is the full-sample metric (or *point_estimate* if provided),
    not the mean of the bootstrap distribution.
    """
    rng = np.random.RandomState(seed)
    unique_patients, p2idx = _patient_index_map(patient_ids)

    if point_estimate is None:
        point = float(metric_fn(np.arange(len(patient_ids)), **metric_kwargs))
    else:
        point = float(point_estimate)

    boots = []
    for _ in range(n_bootstrap):
        idx = _resample_indices(rng, unique_patients, p2idx)
        boots.append(metric_fn(idx, **metric_kwargs))

    boots = np.asarray(boots, dtype=float)
    lo = np.nanpercentile(boots, 100 * alpha / 2)
    hi = np.nanpercentile(boots, 100 * (1 - alpha / 2))
    return point, float(lo), float(hi)


def patient_bootstrap_pvalue(
    delta_fn: Callable,
    patient_ids: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Two-sided patient-clustered bootstrap p-value for a paired delta.

    *delta_fn(idx)* returns Δ on the resampled records (e.g. AUROC_clean - AUROC_noisy).
    p = 2 * min(frac(Δ* ≤ 0), frac(Δ* ≥ 0)), using the bootstrap distribution of Δ
    (shift / percentile method). Returns (observed_delta, p_two_sided).
    """
    rng = np.random.RandomState(seed)
    unique_patients, p2idx = _patient_index_map(patient_ids)
    observed = float(delta_fn(np.arange(len(patient_ids))))
    boots = []
    for _ in range(n_bootstrap):
        idx = _resample_indices(rng, unique_patients, p2idx)
        boots.append(float(delta_fn(idx)))
    boots = np.asarray(boots, dtype=float)
    frac_le = float(np.mean(boots <= 0))
    frac_ge = float(np.mean(boots >= 0))
    p = 2.0 * min(frac_le, frac_ge)
    p = min(1.0, max(p, 1.0 / (n_bootstrap + 1)))
    return observed, p


def stouffer_combine(p_values, two_sided=True):
    """Combine independent p-values via Stouffer: z = Σ z_s / √k."""
    p = np.clip(np.asarray(p_values, dtype=float), 1e-12, 1 - 1e-12)
    if two_sided:
        z = norm.isf(p / 2.0)
    else:
        z = norm.isf(p)
    z_comb = float(np.sum(z) / np.sqrt(len(z)))
    if two_sided:
        p_comb = float(2.0 * norm.sf(abs(z_comb)))
    else:
        p_comb = float(norm.sf(z_comb))
    return z_comb, p_comb
