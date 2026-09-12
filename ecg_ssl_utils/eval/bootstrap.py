"""Patient-level bootstrap confidence intervals and paired tests."""
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm


_DRAW_CACHE = {}


def _patient_index_map(patient_ids: np.ndarray):
    unique_patients = np.unique(patient_ids)
    p2idx = {}
    for i, pid in enumerate(patient_ids):
        p2idx.setdefault(pid, []).append(i)
    for pid in list(p2idx):
        p2idx[pid] = np.asarray(p2idx[pid], dtype=np.int64)
    return unique_patients, p2idx


def _resample_indices(rng, unique_patients, p2idx):
    sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
    return np.concatenate([p2idx[pid] for pid in sampled])


def _cache_key(patient_ids: np.ndarray, n_bootstrap: int, seed: int):
    ids = np.asarray(patient_ids)
    return (
        int(n_bootstrap),
        int(seed),
        ids.shape,
        ids.dtype.str,
        int(ids[0]) if ids.size else 0,
        int(ids[-1]) if ids.size else 0,
        hash(ids.tobytes()),
    )


def make_patient_bootstrap_draws(
    patient_ids: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    use_cache: bool = True,
) -> List[np.ndarray]:
    """
    Patient-clustered index draws.

    Uses the same RNG walk as a fresh ``RandomState(seed)`` loop of
    ``_resample_indices``, so caching is statistically identical to calling
    ``patient_bootstrap_ci`` independently with the same seed.
    """
    key = _cache_key(patient_ids, n_bootstrap, seed)
    if use_cache and key in _DRAW_CACHE:
        return _DRAW_CACHE[key]

    rng = np.random.RandomState(seed)
    unique_patients, p2idx = _patient_index_map(patient_ids)
    draws = [
        _resample_indices(rng, unique_patients, p2idx)
        for _ in range(n_bootstrap)
    ]
    if use_cache:
        _DRAW_CACHE[key] = draws
    return draws


def clear_bootstrap_draw_cache():
    _DRAW_CACHE.clear()


def patient_bootstrap_ci(
    metric_fn: Callable,
    patient_ids: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    point_estimate: Optional[float] = None,
    draws: Optional[Sequence[np.ndarray]] = None,
    **metric_kwargs,
) -> Tuple[float, float, float]:
    """
    Bootstrap CI by resampling patients (not individual ECGs).

    *point* is the full-sample metric (or *point_estimate* if provided),
    not the mean of the bootstrap distribution.
    """
    if point_estimate is None:
        point = float(metric_fn(np.arange(len(patient_ids)), **metric_kwargs))
    else:
        point = float(point_estimate)

    if draws is None:
        draws = make_patient_bootstrap_draws(patient_ids, n_bootstrap, seed)
    else:
        draws = draws[:n_bootstrap]

    boots = np.asarray(
        [metric_fn(idx, **metric_kwargs) for idx in draws],
        dtype=float,
    )
    lo = np.nanpercentile(boots, 100 * alpha / 2)
    hi = np.nanpercentile(boots, 100 * (1 - alpha / 2))
    return point, float(lo), float(hi)


def patient_bootstrap_pvalue(
    delta_fn: Callable,
    patient_ids: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    draws: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[float, float]:
    """
    Two-sided patient-clustered bootstrap p-value for a paired delta.

    *delta_fn(idx)* returns Δ on the resampled records (e.g. AUROC_clean - AUROC_noisy).
    p = 2 * min(frac(Δ* ≤ 0), frac(Δ* ≥ 0)), using the bootstrap distribution of Δ
    (shift / percentile method). Returns (observed_delta, p_two_sided).
    """
    observed = float(delta_fn(np.arange(len(patient_ids))))
    if draws is None:
        draws = make_patient_bootstrap_draws(patient_ids, n_bootstrap, seed)
    else:
        draws = draws[:n_bootstrap]
    boots = np.asarray([float(delta_fn(idx)) for idx in draws], dtype=float)
    frac_le = float(np.mean(boots <= 0))
    frac_ge = float(np.mean(boots >= 0))
    p = 2.0 * min(frac_le, frac_ge)
    p = min(1.0, max(p, 1.0 / (n_bootstrap + 1)))
    return observed, p


def percentile_ci_from_boots(
    boots: np.ndarray,
    point: float,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    boots = np.asarray(boots, dtype=float)
    lo = np.nanpercentile(boots, 100 * alpha / 2)
    hi = np.nanpercentile(boots, 100 * (1 - alpha / 2))
    return float(point), float(lo), float(hi)


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
