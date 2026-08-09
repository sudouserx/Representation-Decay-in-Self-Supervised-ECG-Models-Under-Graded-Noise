"""Patient-level bootstrap confidence intervals."""
import numpy as np
from typing import Callable, Tuple


def patient_bootstrap_ci(
    metric_fn: Callable,
    patient_ids: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    **metric_kwargs,
) -> Tuple[float, float, float]:
    """
    Bootstrap CI by resampling patients (not individual ECGs).
    metric_fn: function that takes indices array + **metric_kwargs → float.
    Returns (point_estimate, ci_lo, ci_hi).
    """
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)
    # Build patient→indices mapping
    p2idx = {}
    for i, pid in enumerate(patient_ids):
        p2idx.setdefault(pid, []).append(i)

    boots = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        idx = []
        for pid in sampled:
            idx.extend(p2idx[pid])
        idx = np.array(idx)
        boots.append(metric_fn(idx, **metric_kwargs))

    boots = np.array(boots)
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    point = np.mean(boots)
    return float(point), float(lo), float(hi)
