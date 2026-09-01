"""Sobol sensitivity analysis for DSS weight stability."""
import numpy as np
from typing import Dict, Callable


def sobol_sensitivity(eval_fn, n_samples=1024, seed=42):
    """
    Simplified Sobol-like analysis: sample weight configs uniformly on simplex,
    evaluate DSS rankings, compute variance contribution per weight.
    eval_fn: takes weights dict → scalar (e.g., rank correlation).
    Returns {weight_name: sensitivity_index}.
    """
    rng = np.random.RandomState(seed)
    names = ['cka', 'erank', 'ece', 'auroc_decay']

    # Sample from Dirichlet (uniform on 4-simplex)
    samples = rng.dirichlet(np.ones(4), size=n_samples)
    results = []
    for s in samples:
        w = {n: float(s[i]) for i, n in enumerate(names)}
        results.append(eval_fn(w))
    results = np.array(results)
    total_var = np.var(results)
    if total_var < 1e-12:
        return {n: 0.25 for n in names}

    # First-order estimate: variance of conditional expectation
    sensitivity = {}
    for i, name in enumerate(names):
        # Bin the i-th weight into 10 bins
        bins = np.linspace(0, 1, 11)
        bin_means = []
        for b in range(10):
            mask = (samples[:, i] >= bins[b]) & (samples[:, i] < bins[b+1])
            if mask.sum() > 0:
                bin_means.append(results[mask].mean())
        var_cond = np.var(bin_means) if len(bin_means) > 1 else 0
        sensitivity[name] = float(var_cond / total_var)

    return sensitivity
