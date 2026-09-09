"""
Robustness index (secondary, within-run ranking convenience).

R = Σ w_k (1 − Δ_k) with pre-registered weights
(default: AUROC 0.30, ECE 0.30, CKA 0.20, ER 0.20).

ΔER is two-sided at the call site: |1 − ER_noisy/ER_clean|.
Gates are decision-support filters only — they do not zero this index.
Deployment cost is a constraint filter, not a score component.
"""
import numpy as np
from typing import Dict, Tuple, Optional


def compute_robustness_score(
    delta_cka: float,
    delta_erank: float,
    delta_ece: float,
    delta_auroc: float,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute Robustness Score from normalized component scores.

    All delta inputs should be pre-normalized to [0,1].
    Higher delta = more decay/degradation (bad).

    Parameters
    ----------
    delta_cka : float
        Normalized CKA decay (1 - CKA).
    delta_erank : float
        Normalized effective rank decay (1 - ER_noisy/ER_clean).
    delta_ece : float
        Normalized ECE increase (higher = worse calibration).
    delta_auroc : float
        Normalized AUROC decrease (higher = worse performance).
    weights : dict, optional
        Component weights. Keys: 'cka', 'erank', 'ece', 'auroc_decay'.
        Must sum to 1.0.

    Returns
    -------
    (score, components) : tuple
        score: float in [0, 1], higher = more robust.
        components: dict of per-axis scores (higher = better).
    """
    if weights is None:
        weights = {'cka': 0.20, 'erank': 0.20, 'ece': 0.30, 'auroc_decay': 0.30}

    # Components: higher = better (1 - decay)
    comps = {
        'cka': 1.0 - delta_cka,
        'erank': 1.0 - delta_erank,
        'ece': 1.0 - delta_ece,
        'auroc_decay': 1.0 - delta_auroc,
    }
    score = sum(weights[k] * comps[k] for k in weights)

    return score, comps
