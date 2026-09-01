"""
Robustness Score (formerly Deployment Safety Score / DSS).

Robustness Score = weighted fusion of:
  - Performance decay (1 - ΔAUROC_norm)
  - Calibration degradation (1 - ΔECE_norm)
  - Representation decay: CKA (1 - ΔCKA_norm) + EffectiveRank (1 - ΔER_norm)

Non-compensatory gates on ECE and AUROC reject unsafe models.
Deployment cost (latency/memory) is a constraint filter in the decision
artifact, not a component of the robustness score itself.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass
class RobustnessResult:
    model_id: str
    noise_type: str
    snr_db: float
    score: float
    component_scores: Dict[str, float]
    weighting: Dict[str, float]
    uncertainty: Tuple[float, float]
    gate_passed: bool


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
        weights = {'cka': 0.25, 'erank': 0.25, 'ece': 0.25, 'auroc_decay': 0.25}

    # Components: higher = better (1 - decay)
    comps = {
        'cka': 1.0 - delta_cka,
        'erank': 1.0 - delta_erank,
        'ece': 1.0 - delta_ece,
        'auroc_decay': 1.0 - delta_auroc,
    }
    score = sum(weights[k] * comps[k] for k in weights)

    return score, comps
