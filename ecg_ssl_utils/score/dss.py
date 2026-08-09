"""
Deployment Safety Score (DSS).
DSS = α(1-ΔCKA_n) + β(1-ΔER_n) + γ(1-ECE_n) + δ(1-Latency_n)
Higher = safer. Non-compensatory gates on ECE and AUROC.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass
class DSSResult:
    model_id: str
    noise_type: str
    snr_db: float
    dss: float
    component_scores: Dict[str, float]
    weighting: Dict[str, float]
    uncertainty: Tuple[float, float]
    gate_passed: bool


def compute_dss(delta_cka, delta_er, ece, latency, auroc=None,
                weights=None, ece_gate=0.15, auroc_gate=0.70):
    """
    Compute DSS from normalized component scores.
    All inputs should be pre-normalized to [0,1].
    delta_cka, delta_er: higher = more decay (bad).
    ece: higher = worse calibration.
    latency: higher = slower.
    Returns (dss_score, components, gate_passed).
    """
    if weights is None:
        weights = {'cka': 0.25, 'erank': 0.25, 'ece': 0.25, 'latency': 0.25}

    # Non-compensatory gates
    gate_passed = True
    if ece > ece_gate:
        gate_passed = False
    if auroc is not None and auroc < auroc_gate:
        gate_passed = False

    # DSS: higher = better
    comps = {
        'cka': 1.0 - delta_cka,
        'erank': 1.0 - delta_er,
        'ece': 1.0 - ece,
        'latency': 1.0 - latency,
    }
    dss = sum(weights[k] * comps[k] for k in weights)

    if not gate_passed:
        dss = 0.0

    return dss, comps, gate_passed
