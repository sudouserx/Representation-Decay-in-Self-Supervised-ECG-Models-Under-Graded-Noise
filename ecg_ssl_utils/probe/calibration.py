"""Post-hoc temperature scaling calibration (Guo et al., 2017)."""
import torch, torch.nn as nn
import numpy as np
from scipy.optimize import minimize_scalar


def temperature_scaling(logits_np, labels_np):
    """Find optimal temperature T* that minimizes NLL on validation set."""
    logits_t = torch.tensor(logits_np, dtype=torch.float32)
    labels_t = torch.tensor(labels_np, dtype=torch.float32)

    def nll(T):
        scaled = logits_t / T
        probs = torch.sigmoid(scaled)
        probs = probs.clamp(1e-7, 1-1e-7)
        loss = -labels_t * probs.log() - (1-labels_t) * (1-probs).log()
        return loss.mean().item()

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    return result.x  # optimal temperature
