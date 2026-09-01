"""
Expected Calibration Error (ECE).
ECE = Σ (|B_m|/n) · |acc(B_m) - conf(B_m)|
Reference: Guo et al., ICML 2017.

METHODOLOGY NOTE:
For multi-label classification (like the 5 diagnostic superclasses), this computes
binary one-vs-rest ECE using independent sigmoid probabilities per class.
Bins are equal-width in the [0,1] probability space.
"""
import numpy as np


def expected_calibration_error(y_true, y_prob, n_bins=15, return_per_class=False):
    """
    Compute ECE for multi-label classification.
    y_true: (N, C) binary, y_prob: (N, C) independent sigmoid probabilities.
    Returns float in [0, 1] if return_per_class is False,
    otherwise returns (macro_ece, {class_index: ece_value}).
    """
    N, C = y_true.shape
    ece_per_class = []
    ece_dict = {}
    for c in range(C):
        if y_true[:, c].sum() < 1:
            continue
        probs = y_prob[:, c]
        correct = (y_true[:, c] == (probs >= 0.5).astype(float)).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        ece_c = 0.0
        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i+1])
            if i == n_bins - 1:  # include right edge
                mask = (probs >= bins[i]) & (probs <= bins[i+1])
            if mask.sum() == 0:
                continue
            acc = correct[mask].mean()
            conf = probs[mask].mean()
            ece_c += mask.sum() / N * abs(acc - conf)
        ece_per_class.append(ece_c)
        ece_dict[str(c)] = ece_c
        
    macro_ece = float(np.mean(ece_per_class)) if ece_per_class else 0.0
    
    if return_per_class:
        return macro_ece, ece_dict
    return macro_ece
