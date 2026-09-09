"""
Expected Calibration Error (ECE).
ECE = Σ (|B_m|/n) · |acc(B_m) - conf(B_m)|
Reference: Guo et al., ICML 2017.

OvR sigmoid outputs, 15-bin default. Accuracy inside a bin uses a 0.5
threshold (threshold-dependent; documented). Equal-width is primary;
equal-mass (quantile) bins are an optional robustness check.
"""
import numpy as np


def _ece_from_bins(probs, correct, edges, N):
    ece_c = 0.0
    n_bins = len(edges) - 1
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (probs >= edges[i]) & (probs <= edges[i + 1])
        else:
            mask = (probs >= edges[i]) & (probs < edges[i + 1])
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = probs[mask].mean()
        ece_c += mask.sum() / N * abs(acc - conf)
    return ece_c


def expected_calibration_error(
    y_true, y_prob, n_bins=15, return_per_class=False, binning="equal_width",
):
    """
    Multi-label OvR ECE.

    binning: 'equal_width' (primary) or 'equal_mass' (quantile bins).
    """
    N, C = y_true.shape
    ece_per_class = []
    ece_dict = {}
    for c in range(C):
        if y_true[:, c].sum() < 1:
            continue
        probs = y_prob[:, c]
        correct = (y_true[:, c] == (probs >= 0.5).astype(float)).astype(float)
        if binning == "equal_mass":
            quantiles = np.linspace(0, 1, n_bins + 1)
            edges = np.unique(np.quantile(probs, quantiles))
            if len(edges) < 2:
                edges = np.array([0.0, 1.0])
        else:
            edges = np.linspace(0, 1, n_bins + 1)
        ece_c = _ece_from_bins(probs, correct, edges, N)
        ece_per_class.append(ece_c)
        ece_dict[str(c)] = ece_c

    macro_ece = float(np.mean(ece_per_class)) if ece_per_class else 0.0
    if return_per_class:
        return macro_ece, ece_dict
    return macro_ece
