"""
Expected Calibration Error (ECE).
ECE = Σ (|B_m|/n) · |acc(B_m) - conf(B_m)|
Reference: Guo et al., ICML 2017.

OvR sigmoid outputs, 15-bin default. Confidence is the probability assigned
to the predicted binary class, ``max(p, 1-p)``. Equal-width is primary;
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


def _ece_equal_width_bincount(probs, correct, n_bins, N):
    """Equal-width bins: [0, 1/n), …, [(n-1)/n, 1], matching ``_ece_from_bins``."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(probs, edges[1:-1], right=False)
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    sum_correct = np.bincount(idx, weights=correct, minlength=n_bins)
    sum_conf = np.bincount(idx, weights=probs, minlength=n_bins)
    acc = np.divide(sum_correct, counts, out=np.zeros(n_bins), where=counts > 0)
    conf = np.divide(sum_conf, counts, out=np.zeros(n_bins), where=counts > 0)
    return float(np.sum((counts / N) * np.abs(acc - conf)))


def expected_calibration_error(
    y_true, y_prob, n_bins=15, return_per_class=False, binning="equal_width",
):
    """
    Multi-label OvR ECE.

    binning: 'equal_width' (primary) or 'equal_mass' (quantile bins).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    N, C = y_true.shape
    ece_per_class = []
    ece_dict = {}
    for c in range(C):
        if y_true[:, c].sum() < 1:
            continue
        positive_probs = y_prob[:, c]
        predictions = (positive_probs >= 0.5).astype(float)
        correct = (y_true[:, c] == predictions).astype(float)
        probs = np.maximum(positive_probs, 1.0 - positive_probs)
        if binning == "equal_mass":
            quantiles = np.linspace(0, 1, n_bins + 1)
            edges = np.unique(np.quantile(probs, quantiles))
            if len(edges) < 2:
                edges = np.array([0.0, 1.0])
            ece_c = _ece_from_bins(probs, correct, edges, N)
        else:
            ece_c = _ece_equal_width_bincount(probs, correct, n_bins, N)
        ece_per_class.append(ece_c)
        ece_dict[str(c)] = ece_c

    macro_ece = float(np.mean(ece_per_class)) if ece_per_class else 0.0
    if return_per_class:
        return macro_ece, ece_dict
    return macro_ece
