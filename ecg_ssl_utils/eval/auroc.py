"""AUROC computation: macro and subgroup."""
from typing import Dict, Optional

import numpy as np


def binary_auroc(y_true, y_score):
    """
    Binary AUROC via Mann–Whitney U with average ranks (ties).

    Matches ``sklearn.metrics.roc_auc_score`` for binary inputs, including
    tied scores (trapezoidal ROC / midranks).
    """
    y = np.asarray(y_true).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    n_pos = float(np.sum(y == 1))
    n_neg = float(y.size - n_pos)
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    ranks = _average_ranks(s)
    sum_pos = float(ranks[y == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def _average_ranks(x: np.ndarray) -> np.ndarray:
    sorter = np.argsort(x, kind="mergesort")
    x_sorted = x[sorter]
    n = x.size
    obs = np.empty(n, dtype=bool)
    obs[0] = True
    if n > 1:
        obs[1:] = x_sorted[1:] != x_sorted[:-1]
    dense = np.cumsum(obs)
    count = np.bincount(dense)[1:]
    min_rank = np.cumsum(np.concatenate(([0], count[:-1])))
    mid = min_rank + (count + 1) * 0.5
    ranks_sorted = np.repeat(mid, count)
    ranks = np.empty(n, dtype=np.float64)
    ranks[sorter] = ranks_sorted
    return ranks


def macro_auroc(y_true, y_prob, min_positives=5):
    """Macro AUROC, skipping classes with < min_positives."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    C = y_true.shape[1]
    aurocs = []
    for c in range(C):
        col = y_true[:, c]
        if col.sum() >= min_positives and (1 - col).sum() >= 1:
            aurocs.append(binary_auroc(col, y_prob[:, c]))
    return float(np.mean(aurocs)) if aurocs else 0.5


def subgroup_auroc(y_true, y_prob, group_labels, min_positives=5):
    """Compute macro AUROC per subgroup. Returns {group_value: auroc}."""
    groups = np.unique(group_labels)
    result = {}
    for g in groups:
        mask = group_labels == g
        if mask.sum() < 10:
            continue
        result[str(g)] = macro_auroc(y_true[mask], y_prob[mask], min_positives)
    return result
