"""AUROC computation: macro and subgroup."""
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Optional


def macro_auroc(y_true, y_prob, min_positives=5):
    """Macro AUROC, skipping classes with < min_positives."""
    N, C = y_true.shape
    aurocs = []
    for c in range(C):
        if y_true[:, c].sum() >= min_positives and (1-y_true[:, c]).sum() >= 1:
            aurocs.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
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
