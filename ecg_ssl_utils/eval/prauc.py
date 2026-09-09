"""Per-class and macro PR-AUC (average precision) for multi-label OvR."""

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score


def per_class_pr_auc(y_true, y_prob, min_positives=5, class_names=None) -> Dict[str, float]:
    """Return {class_name: pr_auc} plus 'macro' over classes with enough positives."""
    n_classes = y_true.shape[1]
    if class_names is None:
        class_names = [str(c) for c in range(n_classes)]
    out = {}
    vals = []
    for c in range(n_classes):
        name = class_names[c] if c < len(class_names) else str(c)
        if y_true[:, c].sum() >= min_positives:
            val = float(average_precision_score(y_true[:, c], y_prob[:, c]))
            out[name] = val
            vals.append(val)
        else:
            out[name] = float("nan")
    out["macro"] = float(np.nanmean(vals)) if vals else float("nan")
    return out


def per_class_brier(y_true, y_prob, class_names=None) -> Dict[str, float]:
    """Per-class Brier score (mean squared error of OvR sigmoid outputs)."""
    n_classes = y_true.shape[1]
    if class_names is None:
        class_names = [str(c) for c in range(n_classes)]
    out = {}
    vals = []
    for c in range(n_classes):
        name = class_names[c] if c < len(class_names) else str(c)
        val = float(np.mean((y_prob[:, c] - y_true[:, c]) ** 2))
        out[name] = val
        vals.append(val)
    out["macro"] = float(np.mean(vals)) if vals else float("nan")
    return out
