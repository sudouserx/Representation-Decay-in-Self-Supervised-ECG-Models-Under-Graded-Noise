"""
Per-class F1 score computation for multi-label classification.
Computes F1 for each superclass (NORM, MI, STTC, CD, HYP).
"""
import numpy as np
from typing import Dict, List, Optional, Sequence, Union


def per_class_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: Union[float, Sequence[float]] = 0.5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute per-class F1 score for multi-label classification.

    Parameters
    ----------
    y_true : np.ndarray
        Shape (N, C), binary ground truth.
    y_prob : np.ndarray
        Shape (N, C), predicted probabilities.
    threshold : float or sequence of float
        Decision threshold(s). A scalar is broadcast; a length-C vector
        uses a per-class validation-tuned threshold.
    class_names : list of str, optional
        Names for each class. If None, uses integer indices.

    Returns
    -------
    f1_dict : dict
        {class_name: f1_score} for each class, plus 'macro' key.
    """
    N, C = y_true.shape
    thr = np.asarray(threshold, dtype=float)
    if thr.ndim == 0:
        thr = np.full(C, float(thr))
    if thr.shape[0] != C:
        raise ValueError(f"threshold length {thr.shape[0]} != n_classes {C}")
    y_pred = (y_prob >= thr[None, :]).astype(float)

    if class_names is None:
        class_names = [str(c) for c in range(C)]

    f1_dict = {}
    f1_values = []

    for c in range(C):
        tp = ((y_pred[:, c] == 1) & (y_true[:, c] == 1)).sum()
        fp = ((y_pred[:, c] == 1) & (y_true[:, c] == 0)).sum()
        fn = ((y_pred[:, c] == 0) & (y_true[:, c] == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        name = class_names[c] if c < len(class_names) else str(c)
        f1_dict[name] = float(f1)
        f1_values.append(f1)

    f1_dict['macro'] = float(np.mean(f1_values)) if f1_values else 0.0
    return f1_dict
