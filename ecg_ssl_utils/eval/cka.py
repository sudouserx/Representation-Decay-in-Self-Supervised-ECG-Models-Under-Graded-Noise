"""
Linear CKA (Centered Kernel Alignment).
Measures similarity between clean and noisy representations.
CKA_linear(X,Y) = ‖Y'X‖²_F / (‖X'X‖_F · ‖Y'Y‖_F)
Reference: Kornblith et al., ICML 2019.
"""
import numpy as np


def _center(X):
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X, Y, var_guard=1e-6, return_collapse_flag=False):
    """
    Compute linear CKA between two representation matrices.
    X, Y: np.ndarray of shape (n_samples, n_features).
    Returns float in [0, 1], or nan if either matrix looks collapsed
    (mean column-std below *var_guard*).
    """
    x_std = float(X.std(axis=0).mean())
    y_std = float(Y.std(axis=0).mean())
    collapsed = x_std < var_guard or y_std < var_guard
    if collapsed:
        return (float("nan"), True) if return_collapse_flag else float("nan")
    X, Y = _center(X), _center(Y)
    YtX = Y.T @ X
    XtX = X.T @ X
    YtY = Y.T @ Y
    num = np.linalg.norm(YtX, "fro") ** 2
    denom = np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro")
    val = float(num / (denom + 1e-12))
    return (val, False) if return_collapse_flag else val
