"""
Linear CKA (Centered Kernel Alignment).
Measures similarity between clean and noisy representations.
CKA_linear(X,Y) = ‖Y'X‖²_F / (‖X'X‖_F · ‖Y'Y‖_F)
Reference: Kornblith et al., ICML 2019.
"""
import numpy as np


def _center(X):
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X, Y):
    """
    Compute linear CKA between two representation matrices.
    X, Y: np.ndarray of shape (n_samples, n_features).
    Returns float in [0, 1].
    """
    X, Y = _center(X), _center(Y)
    YtX = Y.T @ X
    XtX = X.T @ X
    YtY = Y.T @ Y
    num = np.linalg.norm(YtX, 'fro') ** 2
    denom = np.linalg.norm(XtX, 'fro') * np.linalg.norm(YtY, 'fro')
    return float(num / (denom + 1e-12))
