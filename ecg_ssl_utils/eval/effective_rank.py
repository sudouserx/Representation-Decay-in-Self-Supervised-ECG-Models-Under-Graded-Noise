"""
Effective Rank via spectral entropy of singular values.
erank(X) = exp(-Σ σ̄_i · ln(σ̄_i))  where σ̄_i = σ_i / Σ σ_j
Reference: Roy & Vetterli, 2007.
"""
import numpy as np


def effective_rank(X, n_components=None, seed=0):
    """
    Compute effective rank of a representation matrix.
    X: np.ndarray of shape (n_samples, n_features).
    If *n_components* is set, use randomized SVD (bootstrap-friendly).
    Returns float >= 1.
    """
    if n_components is not None and n_components < min(X.shape):
        from sklearn.utils.extmath import randomized_svd
        _, sv, _ = randomized_svd(X, n_components=n_components, random_state=seed)
    else:
        sv = np.linalg.svd(X, compute_uv=False)
    sv = sv[sv > 1e-12]
    if sv.size == 0:
        return float("nan")
    sv_norm = sv / sv.sum()
    entropy = -np.sum(sv_norm * np.log(sv_norm + 1e-12))
    return float(np.exp(entropy))
