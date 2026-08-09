"""
Effective Rank via spectral entropy of singular values.
erank(X) = exp(-Σ σ̄_i · ln(σ̄_i))  where σ̄_i = σ_i / Σ σ_j
Reference: Roy & Vetterli, 2007.
"""
import numpy as np


def effective_rank(X):
    """
    Compute effective rank of a representation matrix.
    X: np.ndarray of shape (n_samples, n_features).
    Returns float >= 1.
    """
    sv = np.linalg.svd(X, compute_uv=False)
    sv = sv[sv > 1e-12]  # filter near-zero
    sv_norm = sv / sv.sum()
    entropy = -np.sum(sv_norm * np.log(sv_norm + 1e-12))
    return float(np.exp(entropy))
