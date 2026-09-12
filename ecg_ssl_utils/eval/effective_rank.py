"""
Effective Rank via spectral entropy of singular values.
erank(X) = exp(-Σ σ̄_i · ln(σ̄_i))  where σ̄_i = σ_i / Σ σ_j
Reference: Roy & Vetterli, 2007.
"""
import numpy as np

from .bootstrap import percentile_ci_from_boots


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
    return _erank_from_sv(sv)


def _erank_from_sv(sv):
    sv = np.asarray(sv, dtype=np.float64)
    sv = sv[sv > 1e-12]
    if sv.size == 0:
        return float("nan")
    sv_norm = sv / sv.sum()
    entropy = -np.sum(sv_norm * np.log(sv_norm + 1e-12))
    return float(np.exp(entropy))


def _try_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def _geometry_device(torch_mod):
    if torch_mod is None:
        return None
    return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")


def _subsample_idx(idx, subsample, seed):
    if subsample is not None and len(idx) > subsample:
        rng = np.random.RandomState(seed)
        return idx[rng.choice(len(idx), subsample, replace=False)]
    return idx


def bootstrap_mean_effective_rank(
    rep_stack,
    draws,
    n_components=64,
    seed=0,
    subsample=1500,
    point_estimate=None,
    alpha=0.05,
    chunk_size=16,
):
    """
    Patient-bootstrap CI of mean (across noise seeds) effective rank.

    Inner row-subsample uses ``RandomState(seed)`` on every replicate, matching
    the original estimator. Bootstrap CIs use batched top-``n_components``
    singular values (exact SVD on the subsampled rows). Point estimates in
    callers should keep ``effective_rank`` (randomized SVD).
    """
    torch = _try_torch()
    rep_stack = np.asarray(rep_stack, dtype=np.float32)
    if rep_stack.ndim == 2:
        rep_stack = rep_stack[None, ...]
    n_seed = rep_stack.shape[0]
    if point_estimate is None:
        point_estimate = float(np.nanmean([
            effective_rank(rep_stack[s], n_components=n_components, seed=seed)
            for s in range(n_seed)
        ]))

    if torch is None:
        boots = []
        for idx in draws:
            vals = []
            for s in range(n_seed):
                sub_idx = _subsample_idx(idx, subsample, seed)
                vals.append(effective_rank(
                    rep_stack[s][sub_idx], n_components=n_components, seed=seed,
                ))
            boots.append(float(np.nanmean(vals)))
        return percentile_ci_from_boots(np.asarray(boots), point_estimate, alpha)

    device = _geometry_device(torch)
    reps_t = torch.from_numpy(np.ascontiguousarray(rep_stack)).to(device)
    boots = np.empty(len(draws), dtype=np.float64)
    for start in range(0, len(draws), chunk_size):
        chunk = draws[start:start + chunk_size]
        seed_er = []
        for s in range(n_seed):
            gathered = []
            for idx in chunk:
                sub_idx = _subsample_idx(idx, subsample, seed)
                gathered.append(reps_t[s].index_select(
                    0, torch.as_tensor(sub_idx, device=device, dtype=torch.long),
                ))
            # Group by row count so SVD is never computed on zero-padded rows.
            lengths = [g.shape[0] for g in gathered]
            er_out = np.full(len(chunk), np.nan, dtype=np.float64)
            for L in sorted(set(lengths)):
                members = [i for i, n in enumerate(lengths) if n == L]
                mat = torch.stack([gathered[i] for i in members], dim=0)
                sv = torch.linalg.svdvals(mat)
                k = min(n_components, sv.shape[-1])
                sv_np = sv[:, :k].detach().cpu().numpy()
                for row, i in enumerate(members):
                    er_out[i] = _erank_from_sv(sv_np[row])
            seed_er.append(er_out)
        boots[start:start + len(chunk)] = np.nanmean(np.stack(seed_er, axis=0), axis=0)
    return percentile_ci_from_boots(boots, point_estimate, alpha)
