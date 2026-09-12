"""
Linear CKA (Centered Kernel Alignment).
Measures similarity between clean and noisy representations.
CKA_linear(X,Y) = ‖Y'X‖²_F / (‖X'X‖_F · ‖Y'Y‖_F)
Reference: Kornblith et al., ICML 2019.
"""
import numpy as np

from .bootstrap import percentile_ci_from_boots


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


def _cka_from_padded(X, Y, mask, var_guard):
    """Batched linear CKA. X,Y: (B, N, D), mask: (B, N) bool."""
    import torch
    w = mask.unsqueeze(-1).to(dtype=X.dtype)
    n = w.sum(dim=1).clamp_min(1.0)
    xc = (X * w).sum(dim=1, keepdim=True) / n.unsqueeze(1)
    yc = (Y * w).sum(dim=1, keepdim=True) / n.unsqueeze(1)
    Xc = (X - xc) * w
    Yc = (Y - yc) * w
    x_std = ((Xc * Xc).sum(dim=1) / n).sqrt().mean(dim=1)
    y_std = ((Yc * Yc).sum(dim=1) / n).sqrt().mean(dim=1)
    ytx = torch.bmm(Yc.transpose(1, 2), Xc)
    xtx = torch.bmm(Xc.transpose(1, 2), Xc)
    yty = torch.bmm(Yc.transpose(1, 2), Yc)
    num = (ytx ** 2).sum(dim=(1, 2))
    denom = xtx.norm(p="fro", dim=(1, 2)) * yty.norm(p="fro", dim=(1, 2))
    val = num / (denom + 1e-12)
    collapsed = (x_std < var_guard) | (y_std < var_guard)
    return torch.where(collapsed, torch.full_like(val, float("nan")), val)


def bootstrap_mean_linear_cka(
    clean_reps,
    noisy_stack,
    draws,
    var_guard=1e-6,
    point_estimate=None,
    alpha=0.05,
    chunk_size=32,
):
    """
    Patient-bootstrap CI of the mean (across noise seeds) of linear CKA.

    Point estimate stays the caller-provided numpy CKA (or nanmean of
    full-sample per-seed CKA). Each resample computes CKA per seed, then
    averages — representations are never averaged across seeds.
    """
    torch = _try_torch()
    clean_reps = np.asarray(clean_reps, dtype=np.float32)
    noisy_stack = np.asarray(noisy_stack, dtype=np.float32)
    if noisy_stack.ndim == 2:
        noisy_stack = noisy_stack[None, ...]
    n_seed = noisy_stack.shape[0]
    if point_estimate is None:
        point_estimate = float(np.nanmean([
            linear_cka(clean_reps, noisy_stack[s], var_guard=var_guard)
            for s in range(n_seed)
        ]))

    if torch is None:
        boots = np.array([
            np.nanmean([
                linear_cka(clean_reps[idx], noisy_stack[s][idx], var_guard=var_guard)
                for s in range(n_seed)
            ])
            for idx in draws
        ], dtype=float)
        return percentile_ci_from_boots(boots, point_estimate, alpha)

    device = _geometry_device(torch)
    clean_t = torch.from_numpy(np.ascontiguousarray(clean_reps)).to(device)
    noisy_t = torch.from_numpy(np.ascontiguousarray(noisy_stack)).to(device)
    max_n = max(len(idx) for idx in draws)
    boots = []
    for start in range(0, len(draws), chunk_size):
        chunk = draws[start:start + chunk_size]
        b = len(chunk)
        idx_pad = np.zeros((b, max_n), dtype=np.int64)
        mask_np = np.zeros((b, max_n), dtype=bool)
        for i, idx in enumerate(chunk):
            n = len(idx)
            idx_pad[i, :n] = idx
            mask_np[i, :n] = True
        idx_t = torch.from_numpy(idx_pad).to(device)
        mask = torch.from_numpy(mask_np).to(device)
        x = clean_t[idx_t]
        seed_vals = []
        for s in range(n_seed):
            y = noisy_t[s][idx_t]
            seed_vals.append(_cka_from_padded(x, y, mask, var_guard))
        mean_cka = torch.stack(seed_vals, dim=0).nanmean(dim=0)
        boots.append(mean_cka.detach().cpu().numpy())
    boots = np.concatenate(boots, axis=0)
    return percentile_ci_from_boots(boots, point_estimate, alpha)
