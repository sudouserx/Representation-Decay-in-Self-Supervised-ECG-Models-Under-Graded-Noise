"""
ECG signal preprocessing: bandpass filtering and normalization.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from typing import Tuple, Optional, Dict
import json


def bandpass_filter(
    signals: np.ndarray,
    low: float = 0.5,
    high: float = 45.0,
    fs: int = 500,
    order: int = 4,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter to ECG signals.

    Parameters
    ----------
    signals : np.ndarray
        Shape (N, 12, L) or (12, L).
    low, high : float
        Cutoff frequencies in Hz.
    fs : int
        Sampling rate.
    order : int
        Filter order.

    Returns
    -------
    filtered : np.ndarray
        Same shape as input.
    """
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype='band')

    if signals.ndim == 2:
        # Single record: (12, L)
        return filtfilt(b, a, signals, axis=-1).astype(np.float32)

    # Batch: (N, 12, L)
    filtered = np.empty_like(signals)
    for i in range(signals.shape[0]):
        filtered[i] = filtfilt(b, a, signals[i], axis=-1)
    return filtered.astype(np.float32)


def compute_norm_stats(
    signals: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute per-lead mean and std from training signals.

    Parameters
    ----------
    signals : np.ndarray
        Shape (N, 12, L).

    Returns
    -------
    dict with 'mean' (12,) and 'std' (12,).
    """
    # Mean and std per lead across all samples and time steps
    mean = signals.mean(axis=(0, 2))   # (12,)
    std = signals.std(axis=(0, 2))     # (12,)
    # Avoid division by zero
    std = np.where(std < 1e-8, 1.0, std)
    return {'mean': mean.astype(np.float32), 'std': std.astype(np.float32)}


def normalize_signals(
    signals: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Apply per-lead z-score normalization.

    Parameters
    ----------
    signals : np.ndarray
        Shape (N, 12, L) or (12, L).
    mean, std : np.ndarray
        Shape (12,). Computed from training set.

    Returns
    -------
    normalized : np.ndarray
    """
    if signals.ndim == 2:
        return ((signals - mean[:, None]) / std[:, None]).astype(np.float32)
    # Batch
    return ((signals - mean[None, :, None]) / std[None, :, None]).astype(np.float32)


def save_norm_stats(stats: Dict[str, np.ndarray], path: str):
    """Save normalization stats to JSON."""
    data = {
        'mean': stats['mean'].tolist(),
        'std': stats['std'].tolist(),
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_norm_stats(path: str) -> Dict[str, np.ndarray]:
    """Load normalization stats from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)
    return {
        'mean': np.array(data['mean'], dtype=np.float32),
        'std': np.array(data['std'], dtype=np.float32),
    }
