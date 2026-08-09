"""
SNR-scaled noise injection into ECG signals.
Deterministic given (sample_id, noise_config, seed).
"""

import numpy as np
from typing import Dict, Optional, List


def compute_signal_power(signal: np.ndarray) -> float:
    """Compute mean signal power (mean of squared values)."""
    return np.mean(signal ** 2)


def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """
    Compute actual SNR in dB between clean and noisy signals.

    Parameters
    ----------
    clean : np.ndarray
    noisy : np.ndarray

    Returns
    -------
    snr_db : float
    """
    noise = noisy - clean
    p_signal = compute_signal_power(clean)
    p_noise = compute_signal_power(noise)
    if p_noise < 1e-12:
        return float('inf')
    return 10.0 * np.log10(p_signal / p_noise)


def scale_noise_to_snr(
    clean_signal: np.ndarray,
    raw_noise: np.ndarray,
    target_snr_db: float,
) -> np.ndarray:
    """
    Scale raw noise so that adding it to clean_signal achieves target_snr_db.

    Formula:
        P_signal = mean(clean^2)
        P_target = P_signal / 10^(SNR_dB / 10)
        k = sqrt(P_target / P_raw)
        scaled_noise = k * raw_noise

    Parameters
    ----------
    clean_signal : np.ndarray
        Clean ECG signal, any shape.
    raw_noise : np.ndarray
        Raw noise signal, same shape as clean_signal.
    target_snr_db : float
        Desired SNR in dB.

    Returns
    -------
    scaled_noise : np.ndarray
    """
    p_signal = compute_signal_power(clean_signal)
    p_target_noise = p_signal / (10.0 ** (target_snr_db / 10.0))

    p_raw = compute_signal_power(raw_noise)
    if p_raw < 1e-12:
        # Noise is effectively zero — return zeros
        return np.zeros_like(raw_noise)

    k = np.sqrt(p_target_noise / p_raw)
    return (k * raw_noise).astype(np.float32)


def inject_noise(
    clean_signal: np.ndarray,
    noise_templates: Dict[str, np.ndarray],
    noise_type: str,
    snr_db: float,
    seed: int,
    record_id: int = 0,
    mixture_components: Optional[List[str]] = None,
    mixture_weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Inject noise into a clean ECG signal at a specified SNR.

    The injection is deterministic given (record_id, noise_type, snr_db, seed).

    Parameters
    ----------
    clean_signal : np.ndarray
        Shape (12, 5000).
    noise_templates : dict
        {noise_type: np.ndarray of shape (n_templates, 12, 5000)}.
    noise_type : str
        Type of noise or 'mixed'.
    snr_db : float
        Target SNR in dB.
    seed : int
        Random seed for determinism.
    record_id : int
        Record identifier (used to vary template selection per record).
    mixture_components : list of str, optional
        Noise types for mixed noise.
    mixture_weights : list of float, optional
        Weights for mixed noise (must sum to 1).

    Returns
    -------
    noisy_signal : np.ndarray
        Shape (12, 5000).
    """
    rng = np.random.RandomState(seed + record_id)

    if mixture_components and mixture_weights:
        # Mixed noise: combine multiple sources
        combined_noise = np.zeros_like(clean_signal)
        for comp, weight in zip(mixture_components, mixture_weights):
            templates = noise_templates[comp]
            tidx = rng.randint(0, len(templates))
            template = templates[tidx]
            # Random start offset for temporal diversity
            offset = rng.randint(0, max(1, template.shape[-1] - clean_signal.shape[-1]))
            noise_segment = template[:, offset:offset + clean_signal.shape[-1]]
            if noise_segment.shape[-1] < clean_signal.shape[-1]:
                noise_segment = np.pad(
                    noise_segment,
                    ((0, 0), (0, clean_signal.shape[-1] - noise_segment.shape[-1])),
                    mode='wrap',
                )
            combined_noise += weight * noise_segment

        scaled = scale_noise_to_snr(clean_signal, combined_noise, snr_db)
    else:
        # Single-source noise
        templates = noise_templates[noise_type]
        tidx = rng.randint(0, len(templates))
        template = templates[tidx]

        # Select segment with temporal diversity
        if template.shape[-1] > clean_signal.shape[-1]:
            offset = rng.randint(0, template.shape[-1] - clean_signal.shape[-1])
            noise_segment = template[:, offset:offset + clean_signal.shape[-1]]
        elif template.shape[-1] < clean_signal.shape[-1]:
            # Tile to fill
            reps = int(np.ceil(clean_signal.shape[-1] / template.shape[-1]))
            noise_segment = np.tile(template, (1, reps))[:, :clean_signal.shape[-1]]
        else:
            noise_segment = template.copy()

        scaled = scale_noise_to_snr(clean_signal, noise_segment, snr_db)

    noisy_signal = clean_signal + scaled
    return noisy_signal.astype(np.float32)
