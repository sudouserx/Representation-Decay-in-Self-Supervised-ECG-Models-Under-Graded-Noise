"""
Synthetic noise generators for ECG corruption.
Covers powerline (50 Hz + harmonics), electrode pop, and inverter switching.
"""

import numpy as np
from typing import Optional


def generate_powerline_noise(
    duration_s: float = 10.0,
    fs: int = 500,
    n_leads: int = 12,
    n_harmonics: int = 3,
    base_freq: float = 50.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate 50 Hz powerline interference with harmonics.

    n(t) = Σ_{h=1}^{H} A_h · sin(2π · 50h · t + φ_h)
    A_1=1.0, A_2=0.5, A_3=0.25; φ_h ~ Uniform(0, 2π)

    Parameters
    ----------
    duration_s : float
        Signal duration in seconds.
    fs : int
        Sampling rate.
    n_leads : int
        Number of ECG leads.
    n_harmonics : int
        Number of harmonics to include.
    base_freq : float
        Fundamental frequency (50 Hz for European mains).
    seed : int
        Random seed for phase.

    Returns
    -------
    noise : np.ndarray
        Shape (n_leads, n_samples).
    """
    rng = np.random.RandomState(seed)
    n_samples = int(duration_s * fs)
    t = np.arange(n_samples) / fs

    amplitudes = [1.0 / (h ** 1.0) for h in range(1, n_harmonics + 1)]

    noise = np.zeros((n_leads, n_samples), dtype=np.float32)
    for lead in range(n_leads):
        for h_idx, (amp, h) in enumerate(zip(amplitudes, range(1, n_harmonics + 1))):
            phase = rng.uniform(0, 2 * np.pi)
            noise[lead] += amp * np.sin(2 * np.pi * base_freq * h * t + phase)

    return noise


def generate_electrode_pop_noise(
    duration_s: float = 10.0,
    fs: int = 500,
    n_leads: int = 12,
    pop_rate: float = 0.5,
    sigma_s: float = 0.005,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate electrode-pop transient artifacts.

    n_pop(t) = Σ_j B_j · exp(-(t - t_j)² / (2σ²))
    t_j ~ Poisson(rate), B_j ~ N(0, 1)

    Parameters
    ----------
    duration_s : float
        Signal duration in seconds.
    fs : int
        Sampling rate.
    n_leads : int
        Number of ECG leads.
    pop_rate : float
        Average pops per second (Poisson rate).
    sigma_s : float
        Width of each pop in seconds.
    seed : int
        Random seed.

    Returns
    -------
    noise : np.ndarray
        Shape (n_leads, n_samples).
    """
    rng = np.random.RandomState(seed)
    n_samples = int(duration_s * fs)
    t = np.arange(n_samples) / fs
    sigma = sigma_s

    noise = np.zeros((n_leads, n_samples), dtype=np.float32)

    for lead in range(n_leads):
        # Number of pops for this lead (Poisson)
        n_pops = rng.poisson(pop_rate * duration_s)
        if n_pops == 0:
            continue

        # Random pop locations and amplitudes
        pop_times = rng.uniform(0, duration_s, size=n_pops)
        pop_amps = rng.randn(n_pops)

        for t_j, b_j in zip(pop_times, pop_amps):
            gaussian = np.exp(-((t - t_j) ** 2) / (2 * sigma ** 2))
            noise[lead] += b_j * gaussian

    return noise


def generate_inverter_switching_noise(
    duration_s: float = 10.0,
    fs: int = 500,
    n_leads: int = 12,
    burst_freq: float = 200.0,
    duty_cycle: float = 0.1,
    switching_period_s: float = 0.005,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate inverter/switching-mode power supply interference.

    Models periodic high-frequency bursts at the switching frequency,
    aliased into the recording bandwidth.

    n_inv(t) = w(t) · sin(2π · f_burst · t)
    w(t) is a periodic rectangular window with given duty cycle.

    Note: This is a synthetic approximation. The spectral profile is based on
    documented switching-mode PSU EMI characteristics, modeled as periodic
    bursts of a carrier frequency (aliased from kHz-range into 0-250 Hz band
    at 500 Hz sampling). Flagged as modeled/synthetic pending field validation.

    Parameters
    ----------
    duration_s : float
        Signal duration in seconds.
    fs : int
        Sampling rate.
    n_leads : int
        Number of ECG leads.
    burst_freq : float
        Carrier frequency of the burst (already aliased to <Nyquist).
    duty_cycle : float
        Fraction of each switching period that the burst is active.
    switching_period_s : float
        Period of the switching (envelope) in seconds.
    seed : int
        Random seed.

    Returns
    -------
    noise : np.ndarray
        Shape (n_leads, n_samples).
    """
    rng = np.random.RandomState(seed)
    n_samples = int(duration_s * fs)
    t = np.arange(n_samples) / fs

    # Rectangular window (periodic)
    window = np.zeros(n_samples, dtype=np.float32)
    for start in np.arange(0, duration_s, switching_period_s):
        active_end = start + duty_cycle * switching_period_s
        mask = (t >= start) & (t < active_end)
        window[mask] = 1.0

    # Carrier signal
    carrier = np.sin(2 * np.pi * burst_freq * t).astype(np.float32)

    # Modulated noise (same for all leads with slight phase variation)
    noise = np.zeros((n_leads, n_samples), dtype=np.float32)
    for lead in range(n_leads):
        phase_offset = rng.uniform(0, 2 * np.pi)
        lead_carrier = np.sin(2 * np.pi * burst_freq * t + phase_offset).astype(np.float32)
        noise[lead] = window * lead_carrier

    return noise


def generate_noise_bank(
    n_templates: int = 10,
    duration_s: float = 10.0,
    fs: int = 500,
    n_leads: int = 12,
    seed: int = 42,
) -> dict:
    """
    Generate a bank of synthetic noise templates for all synthetic types.

    Returns
    -------
    dict: {noise_type: np.ndarray of shape (n_templates, n_leads, n_samples)}
    """
    bank = {}
    rng = np.random.RandomState(seed)

    # Powerline
    powerline_templates = []
    for i in range(n_templates):
        s = rng.randint(0, 100000)
        powerline_templates.append(
            generate_powerline_noise(duration_s, fs, n_leads, seed=s)
        )
    bank['powerline'] = np.stack(powerline_templates, axis=0)

    # Electrode pop
    pop_templates = []
    for i in range(n_templates):
        s = rng.randint(0, 100000)
        pop_templates.append(
            generate_electrode_pop_noise(duration_s, fs, n_leads, seed=s)
        )
    bank['electrode_pop'] = np.stack(pop_templates, axis=0)

    # Inverter switching
    inverter_templates = []
    for i in range(n_templates):
        s = rng.randint(0, 100000)
        inverter_templates.append(
            generate_inverter_switching_noise(duration_s, fs, n_leads, seed=s)
        )
    bank['inverter'] = np.stack(inverter_templates, axis=0)

    return bank
