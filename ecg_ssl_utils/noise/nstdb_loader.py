"""
MIT-BIH Noise Stress Test Database (NSTDB) loader.
Loads the three canonical noise records: baseline wander (bw),
muscle/EMG artifact (ma), and electrode motion artifact (em).
"""

import os
import numpy as np
import wfdb
from typing import Dict, Optional


NSTDB_RECORDS = {
    'bw': 'bw',       # Baseline wander
    'ma': 'ma',       # Muscle (EMG) artifact
    'em': 'em',       # Electrode motion artifact
}


def load_nstdb_noise(
    data_dir: str,
    n_leads_target: int = 12,
    target_fs: int = 500,
    n_templates: int = 10,
    template_duration_s: float = 10.0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Load MIT-BIH NSTDB noise records and extract templates.

    The NSTDB contains 2-lead noise recordings at 360 Hz.
    We resample to target_fs and replicate across n_leads_target leads
    with slight random phase shifts for diversity.

    Parameters
    ----------
    data_dir : str
        Directory containing NSTDB wfdb files (bw.dat, ma.dat, em.dat).
    n_leads_target : int
        Number of ECG leads to generate (replicate with variation).
    target_fs : int
        Target sampling rate.
    n_templates : int
        Number of templates to extract per noise type.
    template_duration_s : float
        Duration of each template in seconds.
    seed : int
        Random seed for template extraction.

    Returns
    -------
    noise_bank : dict
        {noise_type: np.ndarray of shape (n_templates, n_leads, n_samples)}
    """
    from scipy.signal import resample

    rng = np.random.RandomState(seed)
    template_len = int(template_duration_s * target_fs)
    noise_bank = {}

    for noise_type, record_name in NSTDB_RECORDS.items():
        record_path = os.path.join(data_dir, record_name)

        try:
            record = wfdb.rdsamp(record_path)
            signal = record[0]   # (n_samples_orig, n_channels_orig)
            orig_fs = record[1]['fs']
        except Exception as e:
            print(f"Warning: Could not load NSTDB record '{record_name}': {e}")
            print(f"  Generating Gaussian placeholder for '{noise_type}'.")
            # Fallback: generate Gaussian noise as placeholder
            templates = rng.randn(n_templates, n_leads_target, template_len).astype(np.float32)
            noise_bank[noise_type] = templates
            continue

        # Resample to target_fs if needed
        if orig_fs != target_fs:
            n_orig = signal.shape[0]
            n_target = int(n_orig * target_fs / orig_fs)
            signal = resample(signal, n_target, axis=0)

        signal = signal.astype(np.float32)
        total_samples = signal.shape[0]
        n_channels_orig = signal.shape[1]

        templates = []
        for t in range(n_templates):
            # Random start position
            max_start = max(0, total_samples - template_len)
            start = rng.randint(0, max_start + 1) if max_start > 0 else 0
            end = start + template_len

            segment = signal[start:end, :]  # (template_len, n_channels_orig)

            # Pad if segment is too short
            if segment.shape[0] < template_len:
                pad_len = template_len - segment.shape[0]
                segment = np.pad(segment, ((0, pad_len), (0, 0)), mode='wrap')

            # Replicate to n_leads with variation
            multi_lead = np.zeros((n_leads_target, template_len), dtype=np.float32)
            for lead in range(n_leads_target):
                # Select source channel and add slight amplitude and phase variation
                src_ch = lead % n_channels_orig
                amp_scale = rng.uniform(0.8, 1.2)
                shift = rng.randint(0, max(1, template_len // 10))
                multi_lead[lead] = amp_scale * np.roll(segment[:, src_ch], shift)

            templates.append(multi_lead)

        noise_bank[noise_type] = np.stack(templates, axis=0)

    return noise_bank
