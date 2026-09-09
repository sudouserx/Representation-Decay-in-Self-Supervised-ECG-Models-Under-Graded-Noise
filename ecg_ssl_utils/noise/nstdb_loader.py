"""
MIT-BIH Noise Stress Test Database (NSTDB) loader.
Loads the three canonical noise records: baseline wander (bw),
muscle/EMG artifact (ma), and electrode motion artifact (em).
"""

import os
from typing import Dict

import numpy as np
from scipy.signal import resample_poly


NSTDB_RECORDS = {
    "bw": "bw",
    "ma": "ma",
    "em": "em",
}


class NSTDBMissingError(FileNotFoundError):
    """Raised when an NSTDB record cannot be loaded."""


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

    The NSTDB contains 2-lead noise recordings at 360 Hz. We resample with
    ``resample_poly`` to *target_fs* and replicate across *n_leads_target*
    leads with slight random amplitude and circular-roll variation.

    Missing NSTDB files raise ``NSTDBMissingError`` — there is no Gaussian
    fallback (that would silently turn "empirical" noise into white noise).

    Lead replication from 2 source channels is a documented approximation,
    not a physically measured 12-lead artifact field.
    """
    rng = np.random.RandomState(seed)
    template_len = int(template_duration_s * target_fs)
    noise_bank = {}

    for noise_type, record_name in NSTDB_RECORDS.items():
        record_path = os.path.join(data_dir, record_name)
        try:
            import wfdb
            record = wfdb.rdsamp(record_path)
            signal = record[0]
            orig_fs = record[1]["fs"]
        except Exception as e:
            raise NSTDBMissingError(
                f"Could not load NSTDB record '{record_name}' from {record_path}: {e}. "
                "Place bw/ma/em wfdb files in NSTDB_DIR; Gaussian fallback is disabled."
            ) from e

        if orig_fs != target_fs:
            from math import gcd
            g = gcd(int(target_fs), int(orig_fs))
            up, down = int(target_fs) // g, int(orig_fs) // g
            signal = resample_poly(signal, up, down, axis=0)

        signal = signal.astype(np.float32)
        total_samples = signal.shape[0]
        n_channels_orig = signal.shape[1]

        templates = []
        for _ in range(n_templates):
            max_start = max(0, total_samples - template_len)
            start = rng.randint(0, max_start + 1) if max_start > 0 else 0
            end = start + template_len
            segment = signal[start:end, :]

            if segment.shape[0] < template_len:
                pad_len = template_len - segment.shape[0]
                segment = np.pad(segment, ((0, pad_len), (0, 0)), mode="wrap")

            multi_lead = np.zeros((n_leads_target, template_len), dtype=np.float32)
            for lead in range(n_leads_target):
                src_ch = lead % n_channels_orig
                amp_scale = rng.uniform(0.8, 1.2)
                shift = rng.randint(0, max(1, template_len // 10))
                multi_lead[lead] = amp_scale * np.roll(segment[:, src_ch], shift)

            templates.append(multi_lead)

        noise_bank[noise_type] = np.stack(templates, axis=0)

    return noise_bank
