"""
Noise specification and corrupted sample data structures.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class NoiseSpec:
    """Defines a single corruption condition."""
    noise_type: str                     # 'bw', 'ma', 'em', 'powerline', 'electrode_pop', 'inverter'
    source_dataset: str                 # 'nstdb' or 'synthetic'
    snr_db: float                       # target SNR in dB
    mixture_components: Optional[List[str]] = None
    mixture_weights: Optional[List[float]] = None
    seed: int = 42
    provenance: str = ''                # human-readable description

    def __post_init__(self):
        if not self.provenance:
            if self.mixture_components:
                parts = '+'.join(self.mixture_components)
                self.provenance = f"mixed({parts})@{self.snr_db}dB_seed{self.seed}"
            else:
                self.provenance = f"{self.noise_type}@{self.snr_db}dB_seed{self.seed}"


@dataclass
class CorruptedECGSample:
    """Represents a noised ECG sample with full provenance."""
    record_id: int
    patient_id: int
    clean_signal: np.ndarray            # (12, 5000)
    noisy_signal: np.ndarray            # (12, 5000)
    noise_spec: NoiseSpec
    snr_db: float
    seed: int
