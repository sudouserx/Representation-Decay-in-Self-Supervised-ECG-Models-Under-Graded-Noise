"""SNR round-trip on synthetic noise."""
import numpy as np

from ecg_ssl_utils.noise.injection import compute_snr, inject_noise
from ecg_ssl_utils.noise.synthetic_noise import generate_powerline_noise


def test_snr_roundtrip():
    rng = np.random.RandomState(0)
    clean = rng.randn(12, 5000).astype(np.float32)
    templates = np.stack(
        [generate_powerline_noise(seed=i) for i in range(5)], axis=0
    )
    bank = {"powerline": templates}
    for target in [-6.0, 0.0, 12.0, 24.0]:
        noisy = inject_noise(clean, bank, "powerline", target, seed=42, record_id=0)
        actual = compute_snr(clean, noisy)
        assert abs(actual - target) < 0.1, (target, actual)
