"""SNR round-trip on synthetic noise."""
import numpy as np

from ecg_ssl_utils.noise.injection import compute_snr, inject_noise
from ecg_ssl_utils.noise.synthetic_noise import generate_powerline_noise
from ecg_ssl_utils.data.preprocessing import bandpass_filter


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


def test_post_filter_snr_captures_frontend_attenuation():
    rng = np.random.RandomState(4)
    clean = rng.randn(1, 12, 5000).astype(np.float32)
    template = generate_powerline_noise(seed=3)[None, ...]
    noisy = inject_noise(
        clean[0], {"powerline": template}, "powerline", 0.0,
        seed=42, record_id=1,
    )[None, ...]

    clean_f = bandpass_filter(clean, low=0.05, high=45.0, fs=500, order=4)
    noisy_f = bandpass_filter(noisy, low=0.05, high=45.0, fs=500, order=4)

    assert compute_snr(clean_f, noisy_f) > compute_snr(clean, noisy)
