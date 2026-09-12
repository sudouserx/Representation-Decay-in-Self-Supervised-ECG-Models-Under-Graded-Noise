"""Golden-value checks for CKA, ER, ECE, AUROC, and patient bootstrap."""
import numpy as np
from sklearn.metrics import roc_auc_score

from ecg_ssl_utils.config import get_config
from ecg_ssl_utils.eval.auroc import binary_auroc, macro_auroc
from ecg_ssl_utils.eval.bootstrap import (
    _patient_index_map,
    _resample_indices,
    clear_bootstrap_draw_cache,
    make_patient_bootstrap_draws,
    patient_bootstrap_ci,
)
from ecg_ssl_utils.eval.cka import bootstrap_mean_linear_cka, linear_cka
from ecg_ssl_utils.eval.ece import _ece_from_bins, expected_calibration_error
from ecg_ssl_utils.eval.effective_rank import bootstrap_mean_effective_rank, effective_rank


def test_linear_cka_identical_is_one():
    rng = np.random.RandomState(0)
    X = rng.randn(64, 16)
    assert abs(linear_cka(X, X) - 1.0) < 1e-6


def test_linear_cka_collapse_guard():
    X = np.ones((32, 8))
    Y = np.random.RandomState(1).randn(32, 8)
    val, flag = linear_cka(X, Y, var_guard=1e-6, return_collapse_flag=True)
    assert flag and np.isnan(val)


def test_effective_rank_full_rank():
    rng = np.random.RandomState(2)
    X = rng.randn(80, 10)
    er = effective_rank(X)
    assert 5 < er <= 10


def test_ece_binary_confidence_golden():
    """Perfect 0.1/0.9 predictions have confidence 0.9 and ECE 0.1."""
    y = np.zeros((200, 1))
    y[:100, 0] = 1
    p = np.where(y == 1, 0.9, 0.1).astype(float)
    ece = expected_calibration_error(y, p, n_bins=10)
    assert abs(ece - 0.1) < 1e-6


def test_ece_bincount_matches_loop():
    rng = np.random.RandomState(4)
    y = (rng.rand(300, 3) > 0.6).astype(float)
    p = rng.rand(300, 3)
    fast = expected_calibration_error(y, p, n_bins=15)
    N = y.shape[0]
    ref = []
    for c in range(3):
        pos = p[:, c]
        pred = (pos >= 0.5).astype(float)
        correct = (y[:, c] == pred).astype(float)
        conf = np.maximum(pos, 1.0 - pos)
        edges = np.linspace(0, 1, 16)
        ref.append(_ece_from_bins(conf, correct, edges, N))
    assert abs(fast - float(np.mean(ref))) < 1e-12


def test_binary_auroc_matches_sklearn_untied_and_tied():
    rng = np.random.RandomState(5)
    y = rng.randint(0, 2, size=80)
    y[0], y[1] = 0, 1
    s = rng.randn(80)
    assert abs(binary_auroc(y, s) - roc_auc_score(y, s)) < 1e-12
    s_tied = np.round(s, 1)
    assert abs(binary_auroc(y, s_tied) - roc_auc_score(y, s_tied)) < 1e-12


def test_macro_auroc_skips_sparse_classes():
    y = np.zeros((20, 2))
    y[:3, 0] = 1
    y[:10, 1] = 1
    p = np.linspace(0, 1, 20).reshape(-1, 1).repeat(2, axis=1)
    assert np.isfinite(macro_auroc(y, p, min_positives=5))
    only = macro_auroc(y, p, min_positives=10)
    assert abs(only - binary_auroc(y[:, 1], p[:, 1])) < 1e-12


def test_bootstrap_returns_full_sample_point():
    rng = np.random.RandomState(3)
    vals = rng.randn(50)
    pids = np.arange(50)
    point, lo, hi = patient_bootstrap_ci(
        lambda idx: float(vals[idx].mean()),
        pids, n_bootstrap=50, seed=3,
    )
    assert abs(point - vals.mean()) < 1e-12
    assert lo <= point <= hi


def test_bootstrap_draws_match_live_rng():
    clear_bootstrap_draw_cache()
    pids = np.array([0, 0, 1, 1, 1, 2, 3, 3], dtype=np.int64)
    live = []
    rng = np.random.RandomState(42)
    unique, p2idx = _patient_index_map(pids)
    for _ in range(40):
        live.append(_resample_indices(rng, unique, p2idx))
    cached = make_patient_bootstrap_draws(pids, 40, 42, use_cache=True)
    again = make_patient_bootstrap_draws(pids, 40, 42, use_cache=True)
    for a, b in zip(live, cached):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(cached, again):
        np.testing.assert_array_equal(a, b)
    ci_live = patient_bootstrap_ci(
        lambda idx: float(idx.sum()), pids, n_bootstrap=40, seed=42,
        draws=make_patient_bootstrap_draws(pids, 40, 42, use_cache=False),
    )
    ci_cached = patient_bootstrap_ci(
        lambda idx: float(idx.sum()), pids, n_bootstrap=40, seed=42,
    )
    assert ci_live == ci_cached
    clear_bootstrap_draw_cache()


def test_batched_cka_matches_numpy():
    rng = np.random.RandomState(6)
    X = rng.randn(48, 12).astype(np.float32)
    Y = rng.randn(48, 12).astype(np.float32)
    Z = (Y + 0.05 * rng.randn(48, 12)).astype(np.float32)
    stack = np.stack([Y, Z], axis=0)
    draws = [
        np.arange(48),
        rng.choice(48, size=48, replace=True),
        np.concatenate([np.arange(20), np.arange(20)]),
    ]
    expected = []
    for idx in draws:
        expected.append(np.nanmean([
            linear_cka(X[idx], stack[s][idx]) for s in range(2)
        ]))
    point, lo, hi = bootstrap_mean_linear_cka(
        X, stack, draws, point_estimate=float(np.mean(expected)),
    )
    assert abs(point - float(np.mean(expected))) < 1e-6
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= hi


def test_batched_er_finite_and_close_to_point():
    rng = np.random.RandomState(7)
    X = rng.randn(60, 16).astype(np.float32)
    draws = [np.arange(60), rng.choice(60, size=60, replace=True)]
    point = effective_rank(X, n_components=8, seed=0)
    _, lo, hi = bootstrap_mean_effective_rank(
        X, draws, n_components=8, seed=0, subsample=40, point_estimate=point,
    )
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= point + 1.0
    assert hi >= point - 1.0


def test_cka_cpu_vs_gpu_rtol():
    rng = np.random.RandomState(8)
    X = rng.randn(32, 10).astype(np.float32)
    Y = rng.randn(32, 10).astype(np.float32)
    draws = [np.arange(32)]
    try:
        import torch  # noqa: F401
    except ImportError:
        point, _, _ = bootstrap_mean_linear_cka(X, Y, draws)
        assert abs(point - linear_cka(X, Y)) < 1e-6
        return
    point, _, _ = bootstrap_mean_linear_cka(X, Y, draws)
    assert abs(point - linear_cka(X, Y)) < 1e-4


def test_config_noise_grid_includes_synthetic_and_nstdb():
    cfg = get_config()
    types = set(cfg.noise.noise_types_single)
    types.update(m["name"] for m in cfg.noise.mixed_noise_configs)
    for name in ("bw", "ma", "em", "powerline", "electrode_pop", "inverter",
                 "bw_ma", "em_powerline", "bw_ma_power"):
        assert name in types
    assert list(cfg.noise.snr_grid) == [24, 18, 12, 6, 0, -6]
    assert cfg.eval.primary_noise_types == ["bw", "ma", "em"]
    assert cfg.eval.primary_snr_db == 0.0
    assert cfg.eval.bootstrap_n == 1000
    assert cfg.eval.er_bootstrap_n == 200
