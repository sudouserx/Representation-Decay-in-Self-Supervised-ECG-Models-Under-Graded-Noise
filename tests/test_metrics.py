"""Golden-value checks for CKA, ER, ECE, and patient bootstrap."""
import numpy as np

from ecg_ssl_utils.eval.bootstrap import patient_bootstrap_ci
from ecg_ssl_utils.eval.cka import linear_cka
from ecg_ssl_utils.eval.ece import expected_calibration_error
from ecg_ssl_utils.eval.effective_rank import effective_rank


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
