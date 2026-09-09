"""LinearProbe applies temperature exactly once."""
import numpy as np
import torch

from ecg_ssl_utils.probe.linear_probe import LinearProbe


def test_forward_then_sigmoid_matches_predict_proba():
    torch.manual_seed(0)
    probe = LinearProbe(8, 3)
    probe.temperature.data = torch.tensor([2.5])
    h = torch.randn(5, 8)
    logits = probe(h)
    probs = probe.predict_proba(h)
    expected = torch.sigmoid(logits)
    assert torch.allclose(probs, expected, atol=1e-6)
    raw = probe.fc(h)
    twice = torch.sigmoid(logits / 2.5)
    once = torch.sigmoid(raw / 2.5)
    assert not torch.allclose(twice, once, atol=1e-4)
    assert torch.allclose(probs, once, atol=1e-6)


def test_predict_probs_numpy_path():
    torch.manual_seed(1)
    probe = LinearProbe(4, 2)
    T = 1.7
    probe.temperature.data.fill_(1.0)
    h = torch.randn(6, 4)
    raw = probe.fc(h).detach().numpy()
    probs = 1.0 / (1.0 + np.exp(-raw / T))
    probe.temperature.data.fill_(T)
    ref = probe.predict_proba(h).detach().numpy()
    assert np.allclose(probs, ref, atol=1e-6)
