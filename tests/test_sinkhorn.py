"""SwAV Sinkhorn columns must sum to 1 after Q *= B."""
import torch

from ecg_ssl_utils.ssl.swav import sinkhorn


def test_sinkhorn_columns_sum_to_one():
    torch.manual_seed(0)
    scores = torch.randn(16, 32)  # (B, K)
    Q = sinkhorn(scores, niters=3, epsilon=0.05)
    col = Q.sum(dim=1)
    assert torch.allclose(col, torch.ones_like(col), atol=1e-4)
