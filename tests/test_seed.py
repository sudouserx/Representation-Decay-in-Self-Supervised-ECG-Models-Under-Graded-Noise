"""Seeded SimCLR first-batch loss is deterministic on CPU without AMP."""
import torch
from torch.utils.data import TensorDataset

from ecg_ssl_utils.models.projectors import MLPProjector
from ecg_ssl_utils.models.vit_small_1d import ViTSmall1D
from ecg_ssl_utils.repro import make_deterministic_loader, set_global_seed
from ecg_ssl_utils.ssl.augmentations import ECGAugmentation
from ecg_ssl_utils.ssl.simclr import SimCLRTrainer


def _first_loss(seed):
    set_global_seed(seed)
    x = torch.randn(20, 12, 500)
    loader = make_deterministic_loader(
        TensorDataset(x), batch_size=4, seed=seed, workers=0, drop_last=True,
    )
    encoder = ViTSmall1D(signal_length=500, patch_size=50, embed_dim=64, depth=2, num_heads=4)
    projector = MLPProjector(64, 64, 32)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(projector.parameters()), lr=1e-3)
    trainer = SimCLRTrainer(
        encoder, projector, ECGAugmentation(signal_length=500, jitter_max=50), opt,
        use_amp=False, device="cpu", grad_accum_steps=1,
    )
    batch = next(iter(loader))[0]
    return trainer.train_step(batch, 0)


def test_seeded_simclr_first_batch_loss_matches():
    a = _first_loss(123)
    b = _first_loss(123)
    assert abs(a - b) < 1e-5
