"""
ECG-specific data augmentations for self-supervised learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


class ECGAugmentation(nn.Module):
    """
    Stochastic ECG augmentation pipeline for contrastive learning.

    Available transforms:
    1. Random crop + resize
    2. Temporal jitter (circular shift)
    3. Amplitude scaling
    4. Gaussian noise
    5. Lead dropout
    6. Temporal masking
    """

    def __init__(
        self,
        signal_length: int = 5000,
        n_transforms: int = 4,
        crop_range: tuple = (0.8, 1.0),
        jitter_max: int = 500,
        amp_range: tuple = (0.8, 1.2),
        gauss_std: float = 0.05,
        lead_drop_prob: float = 0.3,
        lead_drop_max: int = 3,
        mask_ratio: float = 0.1,
    ):
        super().__init__()
        self.signal_length = signal_length
        self.n_transforms = n_transforms
        self.crop_range = crop_range
        self.jitter_max = jitter_max
        self.amp_range = amp_range
        self.gauss_std = gauss_std
        self.lead_drop_prob = lead_drop_prob
        self.lead_drop_max = lead_drop_max
        self.mask_ratio = mask_ratio

    def _random_crop(self, x: torch.Tensor) -> torch.Tensor:
        """Crop a random portion and resize back to original length."""
        B, C, L = x.shape
        crop_frac = torch.empty(B).uniform_(*self.crop_range)
        results = []
        for i in range(B):
            crop_len = int(crop_frac[i].item() * L)
            start = torch.randint(0, L - crop_len + 1, (1,)).item()
            cropped = x[i, :, start:start + crop_len]
            # Resize back via linear interpolation
            resized = torch.nn.functional.interpolate(
                cropped.unsqueeze(0), size=L, mode='linear', align_corners=False
            ).squeeze(0)
            results.append(resized)
        return torch.stack(results)

    def _temporal_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Random circular shift along time axis."""
        B, C, L = x.shape
        shifts = torch.randint(-self.jitter_max, self.jitter_max + 1, (B,))
        results = []
        for i in range(B):
            results.append(torch.roll(x[i], shifts[i].item(), dims=-1))
        return torch.stack(results)

    def _amplitude_scaling(self, x: torch.Tensor) -> torch.Tensor:
        """Random per-sample amplitude scaling."""
        B = x.shape[0]
        scales = torch.empty(B, 1, 1).uniform_(*self.amp_range).to(x.device)
        return x * scales

    def _gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add small Gaussian noise."""
        noise = torch.randn_like(x) * self.gauss_std
        return x + noise

    def _lead_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out random leads."""
        B, C, L = x.shape
        x = x.clone()
        for i in range(B):
            if torch.rand(1).item() < self.lead_drop_prob:
                n_drop = torch.randint(1, self.lead_drop_max + 1, (1,)).item()
                drop_leads = torch.randperm(C)[:n_drop]
                x[i, drop_leads, :] = 0.0
        return x

    def _temporal_masking(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out a random contiguous segment."""
        B, C, L = x.shape
        mask_len = int(self.mask_ratio * L)
        x = x.clone()
        for i in range(B):
            start = torch.randint(0, L - mask_len + 1, (1,)).item()
            x[i, :, start:start + mask_len] = 0.0
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a random subset of augmentations.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 12, 5000).

        Returns
        -------
        augmented : torch.Tensor
            Same shape.
        """
        all_transforms = [
            self._random_crop,
            self._temporal_jitter,
            self._amplitude_scaling,
            self._gaussian_noise,
            self._lead_dropout,
            self._temporal_masking,
        ]

        # Select random subset
        n = min(self.n_transforms, len(all_transforms))
        indices = torch.randperm(len(all_transforms))[:n]

        for idx in indices:
            x = all_transforms[idx](x)

        return x
