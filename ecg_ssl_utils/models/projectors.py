"""
MLP projector and predictor heads for SSL paradigms.
"""

import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """
    2-layer MLP projection head (used by SimCLR, CLOCS, SwAV).

    Architecture: Linear → BN → ReLU → Linear
    """

    def __init__(self, in_dim: int = 384, hidden_dim: int = 384, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPPredictor(nn.Module):
    """
    2-layer MLP predictor head (used by BYOL, JEPA).

    Architecture: Linear → BN → ReLU → Linear
    """

    def __init__(self, in_dim: int = 256, hidden_dim: int = 4096, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BYOLProjector(nn.Module):
    """
    BYOL-specific projector: Linear → BN → ReLU → Linear.
    Typically larger hidden dim (4096).
    """

    def __init__(self, in_dim: int = 384, hidden_dim: int = 4096, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwAVPrototypes(nn.Module):
    """
    SwAV prototype layer (learnable prototype vectors).
    """

    def __init__(self, in_dim: int = 128, n_prototypes: int = 256):
        super().__init__()
        self.prototypes = nn.Linear(in_dim, n_prototypes, bias=False)
        # Normalize prototype weights
        with torch.no_grad():
            nn.init.uniform_(self.prototypes.weight)
            self.prototypes.weight.copy_(
                nn.functional.normalize(self.prototypes.weight, dim=1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, in_dim) — L2-normalized features.

        Returns
        -------
        scores : torch.Tensor
            Shape (B, n_prototypes) — dot products with prototypes.
        """
        return self.prototypes(x)
