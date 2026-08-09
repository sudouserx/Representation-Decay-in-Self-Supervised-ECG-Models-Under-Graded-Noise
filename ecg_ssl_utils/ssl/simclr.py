"""
SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
Adapted for 1D ECG signals.

Loss: NT-Xent (Normalized Temperature-scaled Cross-Entropy)
  ℓ(i,j) = -log[ exp(sim(z_i, z_j) / τ) / Σ_{k≠i} exp(sim(z_i, z_k) / τ) ]
  sim(u,v) = u·v / (‖u‖·‖v‖)

Reference: Chen et al., ICML 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss.

    Parameters
    ----------
    z1, z2 : torch.Tensor
        Projected representations from two views, shape (B, D).
    temperature : float
        Temperature scaling parameter.

    Returns
    -------
    loss : torch.Tensor
        Scalar loss.
    """
    B = z1.shape[0]
    device = z1.device

    # L2 normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    # Cosine similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * B, device=device).bool()
    sim.masked_fill_(mask, -1e9)

    # Labels: positive pair for i is i+B (and vice versa)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=device),
        torch.arange(0, B, device=device),
    ])

    loss = F.cross_entropy(sim, labels)
    return loss


class SimCLRTrainer:
    """
    SimCLR training wrapper.

    Usage:
        trainer = SimCLRTrainer(encoder, projector, config)
        for epoch in range(epochs):
            for batch in loader:
                loss = trainer.train_step(batch)
    """

    def __init__(
        self,
        encoder: nn.Module,
        projector: nn.Module,
        augmentation: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object] = None,
        temperature: float = 0.1,
        use_amp: bool = True,
        device: str = 'cuda',
    ):
        self.encoder = encoder.to(device)
        self.projector = projector.to(device)
        self.augmentation = augmentation
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.temperature = temperature
        self.use_amp = use_amp
        self.device = device
        self.scaler = torch.amp.GradScaler('cuda') if use_amp else None

    def train_step(self, batch: torch.Tensor) -> float:
        """
        One training step.

        Parameters
        ----------
        batch : torch.Tensor
            Clean ECG batch, shape (B, 12, 5000).

        Returns
        -------
        loss_value : float
        """
        self.encoder.train()
        self.projector.train()

        batch = batch.to(self.device)

        # Generate two augmented views
        view1 = self.augmentation(batch)
        view2 = self.augmentation(batch)

        self.optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # Encode
            h1 = self.encoder(view1)  # (B, 384)
            h2 = self.encoder(view2)

            # Project
            z1 = self.projector(h1)   # (B, 128)
            z2 = self.projector(h2)

            # Loss
            loss = nt_xent_loss(z1, z2, self.temperature)

        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return loss.item()
