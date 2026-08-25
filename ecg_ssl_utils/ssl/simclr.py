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
from typing import Optional, Dict


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.5,
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

    # Force float32 for numerical stability (critical under AMP)
    z1 = z1.float()
    z2 = z2.float()

    # L2 normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    # Cosine similarity matrix in float32
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

    # Clamp to prevent overflow in exp()
    sim = torch.clamp(sim, max=30.0)

    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * B, device=device).bool()
    sim.masked_fill_(mask, float('-inf'))

    # Labels: positive pair for i is i+B (and vice versa)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=device),
        torch.arange(0, B, device=device),
    ])

    loss = F.cross_entropy(sim, labels)
    return loss


class SimCLRTrainer:
    """
    SimCLR training wrapper with collapse detection and gradient accumulation.

    Usage:
        trainer = SimCLRTrainer(encoder, projector, augmentation, optimizer,
                                temperature=0.5, grad_accum_steps=4)
        for epoch in range(epochs):
            for step, batch in enumerate(loader):
                loss = trainer.train_step(batch, step)
    """

    def __init__(
        self,
        encoder: nn.Module,
        projector: nn.Module,
        augmentation: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object] = None,
        temperature: float = 0.5,
        use_amp: bool = True,
        device: str = 'cuda',
        grad_clip_norm: float = 1.0,
        grad_accum_steps: int = 1,
    ):
        self.encoder = encoder.to(device)
        self.projector = projector.to(device)
        self.augmentation = augmentation
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.temperature = temperature
        self.use_amp = use_amp
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.grad_accum_steps = grad_accum_steps
        self.scaler = torch.amp.GradScaler('cuda') if use_amp else None

        # Cache for collapse metrics (populated by compute_collapse_metrics)
        self._last_z1 = None
        self._last_z2 = None

    def train_step(self, batch: torch.Tensor, step_idx: int = 0) -> float:
        """
        One training step with gradient accumulation and clipping.

        Parameters
        ----------
        batch : torch.Tensor
            Clean ECG batch, shape (B, 12, 5000).
        step_idx : int
            Current step index within the epoch (for accumulation sync).

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

        # Only zero gradients at accumulation boundaries
        if step_idx % self.grad_accum_steps == 0:
            self.optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # Encode
            h1 = self.encoder(view1)  # (B, 384)
            h2 = self.encoder(view2)

            # Project
            z1 = self.projector(h1)   # (B, 128)
            z2 = self.projector(h2)

            # Loss (scaled for accumulation)
            loss = nt_xent_loss(z1, z2, self.temperature)
            loss_scaled = loss / self.grad_accum_steps

        # Cache projections for collapse metrics (detached)
        self._last_z1 = z1.detach()
        self._last_z2 = z2.detach()

        if self.scaler:
            self.scaler.scale(loss_scaled).backward()
            # Step only at accumulation boundaries
            if (step_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.projector.parameters()),
                    self.grad_clip_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            loss_scaled.backward()
            if (step_idx + 1) % self.grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.projector.parameters()),
                    self.grad_clip_norm,
                )
                self.optimizer.step()

        return loss.item()  # Return unscaled loss for logging

    @torch.no_grad()
    def compute_collapse_metrics(self) -> Dict[str, float]:
        """
        Compute representation collapse diagnostics from the last batch.

        Returns
        -------
        metrics : dict
            - embed_std: mean std-dev across embedding dimensions (collapse → 0)
            - avg_cosine_sim: mean pairwise cosine similarity of negatives (collapse → 1.0)
        """
        if self._last_z1 is None or self._last_z2 is None:
            return {'embed_std': float('nan'), 'avg_cosine_sim': float('nan')}

        z = torch.cat([self._last_z1, self._last_z2], dim=0).float()  # (2B, D)

        # 1. Embedding std: mean of per-dimension std across the batch
        embed_std = z.std(dim=0).mean().item()

        # 2. Average cosine similarity of all pairs (excluding self)
        z_norm = F.normalize(z, dim=1)
        sim_matrix = torch.mm(z_norm, z_norm.t())  # (2B, 2B)
        B = self._last_z1.shape[0]
        n = 2 * B

        # Mask out diagonal (self-similarity = 1.0)
        mask = ~torch.eye(n, device=z.device).bool()
        avg_cosine_sim = sim_matrix[mask].mean().item()

        return {'embed_std': embed_std, 'avg_cosine_sim': avg_cosine_sim}

