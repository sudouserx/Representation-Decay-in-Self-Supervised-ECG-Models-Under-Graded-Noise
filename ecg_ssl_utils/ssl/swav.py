"""
SwAV: Swapping Assignments between Views.
Prototype-based contrastive with Sinkhorn-Knopp.
Reference: Caron et al., NeurIPS 2020.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional


def sinkhorn(Q, niters=3, epsilon=0.05):
    """Sinkhorn-Knopp for balanced cluster assignment."""
    with torch.no_grad():
        Q = torch.exp(Q / epsilon).T  # (K, B)
        Q /= Q.sum()
        K, B = Q.shape
        for _ in range(niters):
            Q /= Q.sum(dim=1, keepdim=True) * K  # row norm
            Q /= Q.sum(dim=0, keepdim=True) * B  # col norm
    return Q.T  # (B, K)


class SwAVTrainer:
    def __init__(self, encoder, projector, prototypes, augmentation,
                 optimizer, scheduler=None, temperature=0.1,
                 sinkhorn_iters=3, sinkhorn_eps=0.05,
                 use_amp=True, device='cuda', grad_clip_norm=1.0):
        self.encoder = encoder.to(device)
        self.projector = projector.to(device)
        self.prototypes = prototypes.to(device)
        self.augmentation = augmentation
        self.opt = optimizer
        self.sched = scheduler
        self.tau = temperature
        self.sk_iters = sinkhorn_iters
        self.sk_eps = sinkhorn_eps
        self.amp = use_amp
        self.dev = device
        self.grad_clip_norm = grad_clip_norm
        self.scaler = torch.amp.GradScaler('cuda') if use_amp else None

    def _swav_loss(self, z1, z2):
        """Swapped prediction loss."""
        # Normalize features
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        # Prototype scores
        s1 = self.prototypes(z1)  # (B, K)
        s2 = self.prototypes(z2)
        # Codes via Sinkhorn
        q1 = sinkhorn(s1.detach(), self.sk_iters, self.sk_eps)
        q2 = sinkhorn(s2.detach(), self.sk_iters, self.sk_eps)
        # Cross-entropy losses
        p1 = F.log_softmax(s1 / self.tau, dim=1)
        p2 = F.log_softmax(s2 / self.tau, dim=1)
        loss = -0.5 * (q2 * p1 + q1 * p2).sum(dim=1).mean()
        return loss

    def train_step(self, batch):
        self.encoder.train(); self.projector.train()
        batch = batch.to(self.dev)
        v1, v2 = self.augmentation(batch), self.augmentation(batch)
        self.opt.zero_grad()
        with torch.amp.autocast('cuda', enabled=self.amp):
            z1 = self.projector(self.encoder(v1))
            z2 = self.projector(self.encoder(v2))
            loss = self._swav_loss(z1, z2)
        _params = (list(self.encoder.parameters()) +
                   list(self.projector.parameters()) +
                   list(self.prototypes.parameters()))
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(_params, self.grad_clip_norm)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(_params, self.grad_clip_norm)
            self.opt.step()
        # Normalize prototypes
        with torch.no_grad():
            w = self.prototypes.prototypes.weight.data
            self.prototypes.prototypes.weight.copy_(F.normalize(w, dim=1))
        return loss.item()
