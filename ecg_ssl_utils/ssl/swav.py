"""
SwAV: Swapping Assignments between Views.
Prototype-based contrastive with Sinkhorn-Knopp.
Reference: Caron et al., NeurIPS 2020.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional
from ecg_ssl_utils.repro import accumulation_state


def sinkhorn(Q, niters=3, epsilon=0.05):
    """Sinkhorn-Knopp for balanced cluster assignment."""
    with torch.no_grad():
        Q = torch.exp(Q / epsilon).T  # (K, B)
        Q /= Q.sum()
        K, B = Q.shape
        for _ in range(niters):
            Q /= Q.sum(dim=1, keepdim=True) * K  # row norm
            Q /= Q.sum(dim=0, keepdim=True) * B  # col norm
        Q *= B  # columns (samples) must sum to 1
    return Q.T  # (B, K)


class SwAVTrainer:
    def __init__(self, encoder, projector, prototypes, augmentation,
                 optimizer, scheduler=None, temperature=0.1,
                 sinkhorn_iters=3, sinkhorn_eps=0.05,
                 use_amp=True, device='cuda', grad_clip_norm=1.0,
                 grad_accum_steps=1):
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
        self.grad_accum_steps = grad_accum_steps
        self.scaler = torch.amp.GradScaler('cuda') if use_amp else None
        self._last_z = None
        self._last_assignments = None

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
        self._last_assignments = torch.cat([q1, q2], dim=0)
        # Cross-entropy losses
        p1 = F.log_softmax(s1 / self.tau, dim=1)
        p2 = F.log_softmax(s2 / self.tau, dim=1)
        loss = -0.5 * (q2 * p1 + q1 * p2).sum(dim=1).mean()
        return loss

    def train_step(self, batch, step_idx=0, total_steps=1):
        self.encoder.train(); self.projector.train()
        batch = batch.to(self.dev)
        v1, v2 = self.augmentation(batch), self.augmentation(batch)
        if step_idx % self.grad_accum_steps == 0:
            self.opt.zero_grad()
        window_size, do_step = accumulation_state(
            step_idx, total_steps, self.grad_accum_steps,
        )
        with torch.amp.autocast('cuda', enabled=self.amp):
            h1 = self.encoder(v1)
            h2 = self.encoder(v2)
            z1 = self.projector(h1)
            z2 = self.projector(h2)
            loss = self._swav_loss(z1, z2)
            loss_scaled = loss / window_size
        self._last_z = torch.cat([h1.detach(), h2.detach()], dim=0)
        _params = (list(self.encoder.parameters()) +
                   list(self.projector.parameters()) +
                   list(self.prototypes.parameters()))
        if self.scaler:
            self.scaler.scale(loss_scaled).backward()
            if do_step:
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(_params, self.grad_clip_norm)
                self.scaler.step(self.opt)
                self.scaler.update()
        else:
            loss_scaled.backward()
            if do_step:
                nn.utils.clip_grad_norm_(_params, self.grad_clip_norm)
                self.opt.step()
        with torch.no_grad():
            w = self.prototypes.prototypes.weight.data
            self.prototypes.prototypes.weight.copy_(F.normalize(w, dim=1))
        return loss.item()

    @torch.no_grad()
    def compute_collapse_metrics(self):
        if self._last_z is None:
            return {'embed_std': float('nan'), 'avg_cosine_sim': float('nan')}
        z = self._last_z.float()
        embed_std = z.std(dim=0).mean().item()
        zn = F.normalize(z, dim=1)
        sim = zn @ zn.t()
        n = z.shape[0]
        mask = ~torch.eye(n, device=z.device).bool()
        assignments = self._last_assignments.float()
        usage = assignments.mean(dim=0)
        entropy = -(usage * usage.clamp_min(1e-12).log()).sum()
        entropy /= torch.log(torch.tensor(float(len(usage)), device=usage.device))
        occupied = (usage > (1.0 / len(usage)) * 0.1).float().mean()
        return {
            'embed_std': embed_std,
            'avg_cosine_sim': sim[mask].mean().item(),
            'prototype_entropy': entropy.item(),
            'prototype_occupancy': occupied.item(),
        }
