"""
BYOL: Bootstrap Your Own Latent.
No negative pairs. Online predicts target (EMA-updated).
Loss: L = 2 - 2·cos(q(z1), z2')
Reference: Grill et al., NeurIPS 2020.
"""
import torch, torch.nn as nn, torch.nn.functional as F, copy, math
from typing import Optional


class BYOLTrainer:
    def __init__(self, encoder, projector, predictor, augmentation,
                 optimizer, scheduler=None, ema_start=0.996, ema_end=1.0,
                 total_epochs=200, use_amp=True, device='cuda',
                 grad_clip_norm=1.0, grad_accum_steps=1):
        self.online_encoder = encoder.to(device)
        self.online_projector = projector.to(device)
        self.predictor = predictor.to(device)
        self.augmentation = augmentation
        self.target_encoder = copy.deepcopy(encoder).to(device)
        self.target_projector = copy.deepcopy(projector).to(device)
        for p in self.target_encoder.parameters(): p.requires_grad = False
        for p in self.target_projector.parameters(): p.requires_grad = False
        self.opt = optimizer
        self.sched = scheduler
        self.ema_s, self.ema_e = ema_start, ema_end
        self.T = total_epochs
        self.amp = use_amp
        self.dev = device
        self.grad_clip_norm = grad_clip_norm
        self.grad_accum_steps = grad_accum_steps
        self.scaler = torch.amp.GradScaler('cuda') if use_amp else None
        self._last_z = None

    @torch.no_grad()
    def _update_target(self, m):
        for pt, po in zip(self.target_encoder.parameters(), self.online_encoder.parameters()):
            pt.data.mul_(m).add_(po.data, alpha=1-m)
        for pt, po in zip(self.target_projector.parameters(), self.online_projector.parameters()):
            pt.data.mul_(m).add_(po.data, alpha=1-m)

    def _loss(self, q, z):
        q = F.normalize(q, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2 - 2 * (q * z).sum(dim=-1).mean()

    def train_step(self, batch, epoch=0, step_idx=0):
        self.online_encoder.train(); self.online_projector.train(); self.predictor.train()
        batch = batch.to(self.dev)
        v1, v2 = self.augmentation(batch), self.augmentation(batch)
        if step_idx % self.grad_accum_steps == 0:
            self.opt.zero_grad()
        with torch.amp.autocast('cuda', enabled=self.amp):
            o1 = self.predictor(self.online_projector(self.online_encoder(v1)))
            o2 = self.predictor(self.online_projector(self.online_encoder(v2)))
            with torch.no_grad():
                t1 = self.target_projector(self.target_encoder(v1))
                t2 = self.target_projector(self.target_encoder(v2))
            loss = self._loss(o1, t2.detach()) + self._loss(o2, t1.detach())
            loss_scaled = loss / self.grad_accum_steps
        self._last_z = torch.cat([o1.detach(), o2.detach()], dim=0)
        _params = (list(self.online_encoder.parameters()) +
                   list(self.online_projector.parameters()) +
                   list(self.predictor.parameters()))
        do_step = (step_idx + 1) % self.grad_accum_steps == 0
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
        if do_step:
            m = 1-(1-self.ema_s)*0.5*(1+math.cos(math.pi*epoch/self.T))
            self._update_target(m)
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
        return {'embed_std': embed_std, 'avg_cosine_sim': sim[mask].mean().item()}
