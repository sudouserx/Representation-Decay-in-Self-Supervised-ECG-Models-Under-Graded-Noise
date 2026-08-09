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
                 total_epochs=200, use_amp=True, device='cuda'):
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
        self.scaler = torch.amp.GradScaler('cuda') if use_amp else None

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

    def train_step(self, batch, epoch=0):
        self.online_encoder.train(); self.online_projector.train(); self.predictor.train()
        batch = batch.to(self.dev)
        v1, v2 = self.augmentation(batch), self.augmentation(batch)
        self.opt.zero_grad()
        with torch.amp.autocast('cuda', enabled=self.amp):
            o1 = self.predictor(self.online_projector(self.online_encoder(v1)))
            o2 = self.predictor(self.online_projector(self.online_encoder(v2)))
            with torch.no_grad():
                t1 = self.target_projector(self.target_encoder(v1))
                t2 = self.target_projector(self.target_encoder(v2))
            loss = self._loss(o1, t2.detach()) + self._loss(o2, t1.detach())
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward(); self.opt.step()
        m = 1-(1-self.ema_s)*0.5*(1+math.cos(math.pi*epoch/self.T))
        self._update_target(m)
        return loss.item()
