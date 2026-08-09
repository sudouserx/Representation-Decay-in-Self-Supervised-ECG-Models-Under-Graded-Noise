"""
JEPA: Joint-Embedding Predictive Architecture for 1D ECG.
Predicts target embeddings in latent space using EMA target encoder.
Loss: L = E[‖ŝ_y - sg(s_y)‖²]
Reference: Assran et al., CVPR 2023 (I-JEPA).
"""
import torch, torch.nn as nn, copy, math
from typing import Optional, List, Tuple


class JEPAPredictor(nn.Module):
    """Transformer predictor: predicts target embeddings from context."""
    def __init__(self, dim=384, depth=4, nheads=6, mlp_ratio=4.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, nheads, int(dim*mlp_ratio),
                                       dropout=0., activation='gelu',
                                       batch_first=True, norm_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, ctx_tokens, target_pos_embed):
        # ctx_tokens: (B, n_ctx, D), target_pos_embed: (B, n_tgt, D)
        x = torch.cat([ctx_tokens, target_pos_embed], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, ctx_tokens.shape[1]:, :]  # only target predictions


class JEPAModel(nn.Module):
    def __init__(self, encoder, predictor, ema_momentum=0.996):
        super().__init__()
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        self.predictor = predictor
        self.ema_m = ema_momentum
        # Freeze target encoder
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self, m=None):
        if m is None: m = self.ema_m
        for p_t, p_c in zip(self.target_encoder.parameters(),
                            self.context_encoder.parameters()):
            p_t.data.mul_(m).add_(p_c.data, alpha=1-m)

    def _sample_blocks(self, N, n_blocks=4, block_range=(10,15), device='cuda'):
        """Sample contiguous target blocks."""
        tgt_ids = set()
        for _ in range(n_blocks):
            bsz = torch.randint(block_range[0], block_range[1]+1, (1,)).item()
            start = torch.randint(0, max(1, N - bsz), (1,)).item()
            for j in range(start, min(start+bsz, N)):
                tgt_ids.add(j)
        tgt_ids = sorted(tgt_ids)
        ctx_ids = sorted(set(range(N)) - set(tgt_ids))
        return (torch.tensor(ctx_ids, device=device),
                torch.tensor(tgt_ids, device=device))

    def forward(self, x):
        B, C, L = x.shape
        N = self.context_encoder.num_patches
        device = x.device

        ctx_ids, tgt_ids = self._sample_blocks(N, device=device)

        # Full patch tokens
        patches = self.context_encoder.patch_embed(x)  # (B,N,D)

        # Context encoding
        ctx = patches[:, ctx_ids, :]
        cls = self.context_encoder.cls_token.expand(B,-1,-1)
        ctx = torch.cat([cls, ctx], 1)
        cpos = self.context_encoder.pos_embed[:,:1,:]
        vpos = self.context_encoder.pos_embed[:,1:,:][:,ctx_ids,:]
        ctx = ctx + torch.cat([cpos.expand(B,-1,-1), vpos.expand(B,-1,-1)], 1)
        for blk in self.context_encoder.blocks:
            ctx = blk(ctx)
        ctx = self.context_encoder.norm(ctx)[:,1:,:]  # remove CLS

        # Target encoding (no grad)
        with torch.no_grad():
            tgt_patches = patches[:, tgt_ids, :]
            cls_t = self.target_encoder.cls_token.expand(B,-1,-1)
            tgt_in = torch.cat([cls_t, tgt_patches], 1)
            tpos = self.target_encoder.pos_embed[:,1:,:][:,tgt_ids,:]
            tgt_in = tgt_in + torch.cat([
                self.target_encoder.pos_embed[:,:1,:].expand(B,-1,-1),
                tpos.expand(B,-1,-1)
            ], 1)
            for blk in self.target_encoder.blocks:
                tgt_in = blk(tgt_in)
            s_y = self.target_encoder.norm(tgt_in)[:,1:,:]  # (B,n_tgt,D)

        # Predictor: predict target embeddings from context
        tgt_pos = self.context_encoder.pos_embed[:,1:,:][:,tgt_ids,:].expand(B,-1,-1)
        s_y_hat = self.predictor(ctx, tgt_pos)  # (B,n_tgt,D)

        # L2 loss
        loss = ((s_y_hat - s_y.detach())**2).mean()
        return loss


class JEPATrainer:
    def __init__(self, model, optimizer, scheduler=None,
                 ema_start=0.996, ema_end=1.0, total_epochs=200,
                 use_amp=True, device='cuda'):
        self.model = model.to(device)
        self.opt = optimizer
        self.sched = scheduler
        self.ema_start = ema_start
        self.ema_end = ema_end
        self.total_epochs = total_epochs
        self.amp = use_amp
        self.dev = device
        self.scaler = torch.amp.GradScaler('cuda') if use_amp else None

    def get_ema_momentum(self, epoch):
        """Cosine EMA schedule: 0.996 → 1.0"""
        return 1 - (1 - self.ema_start) * 0.5 * (1 + math.cos(math.pi * epoch / self.total_epochs))

    def train_step(self, batch, epoch=0):
        self.model.train()
        batch = batch.to(self.dev)
        self.opt.zero_grad()
        with torch.amp.autocast('cuda', enabled=self.amp):
            loss = self.model(batch)
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            self.opt.step()
        m = self.get_ema_momentum(epoch)
        self.model.update_target(m)
        return loss.item()
