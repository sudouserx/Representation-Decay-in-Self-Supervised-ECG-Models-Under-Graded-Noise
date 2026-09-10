"""
JEPA: Joint-Embedding Predictive Architecture for 1D ECG.
Predicts target embeddings in latent space using EMA target encoder.
Loss: L = E[‖ŝ_y - sg(s_y)‖²]
Reference: Assran et al., CVPR 2023 (I-JEPA).
"""
import torch, torch.nn as nn, copy, math
from typing import Optional, List, Tuple
from ecg_ssl_utils.repro import accumulation_state


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
    """I-JEPA–adapted: target encoder sees target patches + CLS (not the full
    sequence); loss is L2. Disclosed as an adaptation of Assran et al. 2023.
    """
    def __init__(self, encoder, predictor, ema_momentum=0.996,
                 num_target_blocks=4, target_block_size_range=(10, 15)):
        super().__init__()
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        self.predictor = predictor
        self.ema_m = ema_momentum
        self.num_target_blocks = num_target_blocks
        self.target_block_size_range = tuple(target_block_size_range)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self._last_ctx_repr = None

    @torch.no_grad()
    def update_target(self, m=None):
        if m is None: m = self.ema_m
        for p_t, p_c in zip(self.target_encoder.parameters(),
                            self.context_encoder.parameters()):
            p_t.data.mul_(m).add_(p_c.data, alpha=1-m)

    def _sample_blocks_batch(self, B, N, device):
        """Per-sample contiguous target blocks with a shared token count.

        Block size is fixed to the midpoint of *target_block_size_range* so
        the batch can be gathered with advanced indexing.
        """
        lo, hi = self.target_block_size_range
        bsz = max(1, (lo + hi) // 2)
        n_blocks = self.num_target_blocks
        max_start = max(1, N - bsz)
        starts = torch.randint(0, max_start, (B, n_blocks), device=device)
        offsets = torch.arange(bsz, device=device)[None, None, :]
        tgt = (starts.unsqueeze(-1) + offsets).reshape(B, n_blocks * bsz)
        tgt = tgt.clamp(0, N - 1)
        # Context = complement via boolean mask
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, tgt, False)
        n_ctx = int(mask.sum(dim=1).min().item())
        ctx_ids = torch.stack([
            mask[b].nonzero(as_tuple=False).squeeze(-1)[:n_ctx] for b in range(B)
        ], dim=0)
        n_tgt = n_blocks * bsz
        tgt_ids = tgt[:, :n_tgt]
        return ctx_ids, tgt_ids

    def forward(self, x):
        B, C, L = x.shape
        N = self.context_encoder.num_patches
        device = x.device

        ctx_ids, tgt_ids = self._sample_blocks_batch(B, N, device)

        # Full patch tokens
        patches = self.context_encoder.patch_embed(x)  # (B,N,D)

        # Context encoding (per-sample indices: (B, n_ctx))
        ctx = torch.gather(patches, 1, ctx_ids.unsqueeze(-1).expand(-1, -1, patches.shape[-1]))
        cls = self.context_encoder.cls_token.expand(B,-1,-1)
        ctx = torch.cat([cls, ctx], 1)
        cpos = self.context_encoder.pos_embed[:,:1,:].expand(B,-1,-1)
        vpos = torch.gather(
            self.context_encoder.pos_embed[:,1:,:].expand(B,-1,-1),
            1, ctx_ids.unsqueeze(-1).expand(-1, -1, patches.shape[-1]),
        )
        ctx = ctx + torch.cat([cpos, vpos], 1)
        for blk in self.context_encoder.blocks:
            ctx = blk(ctx)
        ctx = self.context_encoder.norm(ctx)[:,1:,:]
        self._last_ctx_repr = ctx.mean(dim=1).detach()

        with torch.no_grad():
            tgt_patches = torch.gather(
                patches, 1, tgt_ids.unsqueeze(-1).expand(-1, -1, patches.shape[-1]),
            )
            cls_t = self.target_encoder.cls_token.expand(B,-1,-1)
            tgt_in = torch.cat([cls_t, tgt_patches], 1)
            tpos = torch.gather(
                self.target_encoder.pos_embed[:,1:,:].expand(B,-1,-1),
                1, tgt_ids.unsqueeze(-1).expand(-1, -1, patches.shape[-1]),
            )
            tgt_in = tgt_in + torch.cat([
                self.target_encoder.pos_embed[:,:1,:].expand(B,-1,-1),
                tpos,
            ], 1)
            for blk in self.target_encoder.blocks:
                tgt_in = blk(tgt_in)
            s_y = self.target_encoder.norm(tgt_in)[:,1:,:]

        tgt_pos = torch.gather(
            self.context_encoder.pos_embed[:,1:,:].expand(B,-1,-1),
            1, tgt_ids.unsqueeze(-1).expand(-1, -1, patches.shape[-1]),
        )
        s_y_hat = self.predictor(ctx, tgt_pos)

        # L2 loss
        loss = ((s_y_hat - s_y.detach())**2).mean()
        return loss


class JEPATrainer:
    def __init__(self, model, optimizer, scheduler=None,
                 ema_start=0.996, ema_end=1.0, total_epochs=200,
                 use_amp=True, device='cuda', grad_clip_norm=1.0,
                 grad_accum_steps=1):
        self.model = model.to(device)
        self.opt = optimizer
        self.sched = scheduler
        self.ema_start = ema_start
        self.ema_end = ema_end
        self.total_epochs = total_epochs
        self.amp = use_amp
        self.dev = device
        self.grad_clip_norm = grad_clip_norm
        self.grad_accum_steps = grad_accum_steps
        self.scaler = torch.amp.GradScaler('cuda') if use_amp else None

    def get_ema_momentum(self, epoch):
        """Cosine EMA schedule: 0.996 → 1.0"""
        return 1 - (1 - self.ema_start) * 0.5 * (1 + math.cos(math.pi * epoch / self.total_epochs))

    def train_step(self, batch, epoch=0, step_idx=0, total_steps=1):
        self.model.train()
        batch = batch.to(self.dev)
        if step_idx % self.grad_accum_steps == 0:
            self.opt.zero_grad()
        window_size, do_step = accumulation_state(
            step_idx, total_steps, self.grad_accum_steps,
        )
        with torch.amp.autocast('cuda', enabled=self.amp):
            loss = self.model(batch)
            loss_scaled = loss / window_size
        _params = (list(self.model.context_encoder.parameters()) +
                   list(self.model.predictor.parameters()))
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
            m = self.get_ema_momentum(epoch)
            self.model.update_target(m)
        return loss.item()

    @torch.no_grad()
    def compute_collapse_metrics(self):
        z = self.model._last_ctx_repr
        if z is None:
            return {'embed_std': float('nan'), 'avg_cosine_sim': float('nan'),
                    'var_below_1e-4': 0, 'cov_abs_mean': float('nan')}
        z = z.float()
        std = z.std(dim=0).mean().item()
        zn = torch.nn.functional.normalize(z, dim=1)
        sim = zn @ zn.t()
        n = z.shape[0]
        mask = ~torch.eye(n, device=z.device).bool()
        cos = sim[mask].mean().item()
        var = z.var(dim=0)
        if n > 1:
            cov = torch.cov(z.T)
            cov = cov.clone()
            cov.fill_diagonal_(0)
            cov_abs = cov.abs().mean().item()
        else:
            cov_abs = float('nan')
        return {
            'embed_std': std,
            'avg_cosine_sim': cos,
            'var_below_1e-4': int((var < 1e-4).sum().item()),
            'cov_abs_mean': cov_abs,
        }
