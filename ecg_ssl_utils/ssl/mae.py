"""
MAE (Masked Autoencoder) for 1D ECG.
Loss: MSE on masked patches. Reference: He et al., CVPR 2022.
"""
import torch
import torch.nn as nn
from ecg_ssl_utils.repro import accumulation_state


class MAEDecoder(nn.Module):
    def __init__(self, num_patches=100, enc_dim=384, dec_dim=192,
                 depth=4, nheads=4, patch_size=50, in_ch=12, mlp_ratio=4.0):
        super().__init__()
        self.dec_dim = dec_dim
        self.patch_dim = patch_size * in_ch
        self.embed = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.pos = nn.Parameter(torch.zeros(1, num_patches, dec_dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dec_dim, nheads, int(dec_dim*mlp_ratio),
                                       dropout=0., activation='gelu',
                                       batch_first=True, norm_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dec_dim)
        self.pred = nn.Linear(dec_dim, self.patch_dim)
        nn.init.trunc_normal_(self.mask_token, std=.02)
        nn.init.trunc_normal_(self.pos, std=.02)

    def forward(self, enc_vis, vis_ids, mask_ids, n_patches):
        B = enc_vis.shape[0]
        x = self.embed(enc_vis)
        # AMP: Linear outputs fp16/bf16 while nn.Parameters stay fp32;
        # scatter_ is not autocast-aware and requires matching dtypes.
        dtype = x.dtype
        mt = self.mask_token.to(dtype=dtype).expand(B, mask_ids.shape[1], -1)
        full = torch.zeros(B, n_patches, self.dec_dim, device=x.device, dtype=dtype)
        vi = vis_ids.unsqueeze(-1).expand(-1,-1,self.dec_dim)
        mi = mask_ids.unsqueeze(-1).expand(-1,-1,self.dec_dim)
        full.scatter_(1, vi, x)
        full.scatter_(1, mi, mt)
        full = full + self.pos.to(dtype=dtype)
        for blk in self.blocks:
            full = blk(full)
        full = self.norm(full)
        pred = self.pred(full)
        mi2 = mask_ids.unsqueeze(-1).expand(-1,-1,pred.shape[-1])
        return torch.gather(pred, 1, mi2)


class MAEModel(nn.Module):
    def __init__(self, encoder, decoder, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        self._last_cls = None

    def _mask(self, B, N, device):
        nm = int(N * self.mask_ratio)
        nv = N - nm
        noise = torch.rand(B, N, device=device)
        ids = torch.argsort(noise, dim=1)
        return torch.sort(ids[:,:nv],1).values, torch.sort(ids[:,nv:],1).values

    def _patchify(self, x, ps=50):
        B,C,L = x.shape
        return x.reshape(B,C,L//ps,ps).permute(0,2,1,3).reshape(B,L//ps,C*ps)

    def forward(self, x):
        B,C,L = x.shape
        ps = self.encoder.patch_embed.patch_size
        N = self.encoder.num_patches
        target = self._patchify(x, ps)
        vis_ids, mask_ids = self._mask(B, N, x.device)
        ptokens = self.encoder.patch_embed(x)
        vi = vis_ids.unsqueeze(-1).expand(-1,-1,ptokens.shape[-1])
        vis_tok = torch.gather(ptokens, 1, vi)
        cls = self.encoder.cls_token.expand(B,-1,-1)
        tok = torch.cat([cls, vis_tok], 1)
        cpos = self.encoder.pos_embed[:,:1,:].expand(B,-1,-1)
        vpi = vis_ids.unsqueeze(-1).expand(-1,-1,self.encoder.embed_dim)
        vpos = torch.gather(self.encoder.pos_embed[:,1:,:].expand(B,-1,-1),1,vpi)
        tok = tok + torch.cat([cpos, vpos], 1)
        for blk in self.encoder.blocks:
            tok = blk(tok)
        tok = self.encoder.norm(tok)
        self._last_cls = tok[:, 0, :].detach()
        enc_vis = tok[:,1:,:]
        pred = self.decoder(enc_vis, vis_ids, mask_ids, N)
        mi = mask_ids.unsqueeze(-1).expand(-1,-1,target.shape[-1])
        tgt = torch.gather(target, 1, mi)
        mu = tgt.mean(-1, keepdim=True)
        var = tgt.var(-1, keepdim=True)
        tgt_n = (tgt - mu) / (var + 1e-6).sqrt()
        return ((pred - tgt_n)**2).mean(), pred, mask_ids


class MAETrainer:
    def __init__(self, model, optimizer, scheduler=None, use_amp=True, device='cuda',
                 grad_clip_norm=1.0, grad_accum_steps=1):
        self.model = model.to(device)
        self.opt = optimizer
        self.sched = scheduler
        self.amp = use_amp
        self.dev = device
        self.grad_clip_norm = grad_clip_norm
        self.grad_accum_steps = grad_accum_steps
        self.scaler = torch.amp.GradScaler('cuda') if use_amp else None

    def train_step(self, batch, step_idx=0, total_steps=1):
        self.model.train()
        batch = batch.to(self.dev)
        if step_idx % self.grad_accum_steps == 0:
            self.opt.zero_grad()
        window_size, do_step = accumulation_state(
            step_idx, total_steps, self.grad_accum_steps,
        )
        with torch.amp.autocast('cuda', enabled=self.amp):
            loss, _, _ = self.model(batch)
            loss_scaled = loss / window_size
        if self.scaler:
            self.scaler.scale(loss_scaled).backward()
            if do_step:
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.opt)
                self.scaler.update()
        else:
            loss_scaled.backward()
            if do_step:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.opt.step()
        return loss.item()

    @torch.no_grad()
    def compute_collapse_metrics(self):
        z = self.model._last_cls
        if z is None:
            return {'embed_std': float('nan'), 'avg_cosine_sim': float('nan')}
        z = z.float()
        std = z.std(dim=0).mean().item()
        zn = torch.nn.functional.normalize(z, dim=1)
        sim = zn @ zn.T
        mask = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
        return {'embed_std': std, 'avg_cosine_sim': sim[mask].mean().item()}

    @torch.no_grad()
    def validation_loss(self, batch, seed=0):
        """Deterministic masked-reconstruction loss on a held-out batch."""
        self.model.eval()
        devices = [torch.cuda.current_device()] if self.dev.startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            loss, _, _ = self.model(batch.to(self.dev))
        return float(loss.item())
