"""
1D Vision Transformer (ViT-Small) for ECG signals.

Architecture:
- Patch embedding: Conv1D(in=12, out=384, kernel=50, stride=50)
- 100 patches + 1 CLS token = 101 tokens
- 12 transformer layers, 384-d, 6 heads
- ~22M parameters
"""

import torch
import torch.nn as nn
import numpy as np
from functools import partial


class PatchEmbed1D(nn.Module):
    """1D patch embedding using Conv1D."""

    def __init__(
        self,
        in_channels: int = 12,
        embed_dim: int = 384,
        patch_size: int = 50,
        signal_length: int = 5000,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = signal_length // patch_size
        self.proj = nn.Conv1d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, C, L) where C=12 leads, L=5000.

        Returns
        -------
        patches : torch.Tensor
            Shape (B, num_patches, embed_dim).
        """
        x = self.proj(x)       # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 6, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feedforward MLP block."""

    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class TransformerBlock(nn.Module):
    """Single transformer block: LayerNorm → Attention → LayerNorm → MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViTSmall1D(nn.Module):
    """
    Vision Transformer Small for 1D ECG signals.

    Architecture:
        - Input: (B, 12, 5000) — 12-lead ECG at 500 Hz
        - Patch embed: Conv1D → (B, 100, 384)
        - CLS token prepended → (B, 101, 384)
        - 12 transformer blocks
        - Output: CLS token → (B, 384)
    """

    def __init__(
        self,
        in_channels: int = 12,
        signal_length: int = 5000,
        patch_size: int = 50,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = signal_length // patch_size

        # Patch embedding
        self.patch_embed = PatchEmbed1D(in_channels, embed_dim, patch_size, signal_length)

        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)

    @staticmethod
    def _init_module_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features (all tokens including CLS).

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 12, 5000).

        Returns
        -------
        tokens : torch.Tensor
            Shape (B, num_patches+1, embed_dim).
        """
        B = x.shape[0]

        # Patch embed
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning CLS token embedding.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 12, 5000).

        Returns
        -------
        cls_embedding : torch.Tensor
            Shape (B, embed_dim=384).
        """
        x = self.forward_features(x)
        return x[:, 0]  # CLS token

    def get_intermediate_layers(self, x: torch.Tensor, n: int = 1):
        """Get output from last n transformer blocks (for analysis)."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i >= len(self.blocks) - n:
                outputs.append(self.norm(x))
        return outputs
