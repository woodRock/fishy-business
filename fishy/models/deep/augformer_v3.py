# -*- coding: utf-8 -*-
"""
AugFormerV3: The "Modern Baseline" Augmentation-as-Sequence Transformer.

A copy of the original AugFormer updated with:
- QK-norm: RMSNorm on Query and Key vectors to stabilize attention.
- Logit Soft-capping: Tanh-based capping for robust training.
- Dual Norm: Pre-norm for stability and Post-norm for feature preservation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from .ttt_mixin import TTTMixin
from .augformer import RMSNorm, SpectralGating, _random_augment, _INFERENCE_AUGS


class MultiHeadAttentionV3(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, use_xsa: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_xsa = use_xsa
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        # QK-norm
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        y = attn @ v

        if self.use_xsa:
            Vn = torch.nn.functional.normalize(v, dim=-1)
            y = y - (y * Vn).sum(dim=-1, keepdim=True) * Vn

        x = y.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlockV3(nn.Module):
    """Dual-norm block: Pre-RMSNorm → Attention/MLP → Post-RMSNorm."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        use_xsa: bool = False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = MultiHeadAttentionV3(embed_dim, num_heads, use_xsa=use_xsa)
        self.norm2 = RMSNorm(embed_dim)

        hidden = embed_dim * mlp_ratio
        self.w1 = nn.Linear(embed_dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, embed_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, hidden, bias=False)

        self.post_norm = RMSNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm branch
        x = x + self.drop(self.attn(self.norm1(x)))

        # SwiGLU FFN
        h = self.norm2(x)
        x = x + self.drop(self.w2(F.silu(self.w1(h)) * self.w3(h)))

        # Post-norm step
        return self.post_norm(x)


class AugFormerV3(nn.Module, TTTMixin):
    """
    AugFormerV3 with QK-norm, Logit Capping, and Dual Norm.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.3,
        num_views: int = 6,
        use_xsa: bool = False,
        logit_cap: float = 30.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_views = num_views
        self.logit_cap = logit_cap

        self.view_embed = nn.Linear(input_dim, hidden_dim, bias=False)
        self.pre_gate = nn.Sequential(
            SpectralGating(hidden_dim, hidden_dim * 2, dropout),
            SpectralGating(hidden_dim, hidden_dim * 2, dropout),
        )

        self.original_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.aug_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.original_embed, std=0.02)
        nn.init.trunc_normal_(self.aug_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerBlockV3(
                    hidden_dim, num_heads, mlp_ratio=2, dropout=dropout, use_xsa=use_xsa
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = RMSNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim, bias=False)

    def _build_views(self, x: torch.Tensor) -> torch.Tensor:
        views = [x]
        if self.training:
            for _ in range(self.num_views):
                views.append(_random_augment(x))
        else:
            for i in range(self.num_views):
                aug_fn = _INFERENCE_AUGS[i % len(_INFERENCE_AUGS)]
                views.append(aug_fn(x))
        return torch.stack(views, dim=1)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False, *args, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        if x.dim() == 3:
            x = x.squeeze(1)

        B = x.shape[0]
        views = self._build_views(x)
        tokens = self.view_embed(views)
        tokens = self.pre_gate(tokens)

        tokens[:, 0:1] = tokens[:, 0:1] + self.original_embed
        tokens[:, 1:] = tokens[:, 1:] + self.aug_embed

        anchor_residual = tokens[:, 0:1].clone()

        for block in self.blocks:
            tokens = block(tokens)

        tokens[:, 0:1] = tokens[:, 0:1] + anchor_residual
        tokens = self.norm(tokens)
        logits = self.fc_out(tokens[:, 0])

        # Logit soft-capping
        if self.logit_cap > 0:
            logits = self.logit_cap * torch.tanh(logits / self.logit_cap)

        if return_attention:
            return logits, []
        return logits


__all__ = ["AugFormerV3"]
