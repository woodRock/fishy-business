# -*- coding: utf-8 -*-
"""
AugFormerV4: High-Efficiency Spectral Transformer.

Incorporates SOTA techniques from modded-nanogpt and Parameter Golf:
- Grouped-Query Attention (GQA): Decoupled Q/KV heads for parameter efficiency.
- Zero-Init Projections: Blocks start as identity, stabilizing early training.
- Learnable Residual Scales: Per-dimension scaling of attention and MLP outputs.
- x0 Routing (Residual Mix): Injects pristine embedding features into deep layers.
- QK-norm & Logit Soft-capping: Continued from V3 for maximum stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from .ttt_mixin import TTTMixin
from .augformer import RMSNorm, SpectralGating, _random_augment, _INFERENCE_AUGS


class MultiHeadAttentionV4(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        use_xsa: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.use_xsa = use_xsa

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Zero-init output projection: block starts as identity
        nn.init.zeros_(self.o_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(x)
            .reshape(B, N, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .reshape(B, N, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # QK-norm
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        # Handle GQA broadcasting if needed (for XSA compatibility)
        if self.num_heads != self.num_kv_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        y = attn @ v

        if self.use_xsa:
            Vn = torch.nn.functional.normalize(v, dim=-1)
            y = y - (y * Vn).sum(dim=-1, keepdim=True) * Vn

        y = y.transpose(1, 2).reshape(B, N, C)
        return self.o_proj(y)


class MLP_V4(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int):
        super().__init__()
        hidden = dim * mlp_ratio
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

        # Zero-init output projection
        nn.init.zeros_(self.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: w2(silu(w1(x)) * w3(x))
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlockV4(nn.Module):
    """
    Advanced block with Learnable Residual Scales and x0 Routing.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        use_xsa: bool = False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = MultiHeadAttentionV4(
            embed_dim, num_heads, num_kv_heads, use_xsa=use_xsa
        )
        self.norm2 = RMSNorm(embed_dim)
        self.mlp = MLP_V4(embed_dim, mlp_ratio)

        # Learnable residual scales (per-dimension)
        self.attn_scale = nn.Parameter(torch.ones(embed_dim))
        self.mlp_scale = nn.Parameter(torch.ones(embed_dim))

        # Residual mix (x0 routing): [0] is x, [1] is x0
        self.resid_mix = nn.Parameter(
            torch.stack([torch.ones(embed_dim), torch.zeros(embed_dim)])
        )

        self.post_norm = RMSNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        # Route original features
        mix = self.resid_mix.to(dtype=x.dtype)
        x_routed = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        # Attention path
        x = x + self.attn_scale[None, None, :] * self.drop(
            self.attn(self.norm1(x_routed))
        )

        # MLP path
        x = x + self.mlp_scale[None, None, :] * self.drop(self.mlp(self.norm2(x)))

        return self.post_norm(x)


class AugFormerV4(nn.Module, TTTMixin):
    """
    State-of-the-Art AugFormer incorporating modern LLM structural advancements.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        num_kv_heads: int = 4,
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
                TransformerBlockV4(
                    hidden_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_ratio=2,
                    dropout=dropout,
                    use_xsa=use_xsa,
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

        views = self._build_views(x)
        tokens = self.view_embed(views)
        tokens = self.pre_gate(tokens)

        tokens[:, 0:1] = tokens[:, 0:1] + self.original_embed
        tokens[:, 1:] = tokens[:, 1:] + self.aug_embed

        # Anchor residual and x0 for routing
        anchor_residual = tokens[:, 0:1].clone()
        x0 = tokens.clone()

        for block in self.blocks:
            tokens = block(tokens, x0)

        tokens[:, 0:1] = tokens[:, 0:1] + anchor_residual
        tokens = self.norm(tokens)
        logits = self.fc_out(tokens[:, 0])

        # Logit soft-capping
        if self.logit_cap > 0:
            logits = self.logit_cap * torch.tanh(logits / self.logit_cap)

        if return_attention:
            return logits, []
        return logits


__all__ = ["AugFormerV4"]
