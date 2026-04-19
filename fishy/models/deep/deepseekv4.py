# -*- coding: utf-8 -*-
"""
DeepSeekV4: Advanced Spectral Transformer with MLA, mHC, and Engram Memory.

Inspired by DeepSeek-V3/V4 and the Frankenstein models from Parameter Golf.
Optimized for high-parameter efficiency and robust feature preservation in
REIMS spectral data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from .ttt_mixin import TTTMixin
from .augformer import RMSNorm, SpectralGating, _random_augment, _INFERENCE_AUGS


# ---------------------------------------------------------------------------
# Advanced Components
# ---------------------------------------------------------------------------


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) inspired by DeepSeek-V3/V4.
    Compresses KV and Q into low-rank latent spaces.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        latent_dim: int = 64,
        kv_lora_rank: int = 64,
        qk_lora_rank: int = 64,
        use_xsa: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.latent_dim = latent_dim
        self.kv_lora_rank = kv_lora_rank
        self.qk_lora_rank = qk_lora_rank
        self.scale = self.head_dim**-0.5
        self.use_xsa = use_xsa

        # KV Compression
        self.kv_a = nn.Linear(embed_dim, kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(kv_lora_rank)
        self.kv_b = nn.Linear(kv_lora_rank, num_heads * self.head_dim * 2, bias=False)

        # Q Compression
        self.q_a = nn.Linear(embed_dim, qk_lora_rank, bias=False)
        self.q_norm = RMSNorm(qk_lora_rank)
        self.q_b = nn.Linear(qk_lora_rank, num_heads * self.head_dim, bias=False)

        self.proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Compress and Expand KV
        kv_latent = self.kv_norm(self.kv_a(x))  # [B, N, kv_lora_rank]
        kv = self.kv_b(kv_latent).reshape(B, N, self.num_heads, 2, self.head_dim)
        k, v = kv.unbind(3)  # [B, N, H, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compress and Expand Q
        q_latent = self.q_norm(self.q_a(x))  # [B, N, qk_lora_rank]
        q = self.q_b(q_latent).reshape(B, N, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # [B, H, N, D]

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        y = attn @ v  # [B, H, N, D]

        if self.use_xsa:
            Vn = torch.nn.functional.normalize(v, dim=-1)
            y = y - (y * Vn).sum(dim=-1, keepdim=True) * Vn

        x = y.transpose(1, 2).reshape(B, N, -1)
        return self.proj(x)


class ManifoldProjection(nn.Module):
    """
    Manifold-Constrained Hyper-Connection (mHC) integrator.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, stream: torch.Tensor) -> torch.Tensor:
        return self.norm(stream + torch.tanh(self.gate) * self.proj(x))


class EngramMemory(nn.Module):
    """
    Engram Memory bank for decoupled static knowledge.
    Includes a 'Trust but Verify' rejection gate for context-aware lookup,
    as specified in the DeepSeek-V4 architecture.
    """

    def __init__(self, dim: int, num_slots: int = 128):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(num_slots, dim))
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        # Rejection Gate: Context-Aware Gating
        # Learns to reject engram lookups that conflict with the global context.
        self.gate_proj = nn.Linear(dim * 2, 1, bias=False)

        self.norm = RMSNorm(dim)
        self.scale = dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(self.memory).unsqueeze(0).expand(B, -1, -1)
        v = self.v_proj(self.memory).unsqueeze(0).expand(B, -1, -1)

        # Scaled Dot-Product Attention for memory retrieval
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v

        # Rejection Gate: Trust but Verify
        # alpha -> 0: Conflict detected, reject engram signal.
        # alpha -> 1: Alignment detected, trust engram signal.
        gate_input = torch.cat([x, out], dim=-1)
        alpha = torch.sigmoid(self.gate_proj(gate_input))

        return self.norm(x + alpha * out)


class DeepSeekTransformerBlock(nn.Module):
    """DeepSeek-style block using MLA."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        use_xsa: bool = False,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = MultiHeadLatentAttention(
            embed_dim,
            num_heads,
            latent_dim=latent_dim,
            kv_lora_rank=latent_dim,
            qk_lora_rank=latent_dim // 2,
            use_xsa=use_xsa,
        )
        self.norm2 = RMSNorm(embed_dim)
        hidden = embed_dim * mlp_ratio
        self.w1 = nn.Linear(embed_dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, embed_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, hidden, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        h = self.norm2(x)
        return x + self.drop(self.w2(F.silu(self.w1(h)) * self.w3(h)))


# ---------------------------------------------------------------------------
# DeepSeekV4 Model
# ---------------------------------------------------------------------------


class DeepSeekV4(nn.Module, TTTMixin):
    """
    Advanced Spectral Transformer incorporating MLA, mHC, and Engram Memory.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        dropout: float = 0.3,
        num_views: int = 6,
        engram_slots: int = 128,
        latent_dim: int = 64,
        use_xsa: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_views = num_views

        self.view_embed = nn.Linear(input_dim, hidden_dim, bias=False)
        self.pre_gate = nn.Sequential(
            SpectralGating(hidden_dim, hidden_dim * 2, dropout),
            SpectralGating(hidden_dim, hidden_dim * 2, dropout),
        )

        self.engram = EngramMemory(hidden_dim, num_slots=engram_slots)

        self.original_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.aug_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.original_embed, std=0.02)
        nn.init.trunc_normal_(self.aug_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                DeepSeekTransformerBlock(
                    hidden_dim,
                    num_heads,
                    mlp_ratio=2,
                    dropout=dropout,
                    use_xsa=use_xsa,
                    latent_dim=latent_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.mhc_projs = nn.ModuleList(
            [ManifoldProjection(hidden_dim) for _ in range(num_layers)]
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

        # DeepSeek-v4 additions
        tokens = self.engram(tokens)

        tokens[:, 0:1] = tokens[:, 0:1] + self.original_embed
        tokens[:, 1:] = tokens[:, 1:] + self.aug_embed

        anchor_residual = tokens[:, 0:1].clone()

        # mHC hyper-stream
        hyper_stream = tokens.clone()
        for i, block in enumerate(self.blocks):
            tokens = block(tokens)
            hyper_stream = self.mhc_projs[i](tokens, hyper_stream)
        tokens = hyper_stream

        tokens[:, 0:1] = tokens[:, 0:1] + anchor_residual
        tokens = self.norm(tokens)
        logits = self.fc_out(tokens[:, 0])

        if return_attention:
            return logits, []
        return logits


__all__ = ["DeepSeekV4"]
