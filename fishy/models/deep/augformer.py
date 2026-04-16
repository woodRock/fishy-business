# -*- coding: utf-8 -*-
"""
AugFormer: Multi-View Self-Attention over Augmented Spectral Views.

The core insight: instead of splitting the spectrum into spatial patches (which
requires large datasets for attention to learn), generate K augmented views of
the same spectrum and treat each view as a token. The transformer can then attend
across views and learn which features are *consistent* across noise, scale, and
shift — i.e., the features that are invariant to plausible measurement variation,
which are exactly the robust, discriminative features for classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union
from .ttt_mixin import TTTMixin


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SpectralGating(nn.Module):
    """Coral-style gating applied spectral-wise to each view independently."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return residual + self.drop(x)


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------


def _noise(x: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return torch.clamp(x + torch.randn_like(x) * std, 0.0)


def _scale(x: torch.Tensor, factor: float) -> torch.Tensor:
    return x * factor


def _shift(x: torch.Tensor, bins: int) -> torch.Tensor:
    return torch.roll(x, shifts=bins, dims=-1)


def _crop(x: torch.Tensor, crop_size: float = 0.85, start: int = None) -> torch.Tensor:
    n = x.shape[-1]
    mask_len = int(n * (1.0 - crop_size))
    if mask_len == 0:
        return x
    out = x.clone()
    if start is None:
        start = torch.randint(0, n - mask_len + 1, (1,)).item()
    start = max(0, min(start, n - mask_len))
    out[..., start : start + mask_len] = 0.0
    return out


def _random_augment(x: torch.Tensor) -> torch.Tensor:
    aug = x.clone()
    applied = []
    if torch.rand(1).item() < 0.6:
        aug = _noise(aug, std=torch.empty(1).uniform_(0.005, 0.04).item())
        applied.append("noise")
    if torch.rand(1).item() < 0.5:
        factor = torch.empty(1).uniform_(0.85, 1.15).item()
        aug = _scale(aug, factor)
        applied.append("scale")
    if torch.rand(1).item() < 0.5:
        bins = torch.randint(-20, 21, (1,)).item()
        aug = _shift(aug, bins)
        applied.append("shift")
    if torch.rand(1).item() < 0.4:
        aug = _crop(aug, crop_size=torch.empty(1).uniform_(0.80, 0.95).item())
        applied.append("crop")
    if not applied:  # guarantee at least one augmentation
        aug = _noise(aug, std=0.02)
    return aug


_INFERENCE_AUGS = [
    lambda x: _noise(x, std=0.01),  # mild noise
    lambda x: _noise(x, std=0.03),  # moderate noise
    lambda x: _scale(x, 0.92),  # scale down
    lambda x: _scale(x, 1.08),  # scale up
    lambda x: _shift(x, -10),  # shift left
    lambda x: _shift(x, +10),  # shift right
    lambda x: _crop(x, crop_size=0.88, start=100),  # crop left region
    lambda x: _crop(x, crop_size=0.88, start=1000),  # crop mid region
]


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
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
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        y = attn @ v

        if self.use_xsa:
            # XSA: Exclusive Self Attention
            # Orthogonalize the output against the value vectors to focus on contextual information
            Vn = torch.nn.functional.normalize(v, dim=-1)
            y = y - (y * Vn).sum(dim=-1, keepdim=True) * Vn

        x = y.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) inspired by DeepSeek-V3/V4.
    Compresses KV and Q into low-rank latent spaces to improve parameter efficiency
    and force more robust feature extraction from spectral tokens.
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
    Ensures that features from all depths are preserved and projected onto
    the optimal representation manifold.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, stream: torch.Tensor) -> torch.Tensor:
        # Weighted integration of the layer output into the hyper-stream
        return self.norm(stream + torch.tanh(self.gate) * self.proj(x))


class EngramMemory(nn.Module):
    """
    Engram Memory bank inspired by DeepSeek-V4.
    Decouples static spectral knowledge (peak signatures) from dynamic reasoning.
    Tokens query a learnable memory bank to retrieve "known" spectral patterns.
    """

    def __init__(self, dim: int, num_slots: int = 128):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(num_slots, dim))
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.norm = RMSNorm(dim)
        self.scale = dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # x are queries, memory are keys/values
        q = self.q_proj(x)
        k = self.k_proj(self.memory).unsqueeze(0).expand(B, -1, -1)
        v = self.v_proj(self.memory).unsqueeze(0).expand(B, -1, -1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        return self.norm(x + out)


class TransformerBlock(nn.Module):
    """Pre-norm block: RMSNorm → Attention → RMSNorm → SwiGLU FFN."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        use_xsa: bool = False,
        use_mla: bool = False,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        if use_mla:
            self.attn = MultiHeadLatentAttention(
                embed_dim,
                num_heads,
                latent_dim=latent_dim,
                kv_lora_rank=latent_dim,
                qk_lora_rank=latent_dim // 2,
                use_xsa=use_xsa,
            )
        else:
            self.attn = MultiHeadAttention(embed_dim, num_heads, use_xsa=use_xsa)
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
# AugFormer
# ---------------------------------------------------------------------------


class AugFormer(nn.Module, TTTMixin):
    """
    Augmentation-as-Sequence Transformer for REIMS spectral classification.
    Iteration 1 Baseline: Anchor-centric, Deep Spectral Gating, no tokenizer.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.3,
        num_views: int = 6,  # augmented views (+ original = 7 spec tokens)
        use_xsa: bool = False,
        use_mla: bool = False,
        use_mhc: bool = False,
        use_engram: bool = False,
        engram_slots: int = 128,
        latent_dim: int = 64,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_views = num_views
        self.use_mhc = use_mhc

        self.view_embed = nn.Linear(input_dim, hidden_dim, bias=False)
        self.pre_gate = nn.Sequential(
            SpectralGating(hidden_dim, hidden_dim * 2, dropout),
            SpectralGating(hidden_dim, hidden_dim * 2, dropout),
        )

        if use_engram:
            self.engram = EngramMemory(hidden_dim, num_slots=engram_slots)
        else:
            self.engram = None

        self.original_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.aug_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.original_embed, std=0.02)
        nn.init.trunc_normal_(self.aug_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim,
                    num_heads,
                    mlp_ratio=2,
                    dropout=dropout,
                    use_xsa=use_xsa,
                    use_mla=use_mla,
                    latent_dim=latent_dim,
                )
                for _ in range(num_layers)
            ]
        )

        if use_mhc:
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
        return torch.stack(views, dim=1)  # [B, V+1, F]

    def forward(
        self, x: torch.Tensor, return_attention: bool = False, *args, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        if x.dim() == 3:
            x = x.squeeze(1)

        B = x.shape[0]
        views = self._build_views(x)  # [B, V+1, F]
        tokens = self.view_embed(views)

        # Apply shared spectral gating to each view independently
        tokens = self.pre_gate(tokens)

        # Retrieve static knowledge from Engram Memory
        if self.engram is not None:
            tokens = self.engram(tokens)

        # Mark original vs augmented
        tokens[:, 0:1] = tokens[:, 0:1] + self.original_embed  # original view
        tokens[:, 1:] = tokens[:, 1:] + self.aug_embed  # all augmented views

        # Save anchor residual
        anchor_residual = tokens[:, 0:1].clone()

        if self.use_mhc:
            hyper_stream = tokens.clone()
            for i, block in enumerate(self.blocks):
                tokens = block(tokens)
                hyper_stream = self.mhc_projs[i](tokens, hyper_stream)
            tokens = hyper_stream
        else:
            for block in self.blocks:
                tokens = block(tokens)

        tokens[:, 0:1] = tokens[:, 0:1] + anchor_residual
        tokens = self.norm(tokens)
        logits = self.fc_out(tokens[:, 0])

        if return_attention:
            return logits, []
        return logits


__all__ = ["AugFormer"]
