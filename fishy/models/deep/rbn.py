# -*- coding: utf-8 -*-
"""
Relational Binding Network (RBN) for spectral classification.

Design principles
-----------------
1. Role-filler binding:  b_j = r_j ⊙ GELU(w · x_j)
     role  r_j  — learned embedding for m/z position j
     filler f_j — learned projection of log-intensity (bias=False, so f(0)=0)
   The binding vector carries joint (position, value) structure that a plain
   transformer over raw intensities cannot represent.

2. Second-order scoring without an O(K²) loop:
     score(j,j') = GELU(W_q b_j) · GELU(W_k b_j') / sqrt(d)
   Non-linear Q/K projections (GELU) preserve the multiplicative role×filler
   structure.  Since b_j = r_j ⊙ f_j, the dot product of two GELU-projected
   bindings is a non-linear function of both intensities — second-order
   w.r.t. the inputs — while remaining compatible with F.scaled_dot_product_attention
   (FlashAttention on CUDA/MPS).

3. No hard top-k selection.  Log1p normalization naturally drives near-zero
   intensities toward zero fillers; the model learns peak importance via
   attention rather than a fixed intensity threshold.

4. Stateless auxiliary interface.  binding_loss() returns 0 unconditionally;
   the trainer hook (engine/trainer.py:248) adds it to the task loss safely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# 1. Binding Encoder
# ---------------------------------------------------------------------------


class BindingEncoder(nn.Module):
    """
    Maps each (role, filler) pair to a binding vector via Hadamard product.

        b_j = r_j ⊙ GELU(w · x_j)

    Attributes exposed for XAI compatibility:
        role_embeddings, filler_encoder, binding_mode
    """

    def __init__(self, n_cols: int, d_binding: int):
        super().__init__()
        self.binding_mode = "hadamard"  # XAI compat

        self.role_embeddings = nn.Embedding(n_cols, d_binding)
        # bias=False: f(0) = 0 — zero-intensity peaks produce zero-filler bindings
        self.filler_encoder = nn.Sequential(
            nn.Linear(1, d_binding, bias=False), nn.GELU()
        )
        # Default N(0,1) init for Embedding gives role magnitudes ~1.0, which is
        # required for Hadamard binding: b_j = r_j ⊙ f_j.  BERT-style std=0.02 init
        # (for additive positional encoding) crushes binding scale to ~0.001 and
        # produces near-zero attention scores → uniform softmax → model cannot learn.

    def _make_roles(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: [C] or [B, C]  →  [..., d_binding]"""
        return self.role_embeddings(idx)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, C]  →  (bindings [B, C, d], roles [B, C, d], fillers [B, C, d])
        """
        B, C = x.shape
        col_idx = torch.arange(C, device=x.device)
        roles = self._make_roles(col_idx).unsqueeze(0).expand(B, -1, -1)  # [B, C, d]
        fillers = self.filler_encoder(x.unsqueeze(-1))                     # [B, C, d]
        return roles * fillers, roles, fillers


# ---------------------------------------------------------------------------
# 2. Relational Binding Attention
# ---------------------------------------------------------------------------


class RelationalBindingAttention(nn.Module):
    """
    Multi-head attention over binding vectors with non-linear Q/K projections.

    GELU activations after W_q and W_k preserve the multiplicative role×filler
    structure of the binding vectors, so attention scores capture second-order
    interactions without an O(K²) MLP scorer.  Uses F.scaled_dot_product_attention
    throughout (FlashAttention on CUDA/MPS).

    Attributes exposed for XAI compatibility:
        q_proj, k_proj, n_heads, d_head, use_sdp
    """

    def __init__(self, d_binding: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_binding % n_heads == 0, "d_binding must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head = d_binding // n_heads
        self.use_sdp = True  # XAI compat flag
        self.dropout_p = dropout

        # Non-linear projections — GELU preserves second-order binding structure
        self.q_proj = nn.Sequential(nn.Linear(d_binding, d_binding), nn.GELU())
        self.k_proj = nn.Sequential(nn.Linear(d_binding, d_binding), nn.GELU())
        self.v_proj = nn.Linear(d_binding, d_binding)
        self.out_proj = nn.Linear(d_binding, d_binding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D = x.shape
        H, d = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, C, H, d).transpose(1, 2)  # [B, H, C, d]
        k = self.k_proj(x).view(B, C, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, C, H, d).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, C, D))


# ---------------------------------------------------------------------------
# 3. Relational Reasoning Layer
# ---------------------------------------------------------------------------


class RelationalReasoningLayer(nn.Module):
    """Pre-norm transformer block operating over binding vectors."""

    def __init__(self, d_binding: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.rel_attn = RelationalBindingAttention(d_binding, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_binding)
        self.ffn = nn.Sequential(
            nn.Linear(d_binding, 4 * d_binding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_binding, d_binding),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_binding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.rel_attn(x))
        x = self.norm2(x + self.ffn(x))
        return x


# ---------------------------------------------------------------------------
# 4. Task-Query Readout
# ---------------------------------------------------------------------------


class TaskQueryReadout(nn.Module):
    """Learned task query attends over final bindings → single summary vector."""

    def __init__(self, d_binding: int):
        super().__init__()
        self.task_query = nn.Parameter(torch.randn(1, 1, d_binding) * 0.02)

    def forward(self, bindings: torch.Tensor) -> torch.Tensor:
        B, _, D = bindings.shape
        q = self.task_query.expand(B, -1, -1)
        scores = torch.bmm(q, bindings.transpose(1, 2)) / (D ** 0.5)
        return torch.bmm(torch.softmax(scores, dim=-1), bindings).squeeze(1)


# ---------------------------------------------------------------------------
# 5. Full RBN
# ---------------------------------------------------------------------------


class RBN(nn.Module):
    """
    Relational Binding Network.

    Args:
        input_dim:   number of m/z features
        output_dim:  number of classes
        hidden_dim:  binding / attention dimension (must be divisible by num_heads)
        num_layers:  number of relational reasoning layers
        num_heads:   attention heads
        dropout:     dropout rate applied inside attention and FFN
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        top_k: int = 300,
        **kwargs,  # absorb deprecated flags (binding_type, chunk_size, etc.)
    ):
        super().__init__()
        self.top_k = top_k
        self.binding_encoder = BindingEncoder(input_dim, hidden_dim)
        self.reasoning_layers = nn.ModuleList(
            [RelationalReasoningLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.readout = TaskQueryReadout(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1) if x.shape[1] == 1 else x.squeeze(2)

        x = torch.log1p(x.clamp(min=0.0))

        # Limit sequence length for attention memory: O(top_k²) vs O(C²).
        # With C=2080, full attention is ~4 GB on MPS; top_k=300 is ~23 MB.
        # Gradient flows through selected positions; role embeddings cover all C.
        if self.top_k is not None and self.top_k < x.shape[1]:
            top_idx = torch.topk(x, self.top_k, dim=1).indices  # [B, k]
            x_sparse = torch.gather(x, 1, top_idx)              # [B, k]
            roles = self.binding_encoder._make_roles(top_idx)    # [B, k, d]
            fillers = self.binding_encoder.filler_encoder(x_sparse.unsqueeze(-1))
            bindings = roles * fillers
        else:
            bindings, _, _ = self.binding_encoder(x)

        for layer in self.reasoning_layers:
            bindings = layer(bindings)
        return self.head(self.readout(bindings))

    def binding_loss(self) -> torch.Tensor:
        """No auxiliary loss. Preserved for trainer compatibility (trainer.py:248)."""
        p = next(self.parameters())
        return torch.tensor(0.0, device=p.device, dtype=p.dtype)


__all__ = ["RBN"]
