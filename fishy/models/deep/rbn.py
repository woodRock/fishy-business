# -*- coding: utf-8 -*-
"""
Relational Binding Network (RBN) for spectral classification.

Strict implementation of the formal specification:
- Role-filler binding via Hadamard or Tensor products.
- Second-order relational attention over pairs of bindings.
- Relational memory slots for global context.
- Binding separability regularisation for interpretability.

Reference:
    Relational Binding Network (RBN): Formal Specification & Implementation Brief
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# 1. Sinusoidal Encoding (optional role grounding)
# ---------------------------------------------------------------------------


class SinusoidalEncoding(nn.Module):
    """Encodes continuous m/z positions into fixed sinusoidal role vectors."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: [B, C] or [C]
        device = positions.device
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = positions.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ---------------------------------------------------------------------------
# 2. Binding Encoder
# ---------------------------------------------------------------------------


class BindingEncoder(nn.Module):
    """
    Maps each (role, filler) pair to a binding vector.

    Role  : column identity (m/z position index) -> learned embedding,
            optionally summed with a sinusoidal encoding.
    Filler: cell intensity -> single layer projection + GELU.
    Binding: r_j ⊙ f_ij (Hadamard) or r_j ⊗ f_ij (Outer product).
    """

    def __init__(
        self,
        n_cols: int,
        d_binding: int,
        binding_mode: str = "hadamard",
        d_role: int = 64,
        d_filler: int = 64,
        use_sinusoidal: bool = False,
    ):
        super().__init__()
        self.binding_mode = binding_mode
        self.d_binding = d_binding
        self.d_role = d_role
        self.d_filler = d_filler
        self.use_sinusoidal = use_sinusoidal

        self.role_embeddings = nn.Embedding(n_cols, d_role)
        if use_sinusoidal:
            self.sin_encoding = SinusoidalEncoding(d_role)

        # Filler encoder: scalar -> vector
        # bias=False ensures f(0) = 0, preserving intensity-based sparsity
        self.filler_encoder = nn.Sequential(
            nn.Linear(1, d_filler, bias=False), nn.GELU()
        )

        if binding_mode == "outer_product":
            # Project flattened outer product (d_role * d_filler) -> d_binding
            self.outer_proj = nn.Linear(d_role * d_filler, d_binding, bias=False)
        elif binding_mode == "hadamard":
            # If Hadamard, we need d_role == d_filler == d_binding
            assert (
                d_role == d_filler == d_binding
            ), "Hadamard requires d_role == d_filler == d_binding"

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.role_embeddings.weight, std=0.02)

    def _make_roles(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: [B, C] or [C] — returns [B, C, d_role]."""
        learned = self.role_embeddings(idx)
        if not self.use_sinusoidal:
            return learned
        # idx may be 1-D (full-spectrum path) or 2-D (top-k path)
        positions = idx.float() if idx.dim() == 2 else idx.float().unsqueeze(0)
        sin = self.sin_encoding(positions)
        if idx.dim() == 1:
            sin = sin.squeeze(0)
        return learned + sin

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, C]
        Returns (bindings, roles, fillers)
        """
        B, C = x.shape
        col_idx = torch.arange(C, device=x.device)

        roles = self._make_roles(col_idx).unsqueeze(0).expand(B, -1, -1)  # [B, C, d_role]
        fillers = self.filler_encoder(x.unsqueeze(-1))  # [B, C, d_filler]

        if self.binding_mode == "outer_product":
            # Batch outer product using einsum: [B, C, d_r, d_f]
            outer = torch.einsum("bcd,bce->bcde", roles, fillers)
            bindings = self.outer_proj(outer.flatten(-2))  # [B, C, d_binding]
        else:
            bindings = roles * fillers  # [B, C, d_binding]

        return bindings, roles, fillers


# ---------------------------------------------------------------------------
# 2. Second-Order Relational Attention
# ---------------------------------------------------------------------------


class SecondOrderRelationalAttention(nn.Module):
    """
    Attention over pairs of bindings with memory-efficient chunking:
    ρ_{jj'} = MLP_rel([b_j || b_j' || b_j ⊙ b_j'])
    """

    def __init__(
        self, d_binding: int, n_heads: int, dropout: float = 0.1, chunk_size: int = 64
    ):
        super().__init__()
        self.d_binding = d_binding
        self.n_heads = n_heads
        self.d_head = d_binding // n_heads
        self.chunk_size = chunk_size
        d = self.d_head

        self.q_proj = nn.Linear(d_binding, d_binding)
        self.k_proj = nn.Linear(d_binding, d_binding)
        self.v_proj = nn.Linear(d_binding, d_binding)

        self.mlp_rel = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))

        self.out_proj = nn.Linear(d_binding, d_binding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D = x.shape
        H = self.n_heads
        d = self.d_head

        q = self.q_proj(x).view(B, C, H, d).transpose(1, 2)  # [B, H, C, d]
        k = self.k_proj(x).view(B, C, H, d).transpose(1, 2)  # [B, H, C, d]
        v = self.v_proj(x).view(B, C, H, d).transpose(1, 2)  # [B, H, C, d]

        # Decompose the first linear layer of the MLP to save memory
        # MLP_rel input is [q || k || q*k]
        w1 = self.mlp_rel[0]
        w1_q = w1.weight[:, :d]
        w1_k = w1.weight[:, d : 2 * d]
        w1_qk = w1.weight[:, 2 * d :]
        bias = w1.bias

        # Pre-project q and k
        q_proj = torch.matmul(q, w1_q.t())  # [B, H, C, d]
        k_proj = torch.matmul(k, w1_k.t())  # [B, H, C, d]

        all_outs = []
        for i in range(0, C, self.chunk_size):
            end_i = min(i + self.chunk_size, C)

            # Memory-efficient score calculation:
            # W1 * [q || k || q*k] = W_q*q + W_k*k + W_qk*(q*k)
            # Use broadcasting instead of expansion

            # [B, H, chunk, 1, d] + [B, H, 1, C, d]
            res = q_proj[:, :, i:end_i, :].unsqueeze(3) + k_proj.unsqueeze(2) + bias

            # Interaction term: [B, H, chunk, 1, d] * [B, H, 1, C, d]
            interaction = q[:, :, i:end_i, :].unsqueeze(3) * k.unsqueeze(2)
            res = res + torch.matmul(interaction, w1_qk.t())

            # Remaining MLP layers
            res = self.mlp_rel[2](self.mlp_rel[1](res)).squeeze(-1)  # [B, H, chunk, C]
            scores = res / (d**0.5)

            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            chunk_out = torch.matmul(attn, v)  # [B, H, chunk, d]
            all_outs.append(chunk_out)

        out = torch.cat(all_outs, dim=2)
        out = out.transpose(1, 2).contiguous().view(B, C, D)
        return self.out_proj(out)


class RelationalReasoningLayer(nn.Module):
    """
    L-th layer of relational reasoning.
    Supports gradient checkpointing to save memory during training.
    """

    def __init__(
        self,
        d_binding: int,
        n_heads: int,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.rel_attn = SecondOrderRelationalAttention(d_binding, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_binding)

        self.ffn = nn.Sequential(
            nn.Linear(d_binding, 4 * d_binding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_binding, d_binding),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_binding)

    def _sub_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.rel_attn(x))
        x = self.norm2(x + self.ffn(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._sub_forward, x, use_reentrant=False
            )
        return self._sub_forward(x)


# ---------------------------------------------------------------------------
# 3. Relational Memory
# ---------------------------------------------------------------------------


class RelationalMemory(nn.Module):
    """
    Update memory slots via attention over bindings.
    """

    def __init__(self, n_slots: int, d_binding: int):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(1, n_slots, d_binding) * 0.02)
        self.norm = nn.LayerNorm(d_binding)

    def forward(self, bindings: torch.Tensor) -> torch.Tensor:
        B = bindings.shape[0]
        d = bindings.shape[-1]
        slots = self.slots.expand(B, -1, -1)

        # Slots as queries, bindings as keys/values
        scores = torch.bmm(slots, bindings.transpose(1, 2)) / (d**0.5)
        attn = torch.softmax(scores, dim=-1)
        memory = torch.bmm(attn, bindings)  # [B, S, d]
        return self.norm(memory.mean(dim=1))  # [B, d]


# ---------------------------------------------------------------------------
# 4. Readout
# ---------------------------------------------------------------------------


class TaskQueryReadout(nn.Module):
    """
    Learned task query attends over final bindings.
    """

    def __init__(self, d_binding: int):
        super().__init__()
        self.task_query = nn.Parameter(torch.randn(1, 1, d_binding) * 0.02)

    def forward(self, bindings: torch.Tensor) -> torch.Tensor:
        B = bindings.shape[0]
        d = bindings.shape[-1]
        q = self.task_query.expand(B, -1, -1)
        scores = torch.bmm(q, bindings.transpose(1, 2)) / (d**0.5)
        attn = torch.softmax(scores, dim=-1)
        return torch.bmm(attn, bindings).squeeze(1)


# ---------------------------------------------------------------------------
# 5. Binding Separability Regulariser
# ---------------------------------------------------------------------------


class BindingSeparabilityRegulariser(nn.Module):
    def __init__(self, d_binding: int, d_role: int, d_filler: int):
        super().__init__()
        self.role_decoder = nn.Linear(d_binding, d_role)
        self.filler_decoder = nn.Linear(d_binding, d_filler)

    def forward(self, bindings, roles, fillers):
        loss_r = F.mse_loss(self.role_decoder(bindings), roles.detach())
        loss_f = F.mse_loss(self.filler_decoder(bindings), fillers.detach())
        return loss_r + loss_f


# ---------------------------------------------------------------------------
# 6. Full RBN
# ---------------------------------------------------------------------------


class RBN(nn.Module):
    """
    Relational Binding Network.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        binding_type: str = "hadamard",
        top_k: Optional[int] = 200,
        n_memory_slots: int = 64,
        lambda_binding: float = 0.01,
        chunk_size: int = 64,
        use_checkpointing: bool = False,
        use_sinusoidal: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.lambda_binding = lambda_binding

        d_role = hidden_dim
        d_filler = hidden_dim
        if binding_type == "outer_product":
            d_role = 16  # As per previous impl or reasonable default
            d_filler = 16

        self.binding_encoder = BindingEncoder(
            input_dim, hidden_dim, binding_type, d_role, d_filler, use_sinusoidal
        )

        self.reasoning_layers = nn.ModuleList(
            [
                RelationalReasoningLayer(
                    hidden_dim, num_heads, dropout, use_checkpoint=use_checkpointing
                )
                for _ in range(num_layers)
            ]
        )

        # Pass chunk_size to attention layers
        for layer in self.reasoning_layers:
            layer.rel_attn.chunk_size = chunk_size

        self.memory = (
            RelationalMemory(n_memory_slots, hidden_dim) if n_memory_slots > 0 else None
        )
        self.readout = TaskQueryReadout(hidden_dim)

        self.head = nn.Linear(hidden_dim, output_dim)

        self.separability = BindingSeparabilityRegulariser(hidden_dim, d_role, d_filler)

        # State for auxiliary loss
        self._last_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape normalize
        if x.dim() == 3:
            x = x.squeeze(1) if x.shape[1] == 1 else x.squeeze(2)

        B, C = x.shape
        # log-transform
        x_log = torch.log1p(x.clamp(min=0.0))

        # 1. Sparse selection based on INTESIS (not binding norm)
        # This ensures we pick peaks even if role embeddings aren't learned yet.
        if self.top_k is not None and self.top_k < C:
            _, top_idx = torch.topk(x_log, self.top_k, dim=1)  # [B, k]
            # Gather relevant intensities
            x_sparse = torch.gather(x_log, 1, top_idx)

            # 2. Binding Encoder
            roles = self.binding_encoder._make_roles(top_idx)  # [B, k, d_role]
            fillers = self.binding_encoder.filler_encoder(
                x_sparse.unsqueeze(-1)
            )  # [B, k, d_filler]

            if self.binding_encoder.binding_mode == "outer_product":
                outer = torch.einsum("bcd,bce->bcde", roles, fillers)
                bindings = self.binding_encoder.outer_proj(outer.flatten(-2))
            else:
                bindings = roles * fillers
        else:
            bindings, roles, fillers = self.binding_encoder(x_log)

        # Update separability loss for later retrieval
        if self.training:
            self._last_aux_loss = self.separability(bindings, roles, fillers)

        # 3. Relational Reasoning
        for layer in self.reasoning_layers:
            bindings = layer(bindings)

        # 4. Global aggregation
        h = self.readout(bindings)
        if self.memory is not None:
            mem = self.memory(bindings)
            h = h + mem  # Additive memory context

        # 5. Classification head
        return self.head(h)

    def binding_loss(self) -> torch.Tensor:
        return self._last_aux_loss * self.lambda_binding


__all__ = ["RBN"]
