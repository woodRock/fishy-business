# -*- coding: utf-8 -*-
"""
Relational Binding Network Plus (RBN++) for spectral classification.

Advanced version of RBN with:
1. Sinusoidal Positional Encodings: Grounding roles in actual m/z values.
2. Gated Relational Memory: GRU-based slot updates for better state retention.
3. Sparse Top-K Relational Attention: $O(C \cdot k)$ reasoning over key peak pairs.
4. Multi-Scale Binding: Encoding both individual peaks and local chemical neighborhoods.
5. Interpretability Report: Automated extraction of driving peak interactions.

Reference:
    Smolensky (1990). Tensor product variable binding.
    Vaswani et al. (2017). Attention is All You Need.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict


# ---------------------------------------------------------------------------
# 1. Advanced Encoders
# ---------------------------------------------------------------------------

class SinusoidalEncoding(nn.Module):
    """Encodes continuous m/z values into high-dimensional role vectors."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: [B, C] or [C]
        device = positions.device
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # [..., C, 1] * [1, half_dim] -> [..., C, half_dim]
        emb = positions.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class MultiScaleBindingEncoder(nn.Module):
    """Binds both raw peaks and local neighborhood features (1D Conv)."""
    def __init__(
        self,
        n_cols: int,
        d_binding: int,
        binding_mode: str = "hadamard",
        d_role: int = 64,
        d_filler: int = 64,
        kernel_size: int = 5
    ):
        super().__init__()
        self.binding_mode = binding_mode
        self.d_binding = d_binding
        self.n_cols = n_cols
        
        # Base Encoders
        self.learned_roles = nn.Embedding(n_cols, d_role)
        self.sin_roles = SinusoidalEncoding(d_role)
        self.filler_encoder = nn.Linear(1, d_filler, bias=False)
        
        # Local Scale: Capture neighborhood features on the FULL spectrum
        self.local_conv = nn.Sequential(
            nn.ReflectionPad1d(kernel_size // 2),
            nn.Conv1d(1, d_filler, kernel_size=kernel_size),
            nn.GELU()
        )
        
        if binding_mode == "outer_product":
            self.outer_proj = nn.Linear(d_role * d_filler, d_binding, bias=False)
        
        self.norm = nn.LayerNorm(d_binding)

    def forward(
        self, 
        x_sparse: torch.Tensor, 
        top_idx: torch.Tensor,
        mz_values: Optional[torch.Tensor] = None,
        x_full: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x_sparse: [B, K] - Selected peak intensities
        top_idx: [B, K] - Original indices of selected peaks
        mz_values: [B, K] - Actual m/z values for selected peaks
        x_full: [B, C] - Full spectrum for local neighborhood conv
        """
        B, K = x_sparse.shape
        device = x_sparse.device
        
        # 1. Role Formation (Sinusoidal + Learned)
        # BUG FIX: Use top_idx to lookup learned roles, not arange(K)
        s_roles = self.sin_roles(mz_values)
        l_roles = self.learned_roles(top_idx) 
        roles = s_roles + l_roles # [B, K, d_role]
        
        # 2. Filler Formation
        # Pointwise intensity
        f_point = self.filler_encoder(x_sparse.unsqueeze(-1)) # [B, K, d_filler]
        
        # Local neighborhood intensity (from full spectrum)
        if x_full is not None:
            f_full = self.local_conv(x_full.unsqueeze(1)).transpose(1, 2) # [B, C, d_filler]
            # Gather neighborhood features for the selected peaks
            idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, f_full.shape[-1])
            f_local = torch.gather(f_full, 1, idx_exp) # [B, K, d_filler]
        else:
            f_local = 0.0
            
        fillers = F.gelu(f_point + f_local)
        
        # 3. Binding
        if self.binding_mode == "outer_product":
            outer = torch.einsum("bkd,bke->bkde", roles, fillers)
            bindings = self.outer_proj(outer.flatten(-2))
        else:
            bindings = roles * fillers
            
        return self.norm(bindings), roles, fillers


# ---------------------------------------------------------------------------
# 2. Sparse Second-Order Attention
# ---------------------------------------------------------------------------

class SparseSecondOrderAttention(nn.Module):
    """Only attends to Top-K relationships per peak to handle large spectra.

    Uses row-wise chunking to avoid materialising the full [B, H, C, C, d]
    interaction tensor, keeping peak memory at O(chunk * C * d) per layer.
    """
    def __init__(self, d_binding: int, n_heads: int, dropout: float = 0.1,
                 sparse_k: int = 128, chunk_size: int = 64):
        super().__init__()
        self.d_binding = d_binding
        self.n_heads = n_heads
        self.d_head = d_binding // n_heads
        self.sparse_k = sparse_k
        self.chunk_size = chunk_size
        d = self.d_head

        self.q_proj = nn.Linear(d_binding, d_binding)
        self.k_proj = nn.Linear(d_binding, d_binding)
        self.v_proj = nn.Linear(d_binding, d_binding)

        self.mlp_rel = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.GELU(),
            nn.Linear(d, 1)
        )
        self.out_proj = nn.Linear(d_binding, d_binding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D = x.shape
        H, d = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, C, H, d).transpose(1, 2)  # [B, H, C, d]
        k = self.k_proj(x).view(B, C, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, C, H, d).transpose(1, 2)

        w1 = self.mlp_rel[0]
        q_proj = torch.matmul(q, w1.weight[:, :d].t())       # [B, H, C, d]
        k_proj = torch.matmul(k, w1.weight[:, d:2*d].t())
        w1_qk = w1.weight[:, 2*d:].t()
        bias = w1.bias

        all_outs = []
        for i in range(0, C, self.chunk_size):
            end_i = min(i + self.chunk_size, C)

            # [B, H, chunk, 1, d] + [B, H, 1, C, d] -> [B, H, chunk, C, d]
            res = q_proj[:, :, i:end_i].unsqueeze(3) + k_proj.unsqueeze(2) + bias
            interaction = q[:, :, i:end_i].unsqueeze(3) * k.unsqueeze(2)
            res = res + torch.matmul(interaction, w1_qk)

            scores = self.mlp_rel[2](self.mlp_rel[1](res)).squeeze(-1)  # [B, H, chunk, C]
            scores = scores / (d ** 0.5)

            # Sparsification: keep only top-k relationships per query position
            if self.sparse_k < C:
                top_vals, _ = torch.topk(scores, self.sparse_k, dim=-1)
                min_val = top_vals[..., -1].unsqueeze(-1)
                scores = torch.where(scores >= min_val, scores, torch.full_like(scores, -1e9))

            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            all_outs.append(torch.matmul(attn, v))  # [B, H, chunk, d]

        out = torch.cat(all_outs, dim=2)             # [B, H, C, d]
        out = out.transpose(1, 2).contiguous().view(B, C, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# 3. Gated Memory
# ---------------------------------------------------------------------------

class GatedRelationalMemory(nn.Module):
    """Uses a GRU-like gate to update memory slots."""
    def __init__(self, n_slots: int, d_binding: int):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(1, n_slots, d_binding) * 0.02)
        self.gru = nn.GRUCell(d_binding, d_binding)
        self.norm = nn.LayerNorm(d_binding)

    def forward(self, bindings: torch.Tensor) -> torch.Tensor:
        B = bindings.shape[0]
        D = bindings.shape[-1]
        slots = self.slots.expand(B, -1, -1)
        
        # Softmax context per slot
        scores = torch.bmm(slots, bindings.transpose(1, 2)) / (D ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, bindings) # [B, S, D]
        
        # Flatten for GRUCell
        S = slots.shape[1]
        slots_flat = slots.reshape(-1, D)
        context_flat = context.reshape(-1, D)
        
        updated = self.gru(context_flat, slots_flat)
        updated = updated.view(B, S, D)
        
        return self.norm(updated.mean(dim=1))


# ---------------------------------------------------------------------------
# 4. RBN++ Model
# ---------------------------------------------------------------------------

class RBNPlus(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        binding_type: str = "outer_product",
        top_k: Optional[int] = 300,
        sparse_k: int = 64,
        n_memory_slots: int = 64,
        lambda_binding: float = 0.01,
        use_checkpointing: bool = False,
        chunk_size: int = 64,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.top_k = top_k
        self.lambda_binding = lambda_binding
        self.use_checkpointing = use_checkpointing
        
        d_inner = 16 if binding_type == "outer_product" else hidden_dim
        
        self.encoder = MultiScaleBindingEncoder(
            input_dim, hidden_dim, binding_type, d_inner, d_inner
        )
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": SparseSecondOrderAttention(hidden_dim, num_heads, dropout, sparse_k, chunk_size),
                "norm1": nn.LayerNorm(hidden_dim),
                "ffn": nn.Sequential(
                    nn.Linear(hidden_dim, 4 * hidden_dim),
                    nn.GELU(),
                    nn.Linear(4 * hidden_dim, hidden_dim)
                ),
                "norm2": nn.LayerNorm(hidden_dim)
            })
            for _ in range(num_layers)
        ])
        
        self.memory = GatedRelationalMemory(n_memory_slots, hidden_dim)
        self.readout = TaskQueryReadoutPlus(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.separability = BindingSeparabilityRegulariserPlus(hidden_dim, d_inner, d_inner)
        
        self._last_aux_loss = torch.tensor(0.0)
        self._last_attn = None

    def forward(self, x: torch.Tensor, mz_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1) if x.shape[1] == 1 else x.squeeze(2)
            
        B, C = x.shape
        x_log = torch.log1p(x.clamp(min=0.0))
        
        # 1. Peak Selection
        if self.top_k and self.top_k < C:
            vals, top_idx = torch.topk(x_log, self.top_k, dim=1)
            x_in = vals
        else:
            top_idx = torch.arange(C, device=x.device).unsqueeze(0).expand(B, -1)
            x_in = x_log

        # 2. Multi-Scale Binding
        # Map mz_values if provided
        if mz_values is not None:
            # Ensure [B, C] shape for gather
            mz_expanded = mz_values
            if mz_expanded.dim() == 1:
                mz_expanded = mz_expanded.unsqueeze(0).expand(B, -1)
            elif mz_expanded.shape[0] != B:
                mz_expanded = mz_expanded.expand(B, -1)
            curr_mz = torch.gather(mz_expanded, 1, top_idx)
        else:
            curr_mz = top_idx.float()
        
        # Pass x_log as x_full to the encoder to capture neighborhood from full spectrum
        bindings, roles, fillers = self.encoder(x_in, top_idx, curr_mz, x_full=x_log)
            
        if self.training:
            self._last_aux_loss = self.separability(bindings, roles, fillers)

        # 3. Reasoning Layers
        for layer in self.layers:
            def sub(b):
                b = layer["norm1"](b + layer["attn"](b))
                b = layer["norm2"](b + layer["ffn"](b))
                return b
            
            if self.use_checkpointing and self.training:
                bindings = torch.utils.checkpoint.checkpoint(sub, bindings, use_reentrant=False)
            else:
                bindings = sub(bindings)
        
        # 4. Readout + Memory
        h = self.readout(bindings)
        mem = self.memory(bindings)
        
        return self.head(h + mem)

    def binding_loss(self) -> torch.Tensor:
        return self._last_aux_loss * self.lambda_binding

    def get_interpretability_report(
        self, x: torch.Tensor, mz_values: Optional[torch.Tensor] = None, top_n: int = 10
    ) -> List[Dict]:
        """Returns the top peak pairs that drive predictions."""
        self.eval()
        with torch.no_grad():
            if x.dim() == 3:
                x = x.squeeze(1) if x.shape[1] == 1 else x.squeeze(2)
            B, C = x.shape
            x_log = torch.log1p(x.clamp(min=0.0))
            
            if self.top_k and self.top_k < C:
                vals, top_idx = torch.topk(x_log, self.top_k, dim=1)
            else:
                top_idx = torch.arange(C, device=x.device).unsqueeze(0).expand(B, -1)
                vals = x_log
            
            if mz_values is None:
                mz_values = torch.arange(C, device=x.device).float().expand(B, -1)
            
            # Ensure [B, C] shape for gather
            mz_expanded = mz_values.to(x.device)
            if mz_expanded.dim() == 1:
                mz_expanded = mz_expanded.unsqueeze(0).expand(B, -1)
            elif mz_expanded.shape[0] != B:
                mz_expanded = mz_expanded.expand(B, -1)
                
            curr_mz = torch.gather(mz_expanded, 1, top_idx)
            # Use fixed encoder call
            bindings, _, _ = self.encoder(vals, top_idx, curr_mz, x_full=x_log)
            
            # Get last layer attention
            # Since we don't store it by default, we'll manually call it
            attn_module = self.layers[-1]["attn"]
            q = attn_module.q_proj(bindings)
            k = attn_module.k_proj(bindings)
            H, d = attn_module.n_heads, attn_module.d_head
            K = bindings.shape[1]
            q = q.view(B, K, H, d).transpose(1, 2)
            k = k.view(B, K, H, d).transpose(1, 2)
            
            w1 = attn_module.mlp_rel[0]
            q_p = torch.matmul(q, w1.weight[:, :d].t())
            k_p = torch.matmul(k, w1.weight[:, d:2*d].t())
            w1_qk = w1.weight[:, 2*d:].t()
            
            res = q_p.unsqueeze(3) + k_p.unsqueeze(2) + w1.bias
            res = res + torch.matmul(q.unsqueeze(3) * k.unsqueeze(2), w1_qk)
            scores = attn_module.mlp_rel[2](attn_module.mlp_rel[1](res)).squeeze(-1)
            
            # Average over batch and heads
            avg_scores = scores.mean(dim=(0, 1)) # [top_k, top_k]
            
            # Find top pairs
            flat = avg_scores.flatten()
            top_vals, indices = torch.topk(flat, top_n)
            
            report = []
            for val, idx in zip(top_vals, indices):
                i, j = idx // K, idx % K
                mz_i = curr_mz[0, i].item()
                mz_j = curr_mz[0, j].item()
                report.append({
                    "mz_a": mz_i,
                    "mz_b": mz_j,
                    "score": val.item()
                })
            return report


class TaskQueryReadoutPlus(nn.Module):
    def __init__(self, d_binding: int):
        super().__init__()
        self.task_query = nn.Parameter(torch.randn(1, 1, d_binding) * 0.02)

    def forward(self, bindings: torch.Tensor) -> torch.Tensor:
        B, C, D = bindings.shape
        q = self.task_query.expand(B, -1, -1)
        scores = torch.bmm(q, bindings.transpose(1, 2)) / (D ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, bindings).squeeze(1)


class BindingSeparabilityRegulariserPlus(nn.Module):
    def __init__(self, d_binding: int, d_role: int, d_filler: int):
        super().__init__()
        self.role_decoder = nn.Linear(d_binding, d_role)
        self.filler_decoder = nn.Linear(d_binding, d_filler)

    def forward(self, bindings, roles, fillers):
        loss_r = F.mse_loss(self.role_decoder(bindings), roles.detach())
        loss_f = F.mse_loss(self.filler_decoder(bindings), fillers.detach())
        return loss_r + loss_f

__all__ = ["RBNPlus"]
