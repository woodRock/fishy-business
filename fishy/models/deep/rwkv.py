# -*- coding: utf-8 -*-
"""
Receptance Weighted Key-Value (RWKV) model for spectral classification.

RWKV combines the training efficiency of Transformers with the inference 
efficiency of RNNs. It uses a unique linear attention mechanism.

References:
1. Peng, B., et al. (2023). RWKV: Reinventing RNNs for Transformers. arXiv:2305.13048.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeMixing(nn.Module):
    """
    RWKV Time Mixing block.
    """
    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = n_embd
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)
        
        # Time decay parameters
        self.time_decay = nn.Parameter(torch.ones(n_embd))
        self.time_first = nn.Parameter(torch.ones(n_embd))

    def forward(self, x: torch.Tensor):
        # x shape: (batch, seq_len, n_embd)
        B, T, C = x.size()
        
        r = self.receptance(x)
        k = self.key(x)
        v = self.value(x)
        
        # Simplified linear attention for 1D sequences
        # In a full RWKV this would be a cumulative sum with decay
        # For spectral data, we treat the peaks as a sequence
        
        w = torch.exp(-torch.exp(self.time_decay))
        
        # Recurrence implementation
        out = torch.zeros_like(v)
        a = torch.zeros(B, C, device=x.device)
        b = torch.zeros(B, C, device=x.device)
        
        for t in range(T):
            # Current time step
            kt = k[:, t]
            vt = v[:, t]
            
            # WKV update (simplified version)
            ww = torch.exp(self.time_first) * kt
            out[:, t] = torch.sigmoid(r[:, t]) * (a + torch.exp(ww) * vt) / (b + torch.exp(ww) + 1e-6)
            
            # Update state for next step
            a = w * a + torch.exp(kt) * vt
            b = w * b + torch.exp(kt)
            
        return self.output(out)


class ChannelMixing(nn.Module):
    """
    RWKV Channel Mixing block.
    """
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        r = torch.sigmoid(self.receptance(x))
        k = torch.square(torch.relu(self.key(x))) # Square ReLU is common in RWKV
        kv = self.value(k)
        return r * kv


class RWKVBlock(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.tm = TimeMixing(n_embd)
        self.cm = ChannelMixing(n_embd, dropout=dropout)

    def forward(self, x: torch.Tensor):
        x = x + self.tm(self.ln1(x))
        x = x + self.cm(self.ln2(x))
        return x


class RWKV(nn.Module):
    """
    RWKV model for spectral classification.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        **kwargs
    ) -> None:
        super(RWKV, self).__init__()
        
        self.embedding = nn.Linear(1, hidden_dim)
        self.blocks = nn.ModuleList([
            RWKVBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input: (batch, input_dim) -> (batch, input_dim, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        x = self.embedding(x) # (batch, input_dim, hidden_dim)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_out(x)
        x = x.mean(dim=1) # Pooling over sequence length
        return self.head(x)


__all__ = ["RWKV", "RWKVBlock"]
