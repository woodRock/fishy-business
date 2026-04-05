# -*- coding: utf-8 -*-
"""
Shared tensor utilities for model forward passes.
"""

import torch


def ensure_conv_input(x: torch.Tensor) -> torch.Tensor:
    """Ensure a 2D (B, F) tensor becomes (B, 1, F) for Conv1D layers."""
    if x.dim() == 2:
        return x.unsqueeze(1)
    return x


def ensure_seq_input(x: torch.Tensor) -> torch.Tensor:
    """Ensure a 2D (B, F) tensor becomes (B, F, 1) for sequence models (LSTM, RWKV, etc.)."""
    if x.dim() == 2:
        return x.unsqueeze(-1)
    return x
