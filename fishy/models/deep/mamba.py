# -*- coding: utf-8 -*-
"""
Mamba model for spectral classification.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class MambaBlock(nn.Module):
    """
    Mamba block implementing the inner and outer functions of the Mamba model.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super(MambaBlock, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(
            self.d_inner, self.d_state * 2 + self.d_inner, bias=False
        )
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, self.d_state + 1, dtype=torch.float32)
                .repeat(self.d_inner, 1)
            )
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.norm = nn.LayerNorm(self.d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Mamba block.
        """
        batch, seq_len, d_model = x.shape
        residual = x
        x = self.norm(x)

        z_x = self.in_proj(x)
        z, x = z_x.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)

        x = F.silu(x)

        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split(
            [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))

        # Simplified Selective Scan (In practice, this is implemented using a fused CUDA kernel)
        # Here we use a basic loop or associative scan for CPU/Generic GPU support
        y = self.selective_scan(x, delta, B, C)

        y = y * F.silu(z)
        out = self.out_proj(y)
        out = self.dropout(out)
        return out + residual

    def selective_scan(self, x, delta, B, C):
        """
        A placeholder for the selective scan operation.
        """
        # This is a highly simplified version for demonstration
        A = -torch.exp(self.A_log)
        delta_A = torch.exp(delta.unsqueeze(-1) * A)
        delta_B_x = delta.unsqueeze(-1) * B.unsqueeze(-2) * x.unsqueeze(-1)

        # Simplified recurrence
        batch, seq_len, d_inner, d_state = delta_B_x.shape
        h = torch.zeros(batch, d_inner, d_state, device=x.device)
        ys = []
        for i in range(seq_len):
            h = delta_A[:, i] * h + delta_B_x[:, i]
            y = torch.einsum("bdn,bn->bd", h, C[:, i])
            ys.append(y)
        y = torch.stack(ys, dim=1)
        return y + x * self.D


class Mamba(nn.Module):
    """
    Mamba-based model for spectral data classification.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs
    ) -> None:
        """
        Initializes the Mamba model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 128.
            num_layers (int, optional): Number of Mamba layers. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
            d_state (int, optional): State dimension. Defaults to 16.
            d_conv (int, optional): Convolution kernel size. Defaults to 4.
            expand (int, optional): Expansion factor. Defaults to 2.
        """
        super(Mamba, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Linear(1, hidden_dim)
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=hidden_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_f = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Mamba model.
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch_size, input_dim, 1)
        
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm_f(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc_out(x)


class SiameseMamba(nn.Module):
    """
    Siamese Mamba architecture for instance recognition.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        **kwargs
    ) -> None:
        super(SiameseMamba, self).__init__()
        self.mamba = Mamba(
            input_dim=input_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Union[torch.Tensor, tuple]:
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        combined = torch.cat((out1, out2), dim=1)
        return self.classifier(combined)
