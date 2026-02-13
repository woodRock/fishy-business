# -*- coding: utf-8 -*-
"""
WaveNet model for spectral classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        **kwargs,
    ) -> None:
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.padding != 0:
            x = x[:, :, : -self.padding]
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.2
    ) -> None:
        super(ResidualBlock, self).__init__()
        self.causal_conv = CausalConv1d(
            channels, channels, kernel_size, dilation=dilation
        )
        self.gate_conv = CausalConv1d(
            channels, channels, kernel_size, dilation=dilation
        )
        self.dropout = nn.Dropout(dropout)
        self.res_conv = nn.Conv1d(channels, channels, 1)
        self.skip_conv = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        tanh_out = torch.tanh(self.causal_conv(x))
        sigm_out = torch.sigmoid(self.gate_conv(x))
        x = tanh_out * sigm_out
        x = self.dropout(x)
        res_out = self.res_conv(x)
        skip_out = self.skip_conv(x)
        return res_out + residual, skip_out


class WaveNet(nn.Module):
    """
    WaveNet-based model for spectral data classification.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        kernel_size: int = 3,
        **kwargs,
    ) -> None:
        """
        Initializes the WaveNet model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes.
            hidden_dim (int, optional): Hidden dimension. Defaults to 128.
            num_layers (int, optional): Number of residual blocks. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
            kernel_size (int, optional): Kernel size. Defaults to 3.
        """
        super(WaveNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.initial_conv = nn.Conv1d(1, hidden_dim, 1)
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, kernel_size, dilation=2**i, dropout=dropout)
                for i in range(num_layers)
            ]
        )

        self.fc_layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim * input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.initial_conv(x)
        skip_connections = []
        for block in self.res_blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        out = sum(skip_connections)
        return self.fc_layers(out)
