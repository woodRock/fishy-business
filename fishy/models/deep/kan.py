# -*- coding: utf-8 -*-
"""
Kolmogorov-Arnold Network (KAN) model for spectral classification.

Restored implementation using learnable spline-based activations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=nn.SiLU,
        grid_range=[-1, 1],
    ):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_range = grid_range

        # Setup grid: (in_features, grid_size + 2 * spline_order + 1)
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size + 1)
        # Extend grid for splines
        step = (grid_range[1] - grid_range[0]) / grid_size
        for _ in range(spline_order):
            grid = torch.cat([grid[0:1] - step, grid, grid[-1:] + step])

        grid = grid.view(1, -1).repeat(in_features, 1)
        self.register_buffer("grid", grid)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            nn.init.normal_(
                self.spline_weight,
                std=self.scale_noise / math.sqrt(self.in_features) / self.grid_size,
            )

    def b_splines(self, x: torch.Tensor):
        """
        Compute B-spline bases.
        x: (batch, in_features)
        """
        grid = self.grid  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)  # (batch, in_features, 1)

        # 0-order splines
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        # High-order splines
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:-k])
                * bases[:, :, 1:]
            )

        return bases.contiguous()  # (batch, in_features, grid_size + spline_order)

    def forward(self, x: torch.Tensor):
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # bases: (batch, in_features, grid_size + spline_order)
        bases = self.b_splines(x)

        # spline_weight: (out_features, in_features, grid_size + spline_order)
        spline_output = torch.einsum("bik,oik->bo", bases, self.spline_weight)

        return base_output + spline_output


class KAN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        grid_size: int = 5,
        spline_order: int = 3,
        **kwargs,
    ) -> None:
        super(KAN, self).__init__()

        layers = []
        curr_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(
                KANLayer(
                    curr_dim, hidden_dim, grid_size=grid_size, spline_order=spline_order
                )
            )
            curr_dim = hidden_dim
        layers.append(
            KANLayer(
                curr_dim, output_dim, grid_size=grid_size, spline_order=spline_order
            )
        )

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


__all__ = ["KAN", "KANLayer"]
