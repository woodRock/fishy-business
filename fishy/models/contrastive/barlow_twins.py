# -*- coding: utf-8 -*-
"""
Barlow Twins model.
"""

import torch
import torch.nn as nn


class BarlowTwinsModel(nn.Module):
    """
    Barlow Twins architecture for self-supervised learning.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int = 128,
        projection_dim: int = 128,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super(BarlowTwinsModel, self).__init__()
        self.encoder = backbone
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        return z1, z2


class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_param: float = 0.005):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, z1, z2):
        batch_size = z1.size(0)
        z1 = (z1 - z1.mean(0)) / z1.std(0)
        z2 = (z2 - z2.mean(0)) / z2.std(0)
        c = torch.mm(z1.T, z2) / batch_size

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.lambda_param * (c.pow(2).sum() - torch.diagonal(c.pow(2)).sum())
        return on_diag + off_diag
