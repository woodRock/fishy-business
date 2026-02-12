# -*- coding: utf-8 -*-
"""
Simple Siamese (SimSiam) model.
"""

import torch
import torch.nn as nn


class SimSiamModel(nn.Module):
    """
    SimSiam architecture for self-supervised learning.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int = 128,
        projection_dim: int = 128,
        dropout: float = 0.2,
        **kwargs
    ) -> None:
        super(SimSiamModel, self).__init__()
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
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 4),
            nn.BatchNorm1d(projection_dim // 4),
            nn.ReLU(),
            nn.Linear(projection_dim // 4, projection_dim),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1, z2


class SimSiamLoss(nn.Module):
    def __init__(self):
        super(SimSiamLoss, self).__init__()

    def forward(self, p1, p2, z1, z2):
        def D(p, z):
            z = z.detach()
            p = nn.functional.normalize(p, dim=1)
            z = nn.functional.normalize(z, dim=1)
            return -(p * z).sum(dim=1).mean()

        return 0.5 * D(p1, z2) + 0.5 * D(p2, z1)
