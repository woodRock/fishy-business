# -*- coding: utf-8 -*-
"""
Bootstrap Your Own Latent (BYOL) model.
"""

import torch
import torch.nn as nn
import copy


class BYOLModel(nn.Module):
    """
    BYOL architecture for self-supervised learning.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int = 128,
        projection_dim: int = 128,
        dropout: float = 0.2,
        **kwargs
    ) -> None:
        super(BYOLModel, self).__init__()
        self.online_encoder = nn.Sequential(
            backbone,
            nn.Linear(embedding_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        p1 = self.predictor(self.online_encoder(x1))
        p2 = self.predictor(self.online_encoder(x2))
        with torch.no_grad():
            z1 = self.target_encoder(x1)
            z2 = self.target_encoder(x2)
        return p1, p2, z1, z2


class BYOLLoss(nn.Module):
    def __init__(self):
        super(BYOLLoss, self).__init__()

    def forward(self, p1, p2, z1, z2):
        loss1 = 2 - 2 * (torch.cosine_similarity(p1, z2.detach(), dim=-1)).mean()
        loss2 = 2 - 2 * (torch.cosine_similarity(p2, z1.detach(), dim=-1)).mean()
        return (loss1 + loss2) / 2
