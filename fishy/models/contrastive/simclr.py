# -*- coding: utf-8 -*-
"""
Simple Framework for Contrastive Learning of Visual Representations (SimCLR).
"""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    """
    SimCLR architecture for self-supervised learning.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int = 128,
        projection_dim: int = 128,
        dropout: float = 0.2,
        **kwargs
    ) -> None:
        super(SimCLRModel, self).__init__()
        self.encoder = backbone
        self.projector = ProjectionHead(embedding_dim, embedding_dim, projection_dim, dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        return z1, z2


class SimCLRLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, z1, z2):
        z = torch.cat([z1, z2], dim=0)
        n = z.size(0)
        sim = torch.matmul(z, z.T) / self.temperature
        
        mask = torch.eye(n, device=z.device).bool()
        sim = sim.masked_fill(mask, -1e9)
        
        targets = torch.arange(n, device=z.device)
        targets[:n//2] = targets[:n//2] + n//2
        targets[n//2:] = targets[n//2:] - n//2
        
        return self.criterion(sim, targets)
