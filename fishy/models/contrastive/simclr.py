# -*- coding: utf-8 -*-
"""
SimCLR with supervised CosineEmbeddingLoss.

Training directly optimises the cosine similarity between pairs using their
ground-truth same/different-class labels, which matches the pairwise cosine
similarity evaluation used throughout this codebase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    """
    Encoder + projection head.  The same projector is used at both train and
    eval time so that cosine similarities are calibrated consistently.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int = 128,
        projection_dim: int = 128,
        **kwargs,
    ):
        super(SimCLRModel, self).__init__()
        self.encoder = backbone
        self.projector = ProjectionHead(embedding_dim, embedding_dim * 2, projection_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        z1 = F.normalize(self.projector(self.encoder(x1)), dim=1)
        z2 = F.normalize(self.projector(self.encoder(x2)), dim=1)
        return z1, z2

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SimCLRLoss(nn.Module):
    """
    Supervised CosineEmbeddingLoss when pair_labels are provided (same/different
    class), falling back to NT-Xent when they are not.

    CosineEmbeddingLoss directly optimises what the evaluation measures:
      y = +1  →  maximise cos(z1, z2)
      y = -1  →  push cos(z1, z2) below `margin`
    """

    def __init__(self, temperature: float = 0.5, margin: float = 0.7):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, pair_labels=None, **kwargs):
        device = z1.device

        if pair_labels is not None:
            # Supervised: pair_labels ∈ {0, 1} → convert to {-1, +1}
            y = pair_labels.float().flatten().to(device)
            y = torch.where(y > 0, torch.ones_like(y), -torch.ones_like(y))
            return F.cosine_embedding_loss(z1, z2, y, margin=self.margin)

        # Unsupervised fallback: NT-Xent with in-batch negatives
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / self.temperature
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask, float("-inf"))
        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device),
        ])
        return F.cross_entropy(sim, targets)
