import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ProjectionHead(nn.Module):
    """A non-linear projection head for mapping encoder outputs to a latent space."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ) -> None:
        """Initializes the projection head with a sequence of layers."""
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the projection head."""
        return F.normalize(self.net(x), dim=1)


class SimCLRModel(nn.Module):
    """Combines an encoder with a projection head to form the full SimCLR model."""

    def __init__(self, encoder: nn.Module, config) -> None:
        """Initializes the SimCLR model with an encoder and a projection head."""
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(
            input_dim=config.embedding_dim,
            hidden_dim=config.embedding_dim,
            output_dim=config.projection_dim,
            dropout=config.dropout,
        )

    def forward(
        self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the SimCLR model."""
        z1 = self.encoder(x1)
        h1 = self.projector(z1)
        if x2 is not None:
            z2 = self.encoder(x2)
            h2 = self.projector(z2)
            return h1, h2
        return h1, None


class SimCLRLoss(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy loss (NT-Xent)."""

    def __init__(self, temperature: float):
        """Initializes the NT-Xent loss function with a temperature parameter."""
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        """Forward pass to compute the NT-Xent loss."""
        batch_size = z1.shape[0]
        features = torch.cat([z1, z2], dim=0)
        similarity = F.cosine_similarity(
            features.unsqueeze(1), features.unsqueeze(0), dim=2
        )

        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
        similarity = similarity[~mask].view(2 * batch_size, -1)

        positives = torch.cat(
            [F.cosine_similarity(z1, z2, dim=1), F.cosine_similarity(z2, z1, dim=1)],
            dim=0,
        )
        positives = positives.view(2 * batch_size, 1)

        numerator = torch.exp(positives / self.temperature)
        denominator = torch.sum(
            torch.exp(similarity / self.temperature), dim=1, keepdim=True
        )

        loss = -torch.log(numerator / denominator).mean()
        return loss


__all__ = ["SimCLRModel", "SimCLRLoss"]
