import torch
import torch.nn as nn
import torch.nn.functional as F

class SimSiamModel(nn.Module):
    """SimSiam model with an encoder, projector, and predictor."""

    def __init__(self, encoder: nn.Module, encoder_output_dim: int, projection_dim: int = 2048, hidden_dim: int = 512):
        super().__init__()

        # online network
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(encoder_output_dim, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.BatchNorm1d(projection_dim, affine=False) # No affine for last BN
        )
        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # compute representations
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))

        # compute predictions
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, z2.detach(), p2, z1.detach()


class SimSiamLoss(nn.Module):
    """SimSiam loss function."""
    def __init__(self):
        super(SimSiamLoss, self).__init__()

    def forward(self, p1: torch.Tensor, z2: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor):
        loss = -(F.cosine_similarity(p1, z2, dim=-1).mean() + F.cosine_similarity(p2, z1, dim=-1).mean()) * 0.5
        return loss

__all__ = ["SimSiamModel", "SimSiamLoss"]