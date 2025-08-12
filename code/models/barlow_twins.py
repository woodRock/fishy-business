import torch
import torch.nn as nn


class BarlowTwinsModel(nn.Module):
    """Barlow Twins model with two identical encoders and projection heads."""

    def __init__(
        self, encoder: nn.Module, encoder_output_dim: int, projection_dim: int = 8192
    ):
        super().__init__()

        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(encoder_output_dim, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        return z1, z2


class BarlowTwinsLoss(nn.Module):
    """Barlow Twins loss function."""

    def __init__(self, lambda_param: float = 5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        # normalize the representations along the batch dimension
        z1_norm = (z1 - z1.mean(dim=0)) / z1.std(dim=0)
        z2_norm = (z2 - z2.mean(dim=0)) / z2.std(dim=0)

        # cross-correlation matrix
        c = torch.matmul(z1_norm.T, z2_norm) / z1_norm.shape[0]

        # loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambda_param * off_diag
        return loss


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


__all__ = ["BarlowTwinsModel", "BarlowTwinsLoss"]
