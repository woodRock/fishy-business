"""BYOL (Bootstrap Your Own Latent) model implementation.

This module defines the BYOL model architecture, which consists of an online network and a target network.
The online network is updated with backpropagation, while the target network is updated using a momentum
update mechanism. The model is designed for self-supervised learning tasks, particularly in computer vision.

References:
1.  Grill, J. B., Strub, F., Altché, F., Tallec, C., Richemond, P., Buchatskaya, E., ... & Valko, M. (2020).
    Bootstrap your own latent-a new approach to self-supervised learning.
    Advances in neural information processing systems, 33, 21271-21284.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class BYOLModel(nn.Module):
    """BYOL model with online and target networks, and a predictor."""

    def __init__(
        self,
        encoder: nn.Module,
        encoder_output_dim: int,
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        m: float = 0.996,
    ):
        super().__init__()

        self.m = m

        # online network
        self.online_encoder = encoder
        self.online_projector = nn.Sequential(
            nn.Linear(encoder_output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

        # target network
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        for param_q, param_k in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # no gradient for target encoder

        for param_q, param_k in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # no gradient for target projector

    @torch.no_grad()
    def _momentum_update_target_network(self):
        """Momentum update of the target network"""
        for param_ol, param_tg in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_tg.data = param_tg.data * self.m + param_ol.data * (1.0 - self.m)

        for param_ol, param_tg in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            param_tg.data = param_tg.data * self.m + param_ol.data * (1.0 - self.m)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # online network
        online_proj_one = self.online_projector(self.online_encoder(x1))
        online_pred_one = self.online_predictor(online_proj_one)

        online_proj_two = self.online_projector(self.online_encoder(x2))
        online_pred_two = self.online_predictor(online_proj_two)

        # target network
        with torch.no_grad():
            self._momentum_update_target_network()
            target_proj_one = self.target_projector(self.target_encoder(x1))
            target_proj_two = self.target_projector(self.target_encoder(x2))

        return online_pred_one, target_proj_two, online_pred_two, target_proj_one


class BYOLLoss(nn.Module):
    """BYOL loss function."""

    def __init__(self):
        super(BYOLLoss, self).__init__()

    def forward(
        self, p1: torch.Tensor, z2: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor
    ):
        loss1 = F.cosine_similarity(p1, z2.detach(), dim=-1).mean()
        loss2 = F.cosine_similarity(p2, z1.detach(), dim=-1).mean()
        loss = 2 - loss1 - loss2  # Maximize cosine similarity, so minimize 2 - cos_sim
        return loss


__all__ = ["BYOLModel", "BYOLLoss"]
