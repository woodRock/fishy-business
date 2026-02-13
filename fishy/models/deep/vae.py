# -*- coding: utf-8 -*-
"""
Variational Autoencoder (VAE) for spectral classification.
"""

import torch
import torch.nn as nn
from typing import Tuple, Union


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model.

    Attributes:
        input_dim (int): Number of input features.
        output_dim (int): Number of output classes.
        hidden_dim (int): Dimension of the latent space.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,  # Used to scale depth
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        """
        Initializes the VAE model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output classes.
            hidden_dim (int, optional): Latent dimension. Defaults to 128.
            num_layers (int, optional): Number of layers. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = hidden_dim

        # Encoder
        encoder_layers = []
        in_features = input_dim
        for i in range(num_layers // 2):
            out_features = max(hidden_dim * 2, in_features // 2)
            encoder_layers.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_features = out_features

        self.encoder_backbone = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(in_features, hidden_dim)
        self.fc_logvar = nn.Linear(in_features, hidden_dim)

        # Decoder (for reconstruction loss if needed)
        decoder_layers = []
        in_features = hidden_dim
        for i in range(num_layers // 2):
            out_features = min(input_dim, in_features * 2)
            decoder_layers.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_features = out_features

        decoder_layers.append(nn.Linear(in_features, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Classifier
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input tensor into mu and logvar.
        """
        h = self.encoder_backbone(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Forward pass. Returns reconstruction and latent params if in training, or just logits.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.classifier(z)

        if self.training:
            recon = self.decoder(z)
            return logits, recon, mu, logvar
        return logits


class SiameseVAE(nn.Module):
    """
    Siamese VAE architecture for instance recognition.
    """

    def __init__(self, vae_backbone: VAE) -> None:
        super(SiameseVAE, self).__init__()
        self.vae = vae_backbone
        self.classifier = nn.Sequential(
            nn.Linear(vae_backbone.latent_dim * 2, vae_backbone.latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(vae_backbone.latent_dim, vae_backbone.output_dim),
        )

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.vae.encode(x)
        return mu

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        combined = torch.cat((out1, out2), dim=1)
        return self.classifier(combined)
