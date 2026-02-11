"""A variational autoencoder (VAE) with a classifier.

This model combines a VAE for unsupervised learning with a classifier for supervised tasks.
It includes an encoder, decoder, and a classifier that predicts class probabilities based on the latent representation
of the input data. The model is designed to handle high-dimensional input data and can be used for tasks such as anomaly detection or semi-supervised learning.


References:

1. Kingma, D. P., & Welling, M. (2013).
    Auto-encoding variational bayes.
    arXiv preprint arXiv:1312.6114.
2. Srivastava, N., Hinton, G., Krizhevsky, A.,
    Sutskever, I., & Salakhutdinov, R. (2014).
    Dropout: a simple way to prevent neural networks from overfitting.
    The journal of machine learning research, 15(1), 1929-1958.
3. Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever,
    I., & Salakhutdinov, R. R. (2012).
    Improving neural networks by preventing co-adaptation of feature detectors.
    arXiv preprint arXiv:1207.0580.
4. Fukushima, K. (1969).
    Visual feature extraction by a multilayered network of analog threshold elements.
    IEEE Transactions on Systems Science and Cybernetics, 5(4), 322-333.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple


class VAE(nn.Module):
    """
    Variational Autoencoder with an integrated classifier head.

    Attributes:
        latent_dim (int): Size of the latent bottleneck.
        num_classes (int): Number of classification targets.
        encoder (nn.Sequential): Feature extraction network.
        fc_mu (nn.Linear): Latent mean projection.
        fc_logvar (nn.Linear): Latent log-variance projection.
        decoder (nn.Sequential): Reconstruction network.
        classifier (nn.Sequential): Classification head.
    """

    def __init__(
        self,
        input_size: int = 1023,
        latent_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """
        Initializes the VAE model.

        Args:
            input_size (int, optional): Size of the input features. Defaults to 1023.
            latent_dim (int, optional): Size of the latent space. Defaults to 64.
            num_classes (int, optional): Number of target classes. Defaults to 2.
            dropout (float, optional): Dropout probability. Defaults to 0.2.
        """
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            # ReLU activation (Fukushima 1969)
            nn.ReLU(),
            # Dropout layer (Srivastava 2014, Hinton 2012)
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        # Mean and log variance layers
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, input_size),
            nn.Sigmoid(),
            nn.Dropout(p=dropout),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input data into latent distribution parameters.

        Args:
            x (torch.Tensor): The input feature tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (mu, logvar) Mean and log-variance tensors.
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Applies the reparameterization trick to sample from the latent distribution.

        Args:
            mu (torch.Tensor): Latent mean.
            logvar (torch.Tensor): Latent log-variance.

        Returns:
            torch.Tensor: The sampled latent vector 'z'.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent representation conditioned on a class label.

        Args:
            z (torch.Tensor): The sampled latent vector.
            c (torch.Tensor): One-hot encoded class label or probabilities.

        Returns:
            torch.Tensor: The reconstructed input tensor.
        """
        zc = torch.cat([z, c], dim=1)
        return self.decoder(zc)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VAE.

        Args:
            x (torch.Tensor): The input feature tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - recon_x: Reconstructed input.
                - mu: Latent mean.
                - logvar: Latent log-variance.
                - class_probs: Predicted class probabilities.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        class_probs = F.softmax(self.classifier(z), dim=1)
        recon_x = self.decode(z, class_probs)
        return recon_x, mu, logvar, class_probs


def vae_classifier_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    class_probs: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
    beta: float = 0.7,
    gamma: float = 0.1,
) -> torch.Tensor:
    """
    Computes the multi-task loss for the VAE classifier.

    Combines Reconstruction loss (BCE), KL Divergence (KLD), and Classification loss (CCE).

    Args:
        recon_x (torch.Tensor): Reconstructed input.
        x (torch.Tensor): Original input.
        mu (torch.Tensor): Latent mean.
        logvar (torch.Tensor): Latent log-variance.
        class_probs (torch.Tensor): Predicted class probabilities.
        labels (torch.Tensor): True class labels (one-hot).
        alpha (float, optional): Weight for reconstruction loss. Defaults to 0.2.
        beta (float, optional): Weight for KLD loss. Defaults to 0.7.
        gamma (float, optional): Weight for classification loss. Defaults to 0.1.

    Returns:
        torch.Tensor: The total weighted loss.
    """
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Simple cross entropy handle
    if labels.dim() > 1 and labels.shape[1] > 1:
        target = labels.argmax(dim=1)
    else:
        target = labels.squeeze().long()

    CCE = F.cross_entropy(class_probs, target)
    return (alpha * BCE) + (beta * KLD) + (gamma * CCE)


class SiameseVAE(nn.Module):
    """
    A Siamese network using a VAE as the backbone.

    Processes pairs of inputs and predicts similarity based on latent distance.

    Args:
        vae_model (VAE): An instance of the VAE backbone.
    """

    def __init__(self, vae_model: VAE) -> None:
        super(SiameseVAE, self).__init__()
        self.vae = vae_model
        self.fc = nn.Linear(1, 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SiameseVAE.

        Args:
            x1 (torch.Tensor): First input tensor.
            x2 (torch.Tensor): Second input tensor.

        Returns:
            torch.Tensor: Similarity score (distance-based).
        """
        mu1, _ = self.vae.encode(x1)
        mu2, _ = self.vae.encode(x2)
        distance = F.pairwise_distance(mu1, mu2)
        output = self.fc(distance.unsqueeze(-1))
        return output
