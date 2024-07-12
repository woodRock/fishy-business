import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class VAE(nn.Module):
    def __init__(self, 
        input_size: int = 1023,
        latent_dim: int = 64, 
        num_classes: int = 2, 
        device: Union[torch.device, str] = 'cpu',
        dropout: float = 0.2
    ) -> None:
        """ A variational autoencoder (VAE) with a classifier.

        Args:
            input_size (int): The size of the inlatentput data.
            latent_dim (int): The size of the latent space.
            num_classes (int): The number of classes in the dataset.
            device (Union[str, torch.device]): The device to run the model on.

        """
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.device = device
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
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
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def encode(self, 
                x: torch.Tensor
    ) -> Union[torch.Tensor, torch.Tensor]:
        """ Encode the input data.

        Args: 
            x (torch.Tensor): The input data.

        Returns:
            mu (torch.Tensor), logvar (torch.tesnor): The mean and log variance of the latent distribution.
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, 
                        mu: torch.Tensor, 
                        logvar: torch.Tensor
        ) -> torch.Tensor:
        """ Reparameterization trick to sample from N(mu, var) from N(0, 1).

        Args: 
            mu (torch.Tensor): The mean of the latent distribution.
            logvar (torch.Tensor): The log variance of the latent distribution.

        Returns:
            z (torch.Tensor): The sampled latent representation.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, 
                z: torch.Tensor, 
                c: torch.Tensor
        ) -> torch.Tensor:
        """ Decode the latent representation and class label.

        Args: 
            z (torch.Tensor): The latent representation.
            c (torch.Tensor): The class label.

        Returns: 
            zc (torch.Tensor): The reconstructed input.
        """
        zc = torch.cat([z, c], dim=1)
        return self.decoder(zc)

    def forward(self, 
            x: torch.Tensor
        ) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward pass of the VAE.

        Args:
            x (torch.Tensor): The input data.

        Returns: 
            recon_x (torch.Tensor): The reconstructed input.
            mu (torch.Tensor): The mean of the latent space.
            logvar (torch.Tensor): The log variance of the latent space.
            class_probs (torch.Tensor): The class probabilities.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar).to(self.device)
        class_probs = F.softmax(self.classifier(z), dim=1).to(self.device)
        recon_x = self.decode(z, class_probs)
        return recon_x, mu, logvar, class_probs


def vae_classifier_loss(
        recon_x: torch.Tensor, 
        x: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor, 
        class_probs: torch.Tensor, 
        labels: torch.Tensor, 
        alpha: int = 0.2, 
        beta: int = 0.7,
        gamma: int = 0.1,
    ) -> float:
    """Classification loss for the VAE.

    Args:
        recon_x (torch.Tensor): The reconstructed input.
        x (torch.Tensor): The input data.
        mu (torch.Tensor): The mean of the latent space.
        logvar (torch.Tensor): The log variance of the latent space.
        class_probs (torch.Tensor): The class probabilities.
        labels (torch.Tensor): The true labels.
        alpha (int): The weight of the KLD loss.
        beta (int): The weight of the classification loss.

    Returns:
        loss (float): The total loss.
    """
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    class_probs = class_probs.argmax(1).float()
    labels = labels.argmax(1).float()
    cce = nn.CrossEntropyLoss()
    CCE = cce(class_probs,labels)
    return (alpha * BCE) + (beta * KLD) + (gamma * CCE)