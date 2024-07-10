import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class MassSpecVAEClassifier(nn.Module):
    def __init__(self, input_size=1023, latent_dim=64, num_classes=2, device=None):
        super(MassSpecVAEClassifier, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.device = device
        
        # Encoderencode_and_classify 
        nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Mean and log variance layers
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_size),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std

    def decode(self, z, c):
        zc = torch.cat([z, c], dim=1)
        return self.decoder(zc)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        class_probs = F.softmax(self.classifier(z), dim=1)
        recon_x = self.decode(z, class_probs)
        return recon_x, mu, logvar, class_probs

# Loss function
def vae_classifier_loss(recon_x, x, mu, logvar, class_probs, labels, alpha=1.0, beta=1.0):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    class_probs = class_probs.float()
    labels = labels.float()
    CE = F.cross_entropy(class_probs, labels, reduction='sum')
    return BCE + alpha * KLD + beta * CE