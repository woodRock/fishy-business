import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from util import preprocess_dataset

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MassSpecVAEClassifier(nn.Module):
    def __init__(self, input_size=1023, latent_dim=64, num_classes=2):
        super(MassSpecVAEClassifier, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = nn.Sequential(
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
        eps = torch.randn_like(std).to(device)
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

# Instantiate the model and move it to GPU
model = MassSpecVAEClassifier().to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, data_loader, num_epochs, alpha=1.0, beta=1.0):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch, labels in data_loader:
            # Move data to GPU
            inputs = batch.float().to(device)
            labels = labels.long().to(device)
            
            # Forward pass
            recon_batch, mu, logvar, class_probs = model(inputs)
            loss = vae_classifier_loss(recon_batch, inputs, mu, logvar, class_probs, labels, alpha, beta)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print average loss for the epoch
        avg_loss = total_loss / len(data_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

# Function to get encoded representation and class prediction
def encode_and_classify(model, data):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        mu, logvar = model.encode(data)
        z = model.reparameterize(mu, logvar)
        class_probs = F.softmax(model.classifier(z), dim=1)
    return z.cpu(), class_probs.cpu()

# Function to generate new samples of a specific class
def generate(model, num_samples, target_class):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        c = F.one_hot(torch.tensor([target_class] * num_samples), num_classes=model.num_classes).float().to(device)
        samples = model.decode(z, c)
    return samples.cpu()

train_loader, val_loader = preprocess_dataset(
    dataset="species",
    is_data_augmentation=False,
    batch_size=64,
    is_pre_train=False
)

# Example usage:
# Assuming you have a DataLoader called 'train_loader'
train(model, train_loader, num_epochs=100)

# To get encoded representation of a single spectrum:
# spectrum = torch.tensor([your_spectrum_data]).float()
# encoded_spectrum = encode(model, spectrum)

# To generate new samples:
new_samples = generate(model, num_samples=10, target_class=0)
first = new_samples[0]
plt.plot(first)
plt.title("Generated Mass Spectrum")
plt.xlabel("m/z")
plt.ylabel("intensity")
plt.savefig("figures/generated_spectra.png")

first = next(iter(train_loader))[0]
plt.plot(first)
plt.title("Real Mass Spectrum")
plt.xlabel("m/z")
plt.ylabel("intensity")
plt.savefig("figures/real_spectra.png")