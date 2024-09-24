import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from util import preprocess_data


class MixtureDensityNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_gaussians):
        super(MixtureDensityNetwork, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.pi_layer = nn.Linear(hidden_dim, n_gaussians)
        self.mu_layer = nn.Linear(hidden_dim, n_gaussians)
        self.sigma_layer = nn.Linear(hidden_dim, n_gaussians)
        
    def forward(self, x):
        h = torch.tanh(self.hidden_layer(x))
        pi = torch.softmax(self.pi_layer(h), dim=1)
        mu = self.mu_layer(h)
        sigma = torch.exp(self.sigma_layer(h))
        return pi, mu, sigma

def mdn_loss(y, pi, mu, sigma):
    normal = torch.distributions.Normal(mu, sigma)
    log_prob = normal.log_prob(y.expand_as(mu))
    weighted_logprob = log_prob + torch.log(pi + 1e-8)  # Add small epsilon to avoid log(0)
    return -torch.logsumexp(weighted_logprob, dim=1).mean()

def train_mdn(model, X, y, epochs=1000, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        pi, mu, sigma = model(X)
        loss = mdn_loss(y, pi, mu, sigma)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    return model, losses

def sample_from_mdn(model, X, n_samples=1):
    pi, mu, sigma = model(X)
    mixture_idx = torch.multinomial(pi, n_samples, replacement=True)
    selected_mu = mu.gather(1, mixture_idx)
    selected_sigma = sigma.gather(1, mixture_idx)
    epsilon = torch.randn_like(selected_mu)
    return selected_mu + selected_sigma * epsilon

# Generate synthetic high-dimensional data
np.random.seed(42)
n_samples = 1000


(X_train, y_train) , (X_test, y_test) = preprocess_data()

print(f"X: {X_train.shape}")
print(f"y: {y_train.shape}")

# Apply PCA
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_train)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)

# Create and train the model
model = MixtureDensityNetwork(input_dim=2046, hidden_dim=100, n_gaussians=5)
trained_model, losses = train_mdn(model, X_train, y_train, epochs=10_000, lr=0.0001)

# Generate predictions
# X_test_pca = pca.transform(X_test)
X_test = torch.FloatTensor(X_test)
y_samples = sample_from_mdn(trained_model, X_test, n_samples=100)

# Plotting
plt.figure(figsize=(15, 5))

# Plot 1: Data and predictions
plt.subplot(1, 3, 1)
plt.scatter(X_train[:, 0], y_train, alpha=0.3, label='Data')
plt.scatter(X_test[:, 0], y_samples.mean(dim=1).detach().numpy(), c='r', alpha=0.5, label='Mean prediction')
plt.legend()
plt.title('MDN Predictions after PCA')
plt.xlabel('First PCA Component')
plt.ylabel('Y')

# Plot 2: PCA components
plt.subplot(1, 3, 2)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
plt.colorbar(label='Y value')
plt.title('Data in PCA Space')
plt.xlabel('First PCA Component')
plt.ylabel('Second PCA Component')

# Plot 3: Loss history
plt.subplot(1, 3, 3)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Negative Log-Likelihood')

plt.tight_layout()
plt.savefig("figures/plot.png")
plt.show()

# Print explained variance ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)