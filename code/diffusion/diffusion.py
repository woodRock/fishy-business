import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += self.shortcut(residual)
        x = self.relu(x)
        return x

class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down1 = ResidualBlock(in_channels, 64)
        self.down2 = ResidualBlock(64, 128)
        self.down3 = ResidualBlock(128, 256)
        self.down4 = ResidualBlock(256, 512)
        
        self.up4 = ResidualBlock(512, 256)
        self.up3 = ResidualBlock(512, 128)
        self.up2 = ResidualBlock(256, 64)
        self.up1 = ResidualBlock(128, 64)
        
        self.final = nn.Conv1d(64, out_channels, 1)
        
        self.maxpool = nn.MaxPool1d(2)

    def forward(self, x, t):
        t = t.unsqueeze(-1).repeat(1, x.shape[-1]).unsqueeze(1)
        x = torch.cat([x, t], dim=1)
        
        x1 = self.down1(x)
        x2 = self.down2(self.maxpool(x1))
        x3 = self.down3(self.maxpool(x2))
        x4 = self.down4(self.maxpool(x3))
        
        x = F.interpolate(x4, size=x3.shape[2], mode='linear', align_corners=False)
        x = self.up4(x)
        x = torch.cat([x, x3], dim=1)
        
        x = F.interpolate(x, size=x2.shape[2], mode='linear', align_corners=False)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        
        x = F.interpolate(x, size=x1.shape[2], mode='linear', align_corners=False)
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        
        x = self.up1(x)
        
        return self.final(x)

class DiffusionModel(nn.Module):
    def __init__(self, feature_size=1023, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        super().__init__()
        self.feature_size = feature_size
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.model = UNet1D(in_channels=2, out_channels=1).to(device)

    def forward(self, x, t):
        return self.model(x, t)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, n):
        self.model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.feature_size)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = self.model(x, t.float() / self.noise_steps)
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        self.model.train()
        return x

def train_diffusion_model(model, train_loader, val_loader, num_epochs=100, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    best_val_loss = float('inf')

    batch_size = 64
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(model.device), y.to(model.device)
            t = model.sample_timesteps(x.shape[0]).to(model.device)
            x_t, noise = model.noise_images(x, t)
            predicted_noise = model(x_t, t.float() / model.noise_steps)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(model.device), y.to(model.device)
                t = model.sample_timesteps(x.shape[0]).to(model.device)
                x_t, noise = model.noise_images(x, t)
                predicted_noise = model(x_t, t.float() / model.noise_steps)
                loss = mse(noise, predicted_noise)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_diffusion_model.pth')
    
    model.load_state_dict(torch.load('best_diffusion_model.pth'))
    return model

def classify(model, x, num_samples=100, threshold=0.5):
    model.eval()
    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            sample = model.sample(x.shape[0])
            samples.append(sample)
    samples = torch.cat(samples, dim=0)
    mean_sample = samples.mean(dim=0)
    return (mean_sample > threshold).float()

# Example usage:
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load your data here
    # X should be of shape (num_samples, 1023)
    # y should be of shape (num_samples,) and contain binary labels
    X = np.random.randn(1000, 1023)  # Replace with your data
    y = np.random.randint(0, 2, 1000)  # Replace with your labels
    
    # Preprocess the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train).unsqueeze(1), torch.FloatTensor(y_train).unsqueeze(1))
    test_dataset = TensorDataset(torch.FloatTensor(X_test).unsqueeze(1), torch.FloatTensor(y_test).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize and train the model
    diffusion = DiffusionModel(feature_size=1023, device=device)
    trained_model = train_diffusion_model(diffusion, train_loader, test_loader, num_epochs=100)
    
    # Evaluate the model
    correct = 0
    total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        predictions = classify(trained_model, x)
        correct += (predictions.squeeze() == y).sum().item()
        total += y.size(0)
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")