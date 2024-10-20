import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import numpy as np
from tqdm import tqdm
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    """ODE function for the Neural ODE block"""
    def __init__(self, channels, dropout=0.5):
        super(ODEFunc, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        
    def forward(self, t, x):
        """
        t: scalar time parameter
        x: feature tensor
        """
        dx = self.conv1(x)
        dx = self.bn1(dx)
        dx = self.relu(dx)
        dx = self.conv2(dx)
        dx = self.bn2(dx)
        dx = self.dropout(dx)
        return dx

class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.register_buffer('integration_times', torch.linspace(0, 1, 2))
        
    def forward(self, x):
        # integration_times will automatically be on the same device as x
        out = odeint(self.odefunc, x, self.integration_times, method='rk4')
        return out[-1]  # Return only the final state

class NeuralODEEncoder(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.3):
        super(NeuralODEEncoder, self).__init__()
        
        # Initial convolution to get to the desired channel size
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # ODE blocks
        self.ode_block1 = ODEBlock(ODEFunc(32, dropout=dropout))
        self.downsample1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.ode_block2 = ODEBlock(ODEFunc(64, dropout=dropout))
        
        # Global pooling and final layers
        self.adaptive_pool = nn.AdaptiveMaxPool1d(4)
        self.flatten = nn.Flatten()
        self.flat_features = 64 * 4
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_features, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.initial_conv(x)
        x = self.ode_block1(x)
        x = self.downsample1(x)
        x = self.ode_block2(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

class ContrastiveNeuralODE(nn.Module):
    def __init__(self, input_channels, d_model, num_classes):
        super().__init__()
        self.encoder = NeuralODEEncoder(input_channels, d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x1, x2=None):
        z1 = self.encoder(x1)
        if x2 is not None:
            z2 = self.encoder(x2)
            return z1, z2
        return z1

    def classify(self, x):
        z = self.encoder(x)
        return self.classifier(z)

def contrastive_loss(z1, z2, y, temperature=0.5):
    similarity = nn.functional.cosine_similarity(z1, z2)
    loss = y * torch.pow(1 - similarity, 2) + (1 - y) * torch.pow(torch.clamp(similarity - 0.1, min=0.0), 2)
    return loss.mean()

def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = (anchor - positive).pow(2).sum(1)
    distance_negative = (anchor - negative).pow(2).sum(1)
    losses = torch.relu(distance_positive - distance_negative + margin)
    return losses.mean()

def balanced_accuracy_loss(preds, labels):
    ba = balanced_accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    return 1 - ba

def combined_loss(z1, z2, y, preds, anchor, positive, negative, logits, alpha=0.3, beta=0.3, gamma=0.2, delta=0.2, temperature=0.5, margin=1.0):
    cl = contrastive_loss(z1, z2, y, temperature)
    tl = triplet_loss(anchor, positive, negative, margin)
    ce = nn.functional.cross_entropy(logits, y.long())
    bal = balanced_accuracy_loss(preds, y)
    return alpha * cl + beta * tl + gamma * ce + delta * bal

def train_epoch(model, dataloader, optimizer, device, alpha=0.3, beta=0.3, gamma=0.2, delta=0.2):
    model.train()
    all_preds = []
    all_labels = []
    total_loss = 0
    for x1, x2, y in tqdm(dataloader, desc="Training"):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        z1, z2 = model(x1, x2)
        logits = model.classify(x1)
        
        # Ensure y is a 1D tensor
        y = y.squeeze()
        
        similarity = nn.functional.cosine_similarity(z1, z2)
        preds = (similarity > 0.5).float()
        
        # Handle cases where all samples are positive or all are negative
        if torch.all(y == 1) or torch.all(y == 0):
            loss = combined_loss(z1, z2, y, preds, z1, z1, z1, logits, alpha, 0, gamma, delta)
        else:
            positive_indices = torch.where(y == 1)[0]
            negative_indices = torch.where(y == 0)[0]
            
            num_triplets = min(len(positive_indices), len(negative_indices), len(z1))
            anchor_indices = torch.randperm(len(z1))[:num_triplets]
            
            anchor = z1[anchor_indices]
            positive = z2[positive_indices[torch.randperm(len(positive_indices))[:num_triplets]]]
            negative = z2[negative_indices[torch.randperm(len(negative_indices))[:num_triplets]]]
            
            loss = combined_loss(z1, z2, y, preds, anchor, positive, negative, logits, alpha, beta, gamma, delta)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    loss = total_loss / len(dataloader)
    return loss, accuracy, balanced_accuracy

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x1, x2, y in tqdm(dataloader, desc="Evaluating"):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            z1, z2 = model(x1, x2)
            similarity = nn.functional.cosine_similarity(z1, z2)
            preds = (similarity > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    return accuracy, balanced_accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from util import preprocess_dataset

    # Load data
    train_loader, val_loader = preprocess_dataset(dataset="instance-recognition", batch_size=64)

    # Model parameters
    input_channels = 1
    d_model = 128
    num_classes = 2

    model = ContrastiveNeuralODE(input_channels, d_model, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    num_epochs = 200
    best_val_accuracy = 0
    # Using the same loss weights as the original implementation
    alpha, beta, gamma, delta = 0.8, 0.0, 0.2, 0.0
    patience = 20
    initial_patience = patience

    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_balanced_accuracy = train_epoch(
            model, train_loader, optimizer, device, alpha, beta, gamma, delta)
        val_accuracy, val_balanced_accuracy = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Train Balanced Accuracy: {train_balanced_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Balanced Accuracy: {val_balanced_accuracy:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_balanced_accuracy > best_val_accuracy:
            best_val_accuracy = val_balanced_accuracy
            torch.save(model.state_dict(), "best_model_neural_ode.pth")
            patience = initial_patience
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping triggered")
        
        print("------------------------")

    print(f"Best Validation Balanced Accuracy: {best_val_accuracy:.4f}")

if __name__ == "__main__":
    main()