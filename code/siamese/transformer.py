import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math

class NumericalTransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads=4, dim_feedforward=1024, num_layers=3, dropout=0.1):
        super(NumericalTransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, 256)
        self.pos_encoder = PositionalEncoding(256, dropout)
        self.layer_norm = nn.LayerNorm(256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=num_heads, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(256, 128)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.layer_norm(x)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class ContrastiveLearningModel(nn.Module):
    def __init__(self, input_dim, projection_dim=64):
        super(ContrastiveLearningModel, self).__init__()
        self.encoder = NumericalTransformerEncoder(input_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z


def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature
    sim_i_j = torch.diag(sim, N)
    sim_j_i = torch.diag(sim, -N)
    
    positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(2*N, 1)
    mask = torch.eye(N, dtype=bool, device=z1.device)
    mask = mask.repeat(2, 2)
    negative_samples = sim[~mask].reshape(2*N, -1)
    
    labels = torch.zeros(2*N, device=z1.device).long()
    logits = torch.cat([positive_samples, negative_samples], dim=1)
    loss = F.cross_entropy(logits, labels)
    return loss

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in data_loader:
        x1, x2, _ = batch  # Ignore labels for contrastive learning
        x1, x2 = x1.to(device), x2.to(device)
        
        optimizer.zero_grad()
        _, z1 = model(x1)
        _, z2 = model(x2)
        loss = nt_xent_loss(z1, z2)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            x1, _, labels = batch  # Use only the first input tensor
            x1 = x1.to(device)
            h, _ = model(x1)
            all_embeddings.append(h.cpu())
            all_labels.append(labels.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_embeddings, all_labels

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = 1023  # Adjust this based on your data
model = ContrastiveLearningModel(input_dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

from util import preprocess_dataset

train_loader, val_loader = preprocess_dataset(dataset="instance-recognition")

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

    if (epoch + 1) % 5 == 0:
        embeddings, labels = evaluate(model, val_loader, device)
        print(f"Validation embeddings shape: {embeddings.shape}")
        print(f"Validation labels shape: {labels.shape}")

print("Training completed!")

# After training, you can use the encoder part of the model to get embeddings for your data
def get_embeddings(model, data_loader, device):
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for batch in data_loader:
            x, _, _ = batch  # Use only the first input tensor
            x = x.to(device)
            h, _ = model(x)
            all_embeddings.append(h.cpu())
    
    return torch.cat(all_embeddings, dim=0)

# Example of how to get embeddings for your data
final_embeddings = get_embeddings(model, val_loader, device)
print(f"Final embeddings shape: {final_embeddings.shape}")