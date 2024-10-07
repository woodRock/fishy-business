import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import numpy as np
from tqdm import tqdm

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x.unsqueeze(1))  # Add sequence dimension
        x = h_n[-1]
        x = self.fc(x)
        x = self.dropout(x)
        return x

class ContrastiveModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

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
    similarity = nn.functional.cosine_similarity(z1, z2, dim=1)
    loss = y.squeeze() * torch.pow(1 - similarity, 2) + (1 - y.squeeze()) * torch.pow(torch.clamp(similarity - 0.1, min=0.0), 2)
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
    ce = nn.functional.cross_entropy(logits, y.long().squeeze())
    bal = balanced_accuracy_loss(preds, y.squeeze())
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
        
        similarity = nn.functional.cosine_similarity(z1, z2)
        preds = (similarity > 0.5).float()
        
        y = y.squeeze()
        
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

    train_loader, val_loader = preprocess_dataset(dataset="instance-recognition", batch_size=64)

    input_dim = 1023  # Make sure this matches your data's feature dimension
    hidden_dim = 256
    num_layers = 4
    num_classes = 2
    learning_rate = 1E-3

    model = ContrastiveModel(input_dim, hidden_dim, num_layers, num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    num_epochs = 200
    best_val_accuracy = 0
    alpha, beta, gamma, delta = 0.5, 0.0, 0.5, 0.0

    # Early stopping parameters
    patience = 10
    min_delta = 0.001
    patience_counter = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_balanced_accuracy = train_epoch(model, train_loader, optimizer, device, alpha, beta, gamma, delta)
        val_accuracy, val_balanced_accuracy = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Train Balanced Accuracy: {train_balanced_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Balanced Accuracy: {val_balanced_accuracy:.4f}")
        
        # Early stopping check
        if val_balanced_accuracy > best_val_accuracy + min_delta:
            best_val_accuracy = val_balanced_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break
        
        print("------------------------")

    print(f"Best Validation Balanced Accuracy: {best_val_accuracy:.4f}")

if __name__ == "__main__":
    main()