import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

class RWKV(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RWKV, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Linear layers for key, value, and output
        self.key_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)
        self.recurrent_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Initialization
        self.hidden = None

    def forward(self, x):
        # Compute keys and values
        keys = self.key_layer(x)
        values = self.value_layer(x)

        # If hidden state is not initialized, initialize it
        if self.hidden is None:
            self.hidden = torch.zeros(x.size(0), self.hidden_size).to(x.device)

        # Update hidden state with current keys and values
        self.hidden = self.hidden + torch.tanh(keys + self.recurrent_layer(values))

        # Compute output
        output = self.output_layer(self.hidden)

        # Reset the hidden state.
        self.hidden = None
        return output

class ContrastiveModel(nn.Module):
    def __init__(self, input_dim, hidden_size, output_size, num_classes, dropout=0.1):
        super().__init__()
        self.encoder = RWKV(input_dim, hidden_size, output_size)
        self.classifier = nn.Linear(output_size, num_classes)

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
    similarity = F.cosine_similarity(z1, z2)
    loss = y * torch.pow(1 - similarity, 2) + (1 - y) * torch.pow(torch.clamp(similarity - 0.1, min=0.0), 2)
    return loss.mean()

def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = (anchor - positive).pow(2).sum(1)
    distance_negative = (anchor - negative).pow(2).sum(1)
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean()

def balanced_accuracy_loss(preds, labels):
    ba = balanced_accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    return 1 - ba

def combined_loss(z1, z2, y, preds, anchor, positive, negative, logits, alpha=0.3, beta=0.3, gamma=0.2, delta=0.2, temperature=0.5, margin=1.0):
    cl = contrastive_loss(z1, z2, y, temperature)
    tl = triplet_loss(anchor, positive, negative, margin)
    ce = F.cross_entropy(logits, y.long())
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
        
        y = y.squeeze()
        
        similarity = F.cosine_similarity(z1, z2)
        preds = (similarity > 0.5).float()
        
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
            similarity = F.cosine_similarity(z1, z2)
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

    # Load your data
    train_loader, val_loader = preprocess_dataset(dataset="instance-recognition", batch_size=64)

    # Model parameters
    input_dim = next(iter(train_loader))[0].shape[1]  # Get input dimension from data
    hidden_size = 128
    output_size = 64
    num_classes = 2  # Adjust based on your task
    dropout = 0.1

    model = ContrastiveModel(input_dim, hidden_size, output_size, num_classes, dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)

    num_epochs = 200
    best_val_accuracy = 0
    alpha, beta, gamma, delta = 1.0, 0.0, 0.0, 0.0  # Weights for different loss components
    patience = 20
    initial_patience = patience

    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_balanced_accuracy = train_epoch(model, train_loader, optimizer, device, alpha, beta, gamma, delta)
        val_accuracy, val_balanced_accuracy = evaluate(model, val_loader, device)
        
        scheduler.step(val_balanced_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Train Balanced Accuracy: {train_balanced_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Balanced Accuracy: {val_balanced_accuracy:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_balanced_accuracy > best_val_accuracy:
            best_val_accuracy = val_balanced_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            patience = initial_patience
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping triggered")
                # DEBUG - turn off early stopping
                # break
        
        print("------------------------")

    print(f"Best Validation Balanced Accuracy: {best_val_accuracy:.4f}")

if __name__ == "__main__":
    main()