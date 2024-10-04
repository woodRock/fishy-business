import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from siamese import SiameseNetwork

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

def get_optimal_threshold(model, data_loader, device):
    model.eval()
    distances = []
    labels = []
    with torch.no_grad():
        for X1, X2, y in data_loader:
            X1, X2 = X1.to(device), X2.to(device)
            outputs1, outputs2 = model(X1, X2)
            dist = F.pairwise_distance(outputs1, outputs2)
            distances.extend(dist.cpu().numpy())
            labels.extend(y.numpy())
    
    distances = np.array(distances)
    labels = np.array(labels)
    
    # Find the threshold that maximizes balanced accuracy
    thresholds = np.linspace(distances.min(), distances.max(), num=100)
    best_threshold = thresholds[0]
    best_accuracy = 0
    
    for threshold in thresholds:
        predictions = (distances > threshold).astype(int)
        accuracy = balanced_accuracy_score(labels, predictions)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    return best_threshold

def compute_balanced_accuracy(model, data_loader, device, threshold):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for X1, X2, labels in data_loader:
            X1, X2 = X1.to(device), X2.to(device)
            outputs1, outputs2 = model(X1, X2)
            distances = F.pairwise_distance(outputs1, outputs2)
            predictions = (distances > threshold).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    return balanced_accuracy_score(all_labels, all_predictions)

def train_siamese_network(train_loader, val_loader, input_dim, epochs=50, learning_rate=0.001, margin=1.0):
    model = SiameseNetwork(input_dim)
    criterion = ContrastiveLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_val_acc = 0
    patience = 10
    counter = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (X1, X2, labels) in enumerate(train_loader):
            X1, X2, labels = X1.to(device), X2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs1, outputs2 = model(X1, X2)
            loss = criterion(outputs1, outputs2, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Compute optimal threshold
        threshold = get_optimal_threshold(model, train_loader, device)
        
        train_balanced_acc = compute_balanced_accuracy(model, train_loader, device, threshold)
        val_balanced_acc = compute_balanced_accuracy(model, val_loader, device, threshold)
        
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, '
              f'Threshold: {threshold:.4f}, '
              f'Train Balanced Accuracy: {train_balanced_acc:.4f}, '
              f'Val Balanced Accuracy: {val_balanced_acc:.4f}')
        
        # Early stopping
        if val_balanced_acc > best_val_acc:
            best_val_acc = val_balanced_acc
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    return model, threshold

def predict_similarity(model, X1, X2):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        X1 = torch.FloatTensor(X1).unsqueeze(0).to(device)
        X2 = torch.FloatTensor(X2).unsqueeze(0).to(device)
        output1, output2 = model(X1, X2)
        euclidean_distance = F.pairwise_distance(output1, output2)
    return euclidean_distance.cpu().numpy()