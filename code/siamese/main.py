from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Type
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score

from transformer import Transformer
from lstm import LSTM
from cnn import CNN
from util import prepare_dataset, DataConfig

@dataclass
class SimCLRConfig:
    """Configuration for SimCLR model"""
    temperature: float = 0.5
    projection_dim: int = 128
    embedding_dim: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    batch_size: int = 32
    num_epochs: int = 100
    input_dim: int = 2080
    num_heads: int = 4
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.1

class ProjectionHead(nn.Module):
    """Non-linear projection head for SimCLR"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)

class SimCLRModel(nn.Module):
    """SimCLR model with flexible encoder and projection head"""
    def __init__(
        self, 
        encoder: nn.Module,
        config: SimCLRConfig
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(
            input_dim=config.embedding_dim,
            hidden_dim=config.embedding_dim,
            output_dim=config.projection_dim
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.projection_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.embedding_dim, 2)
        )
    
    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Reshape inputs for transformer: [batch_size, features] -> [batch_size, 1, features]
        if len(x1.shape) == 2:
            x1 = x1.unsqueeze(1)  # [B, F] -> [B, 1, F]
        if x2 is not None and len(x2.shape) == 2:
            x2 = x2.unsqueeze(1)  # [B, F] -> [B, 1, F]   

        # Get representations
        z1 = self.encoder(x1)
        h1 = self.projector(z1)
        
        if x2 is not None:
            z2 = self.encoder(x2)
            h2 = self.projector(z2)
            return h1, h2
        
        return h1, None
    
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        z = self.encoder(x)
        h = self.projector(z)
        return self.classifier(h)

class SimCLRLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.shape[0]
        features = torch.cat([z1, z2], dim=0)  # 2N x D
        
        # Compute similarity matrix
        sim = torch.matmul(features, features.T)  # 2N x 2N
        sim = sim / self.temperature
        
        # Create mask for positive pairs
        pos_mask = torch.zeros((2 * batch_size, 2 * batch_size), device=z1.device)
        # Mark positive pairs (i, i+N) and (i+N, i)
        pos_mask[torch.arange(batch_size), torch.arange(batch_size, 2*batch_size)] = 1
        pos_mask[torch.arange(batch_size, 2*batch_size), torch.arange(batch_size)] = 1
        
        # Remove diagonal (self-similarity)
        mask_no_diag = ~torch.eye(2 * batch_size, dtype=bool, device=z1.device)
        
        # Get positive similarities
        sim_pos = sim[pos_mask.bool()].view(2 * batch_size, 1)
        
        # Get negative similarities (excluding self-similarity)
        sim_neg = sim[mask_no_diag].view(2 * batch_size, -1)
        
        # Concatenate positive and negative similarities
        logits = torch.cat([sim_pos, sim_neg], dim=1)
        
        # Labels: positive pair is the first element (index 0)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z1.device)
        
        # Compute cross entropy loss
        return F.cross_entropy(logits, labels)
    
class SimCLRTrainer:
    def __init__(
        self,
        model: SimCLRModel,
        config: SimCLRConfig,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device
        
        self.optimizer = optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': config.learning_rate},
            {'params': model.projector.parameters(), 'lr': config.learning_rate},
            {'params': model.classifier.parameters(), 'lr': config.learning_rate * 10}
        ], weight_decay=config.weight_decay)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        self.contrastive_loss = SimCLRLoss(temperature=config.temperature)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def process_labels(self, one_hot_labels: torch.Tensor) -> torch.Tensor:
        """Convert one-hot encoded labels to class indices."""
        return torch.argmax(one_hot_labels, dim=1)
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (x1, x2, labels) in enumerate(train_loader):
            # Move to device and ensure float32
            x1 = x1.float().to(self.device)
            x2 = x2.float().to(self.device)
            labels = labels.float().to(self.device)
            
            # Convert one-hot labels to indices
            label_indices = torch.argmax(labels, dim=1)
            
            # Forward pass and get projections
            z1, z2 = self.model(x1, x2)
            
            # Compute contrastive loss
            contrastive_loss = self.contrastive_loss(z1, z2)
            
            # Get classifier predictions (if needed)
            logits = self.model.classify(x1)
            classification_loss = self.ce_loss(logits, label_indices)
            
            # Combine losses (you can adjust the weights)
            loss = contrastive_loss + 0.1 * classification_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Compute accuracy
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(label_indices).sum().item()
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Batch [{batch_idx}/{len(train_loader)}], '
                    f'Loss: {loss.item():.4f}, '
                    f'Contrastive: {contrastive_loss.item():.4f}, '
                    f'Classification: {classification_loss.item():.4f}')
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
    
        return avg_loss, accuracy
    

    def compute_accuracy(self, embeddings_1, embeddings_2, labels):
        """
        Compute balanced classification accuracy based on embedding similarity.
        Args:
            embeddings_1: First set of embeddings (n_samples, embedding_dim)
            embeddings_2: Second set of embeddings (n_samples, embedding_dim)
            labels: One-hot encoded labels (n_samples, n_classes)
        Returns:
            float: Balanced classification accuracy
        """
        # Convert one-hot labels to class indices
        labels = torch.argmax(labels, dim=1).cpu().numpy()
        
        # Compute cosine similarity between pairs
        similarity = F.cosine_similarity(embeddings_1, embeddings_2)
        
        # Convert similarities to binary predictions (1 if similar, 0 if different)
        predictions = (similarity > 0).cpu().numpy()
        
        # For each pair, check if they're from the same class
        true_labels = labels.astype(int)  # Changed this line
        
        print(f"Predictions: {np.unique(predictions, return_counts=True)}")
        print(f"True labels: {np.unique(true_labels, return_counts=True)}")
        print(f"Similarities range: [{similarity.min().item():.3f}, {similarity.max().item():.3f}]")
        
        # Ensure predictions and true_labels have the same shape
        assert len(predictions) == len(true_labels), f"Predictions shape {predictions.shape} != True labels shape {true_labels.shape}"
        
        # Compute balanced accuracy
        return balanced_accuracy_score(true_labels, predictions)


    def evaluate_model(self, model: SimCLRModel, loader: DataLoader, device: str) -> Tuple[float, float]:
        """
        Evaluate model on a data loader using balanced similarity-based metrics.
        Args:
            model: SimCLR model with encoder and projection head
            loader: DataLoader containing validation/test data
            device: Device to run evaluation on
        Returns:
            tuple: (balanced_accuracy, average_loss)
        """
        model.eval()
        total_loss = 0
        all_embeddings_1 = []
        all_embeddings_2 = []
        all_labels = []
        
        criterion = SimCLRLoss()
        
        with torch.no_grad():
            for spec1, spec2, labels in loader:
                # Move data to device
                spec1 = spec1.float().to(device)
                spec2 = spec2.float().to(device)
                labels = labels.float().to(device)
                
                # Get embeddings through model
                z1, z2 = model(spec1, spec2)
                
                # Store embeddings and labels for accuracy computation
                all_embeddings_1.append(z1)
                all_embeddings_2.append(z2)
                all_labels.append(labels)
                
                # Compute contrastive loss
                loss = criterion(z1, z2)
                total_loss += loss.item()
        
        # Concatenate all batches
        embeddings_1 = torch.cat(all_embeddings_1)
        embeddings_2 = torch.cat(all_embeddings_2)
        labels = torch.cat(all_labels)
        
        accuracy = self.compute_accuracy(embeddings_1, embeddings_2, labels)
        
        # Compute average loss
        avg_loss = total_loss / len(loader)
        
        return accuracy, avg_loss

def create_transformer(config: SimCLRConfig) -> nn.Module:
    """Creates a transformer encoder with proper configuration"""
    return Transformer(
        input_dim=config.input_dim,
        output_dim=config.embedding_dim,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout
    )

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = prepare_dataset(DataConfig())

    # Debug: Check first batch
    first_batch = next(iter(train_loader))
    print("\nFirst batch info:")
    print(f"x1 shape: {first_batch[0].shape}")
    print(f"x2 shape: {first_batch[1].shape}")
    print(f"labels shape: {first_batch[2].shape}")
    print(f"unique labels: {torch.unique(first_batch[2])}")
    
    # Input dimension is the number of features.
    input_dim = 2080
    
    # Create configs
    simclr_config = SimCLRConfig(
        input_dim=input_dim,
        embedding_dim=256,
        projection_dim=128,
        num_heads=4,
        hidden_dim=256,
        num_layers=4
    )
    
    # Create encoder
    encoder = create_transformer(simclr_config)
    
    # Create SimCLR model
    model = SimCLRModel(
        encoder=encoder,
        config=simclr_config
    ).to(device)
    
    # Create trainer
    trainer = SimCLRTrainer(model, simclr_config, device)
    
    # Training loop
    print("\nStarting training...")
    best_acc = 0
    for epoch in range(simclr_config.num_epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch+1}/{simclr_config.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        train_acc, _ = trainer.evaluate_model(model, train_loader, device)
        val_acc, _ = trainer.evaluate_model(model, val_loader, device)
        print(f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'accuracy': train_acc,
            }, "best_model.pth")
            print(f"New best model saved with accuracy: {train_acc:.2f}%")
        print("-" * 50)

if __name__ == "__main__":
    main()