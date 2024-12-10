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
from rcnn import RCNN
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
    num_epochs: int = 1000
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
            nn.LayerNorm(input_dim),
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

class SimCLRLoss(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, z1, z2, labels):
        z1 = F.normalize(z1 + 1e-8, dim=1)
        z2 = F.normalize(z2 + 1e-8, dim=1)
        
        similarities = F.cosine_similarity(z1, z2)
        label_pairs = torch.argmax(labels, dim=1)
        
        # Scale similarities more aggressively
        probs = torch.clamp((similarities + 1) / 2, min=1e-6, max=1-1e-6)
        
        # Remove label smoothing to allow perfect classification
        loss = F.binary_cross_entropy(probs, label_pairs.float())
        
        return loss
    
def compute_accuracy(embeddings_1, embeddings_2, labels):
    """Basic accuracy computation"""
    with torch.no_grad():  # Ensure no gradients are computed
        z1 = F.normalize(embeddings_1, dim=1)
        z2 = F.normalize(embeddings_2, dim=1)
        labels = torch.argmax(labels, dim=1).cpu().numpy()
        
        # Use cosine similarity and detach tensors
        similarity = F.cosine_similarity(z1.detach(), z2.detach()).cpu().numpy()
        
        # Simple threshold at mean
        threshold = np.mean(similarity)
        predictions = (similarity > threshold).astype(int)
        
        return balanced_accuracy_score(labels, predictions)
        
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
        ], weight_decay=config.weight_decay)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        self.contrastive_loss = SimCLRLoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0

        all_embeddings_1 = []
        all_embeddings_2 = []
        all_labels = []
        
        for batch_idx, (x1, x2, labels) in enumerate(train_loader):
            # Move to device and ensure float32
            x1 = x1.float().to(self.device)
            x2 = x2.float().to(self.device)
            labels = labels.float().to(self.device)
            
            # Forward pass and get projections
            z1, z2 = self.model(x1, x2)
            
            # Compute contrastive loss
            contrastive_loss = self.contrastive_loss(z1, z2, labels)
            loss = contrastive_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Store embeddings and labels for accuracy computation
            all_embeddings_1.append(z1)
            all_embeddings_2.append(z2)
            all_labels.append(labels)
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Batch [{batch_idx}/{len(train_loader)}], '
                    f'Loss: {loss.item():.4f}, '
                    f'Contrastive: {contrastive_loss.item():.4f}, ')

        # Concatenate all batches
        embeddings_1 = torch.cat(all_embeddings_1)
        embeddings_2 = torch.cat(all_embeddings_2)
        labels = torch.cat(all_labels)
        
        accuracy = compute_accuracy(embeddings_1, embeddings_2, labels)
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
        """Basic accuracy computation"""
        with torch.no_grad():  # Ensure no gradients are computed
            z1 = F.normalize(embeddings_1, dim=1)
            z2 = F.normalize(embeddings_2, dim=1)
            labels = torch.argmax(labels, dim=1).cpu().numpy()
            
            # Use cosine similarity and detach tensors
            similarity = F.cosine_similarity(z1.detach(), z2.detach()).cpu().numpy()
            
            # Simple threshold at mean
            threshold = np.mean(similarity)
            predictions = (similarity > threshold).astype(int)
            
            return balanced_accuracy_score(labels, predictions)

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
                loss = criterion(z1, z2, labels)
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

def create_cnn(config: SimCLRConfig) -> nn.Module:
    """Creates a cnn encoder"""
    return CNN (
        input_dim=config.input_dim,
        output_dim=config.embedding_dim,
        d_model=128,
        input_channels=1,
        dropout=config.dropout,
    )

def create_rcnn(config: SimCLRConfig) -> nn.Module:
    """Creates a rcnn encoder"""
    return RCNN (
        input_dim=config.input_dim,
        output_dim=config.embedding_dim,
        dropout=config.dropout,
    )

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = prepare_dataset(DataConfig())
    
    # Input dimension is the number of features
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
    
    # Create encoder and model
    encoder = create_rcnn(simclr_config)
    model = SimCLRModel(
        encoder=encoder,
        config=simclr_config
    ).to(device)
    
    # Create trainer
    trainer = SimCLRTrainer(model, simclr_config, device)
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0
    best_model_state = None
    best_epoch = 0
    best_train_acc = 0
    best_metrics = None
    
    for epoch in range(simclr_config.num_epochs):
        # Training
        train_loss, train_acc = trainer.train_epoch(train_loader)
        
        # Evaluation
        train_acc, train_loss = trainer.evaluate_model(model, train_loader, device)
        val_acc, val_loss = trainer.evaluate_model(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{simclr_config.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_acc = train_acc
            best_epoch = epoch
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
            }
            best_metrics = {
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            torch.save(best_model_state, "best_model.pth")
            print(f"New best model saved! Validation accuracy: {val_acc:.2f}%")
        print("-" * 50)
    
    # Load best model and do final evaluation
    print(f"\nLoading best model from epoch {best_epoch+1}...")
    model.load_state_dict(best_model_state['model_state_dict'])
    
    print("\nFinal Results (Best Model):")
    print(f"Best Epoch: {best_epoch+1}")
    print(f"Train Accuracy: {best_metrics['train_accuracy']:.2f}%")
    print(f"Train Loss: {best_metrics['train_loss']:.4f}")
    print(f"Validation Accuracy: {best_metrics['val_accuracy']:.2f}%")
    print(f"Validation Loss: {best_metrics['val_loss']:.4f}")
    
    return model, best_model_state, best_metrics

if __name__ == "__main__":
    best_model, best_state, best_metrics = main()