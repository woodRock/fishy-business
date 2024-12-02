from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, List, Type
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, accuracy_score

from util import prepare_dataset, DataConfig
from lstm import LSTM
from transformer import Transformer
from cnn import CNN
from rcnn import RCNN
from mamba import Mamba
from kan import KAN
from vae import VAE
from rwkv import RWKV
from ode import ODE
from dense import Dense
from tcn import TCN
from MOE import MOE

@dataclass
class BaseConfig:
    """Base configuration for all encoders."""
    input_dim: int = 2080
    output_dim: int = 128
    dropout: float = 0.1

@dataclass
class TransformerConfig(BaseConfig):
    """Configuration for transformer encoder."""
    num_heads: int = 4
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.1

@dataclass
class CNNConfig(BaseConfig):
    """Configuration for CNN encoder."""
    input_channels: int = 1
    d_model: int = 128

@dataclass
class LSTMConfig(BaseConfig):
    """Configuration for LSTM encoder.""" 
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.1

@dataclass
class RCNNConfig(BaseConfig):
    """Configuration for RCNN encoder."""
    dropout: float = 0.3

@dataclass
class MambaConfig(BaseConfig):
    """Configuration for Mamba encoder."""
    d_state: int = 128
    num_layers: int = 4
    dropout: float = 0.1

@dataclass
class KANConfig(BaseConfig):
    """Configuration for KAN encoder."""
    hidden_dim: int = 128 
    num_inner_functions: int = 10

@dataclass
class VAEConfig(BaseConfig):
    """Configuration for VAE encoder."""
    hidden_dim: int = 128
    latent_dim: int = 128  # Changed to match classifier input dimension
    dropout: float = 0.1

@dataclass
class RWKVConfig(BaseConfig):
    hidden_dim: int = 128

@dataclass
class ODEConfig(BaseConfig):
    dropout: float = 0.3

@dataclass
class DenseConfig(BaseConfig):
    dropout: float = 0.3

@dataclass
class TCNConfig(BaseConfig):
    dropout: float = 0.3

@dataclass
class MOEConfig(BaseConfig):
    num_heads: int = 4
    hidden_dim: int = 128
    num_layers: int = 1
    num_experts: int = 4
    k: int = 2
    dropout: float = 0.1

@dataclass
class TrainConfig:
    """Configuration for training parameters."""
    learning_rate: float = 1e-5
    num_epochs: int = 200
    batch_size: int = 64
    temperature: float = 0.5
    margin: float = 1.0
    contrastive_weight: float = 0.5
    triplet_weight: float = 0.0
    crossentropy_weight: float = 0.5
    balanced_acc_weight: float = 0.0

class ModelFactory:
    """Factory for creating different types of encoders."""
    
    def __init__(self):
        self._encoders: Dict[str, Type[nn.Module]] = {}
        self._configs: Dict[str, Type[BaseConfig]] = {}
        
        # Register default encoders
        self.register_encoder("transformer", Transformer, TransformerConfig)
        self.register_encoder("cnn", CNN, CNNConfig)
        self.register_encoder("lstm", LSTM, LSTMConfig)
        self.register_encoder("rcnn", RCNN, RCNNConfig)
        self.register_encoder("mamba", Mamba, MambaConfig)
        self.register_encoder("kan", KAN, KANConfig)
        self.register_encoder("vae", VAE, VAEConfig)
        self.register_encoder("rwkv", RWKV, RWKVConfig)
        self.register_encoder("ode", ODE, ODEConfig)
        self.register_encoder("dense", Dense, DenseConfig)
        self.register_encoder("tcn", TCN, TCNConfig)
        self.register_encoder("moe", MOE, MOEConfig)
    
    def register_encoder(self, 
                        name: str, 
                        encoder_class: Type[nn.Module],
                        config_class: Type[BaseConfig]) -> None:
        """Register a new encoder type."""
        self._encoders[name] = encoder_class
        self._configs[name] = config_class
    
    def create_encoder(self, name: str, **config_kwargs) -> nn.Module:
        """Create an encoder instance of the specified type."""
        if name not in self._encoders:
            raise ValueError(f"Unknown encoder type: {name}")
        config = self._configs[name](**config_kwargs)
        return self._encoders[name](**asdict(config))
    
    def list_available_encoders(self) -> list[str]:
        """List all registered encoder types."""
        return list(self._encoders.keys())

class ContrastiveModel(nn.Module):
    def __init__(self, encoder: nn.Module, embedding_dim=128):
        super().__init__()
        self.encoder = encoder
        
        # Higher dropout and stochastic depth
        self.dropout_rate = 0.5  # Increased dropout
        self.drop_path = 0.2  # Stochastic depth rate
        
        # Mixup parameters
        self.mixup_alpha = 0.2
        
        # Feature processing with regularization
        self.feature_net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(embedding_dim),  # Added batch norm
        )
        
        # Embedding network with very aggressive regularization
        self.embedding_net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim//2),
            nn.LayerNorm(embedding_dim//2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(embedding_dim//2, embedding_dim//2),
            nn.BatchNorm1d(embedding_dim//2)
        )
        
        # Classification head with L1 regularization
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim//2, 2),
            nn.BatchNorm1d(2)
        )

    def _mixup(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Apply mixup augmentation"""
        if self.training:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            batch_size = x.size(0)
            index = torch.randperm(batch_size, device=x.device)
            mixed_x = lam * x + (1 - lam) * x[index]
            return mixed_x, index, lam
        return x, None, 1.0

    def _apply_stochastic_depth(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth during training"""
        if self.training and np.random.random() < self.drop_path:
            return torch.zeros_like(x)
        return x

    def forward(self, x1: torch.Tensor, 
                x2: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Get base features
        if isinstance(self.encoder, VAE):
            _, _, _, z1 = self.encoder(x1)
        else:
            z1 = self.encoder(x1)
        
        # Apply mixup if training
        z1, mix_idx, lam = self._mixup(z1)
        
        # Process features
        z1 = self.feature_net(z1)
        z1 = self._apply_stochastic_depth(z1)
        z1 = self.embedding_net(z1)
        
        if x2 is not None:
            if isinstance(self.encoder, VAE):
                _, _, _, z2 = self.encoder(x2)
            else:
                z2 = self.encoder(x2)
            z2 = self.feature_net(z2)
            z2 = self._apply_stochastic_depth(z2)
            z2 = self.embedding_net(z2)
            return z1, z2
        return z1, None

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.encoder, VAE):
            _, _, _, z = self.encoder(x)
        else:
            z = self.encoder(x)
        
        z = self.feature_net(z)
        z = self._apply_stochastic_depth(z)
        z = self.embedding_net(z)
        return self.classifier(z)

class LossComputer:
    """Handles computation of various losses used in contrastive learning."""
    
    # @staticmethod
    # def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, y: torch.Tensor, 
    #                     temperature: float = 0.5) -> torch.Tensor:
    #     similarity = nn.functional.cosine_similarity(z1, z2)
    #     loss = y * torch.pow(1 - similarity, 2) + \
    #            (1 - y) * torch.pow(torch.clamp(similarity - 0.1, min=0.0), 2)
    #     return loss.mean()

    @staticmethod
    def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, y: torch.Tensor, 
                        temperature: float = 0.1, margin: float = 0.5) -> torch.Tensor:
        """Cosine similarity loss with regularization"""
        # Normalize embeddings
        z1_norm = nn.functional.normalize(z1, dim=1)
        z2_norm = nn.functional.normalize(z2, dim=1)
        
        # Compute cosine similarity
        similarity = nn.functional.cosine_similarity(z1_norm, z2_norm)
        
        # Loss for positive and negative pairs with margin
        pos_loss = y * (1 - similarity).pow(2)
        neg_loss = (1 - y) * torch.clamp(similarity - margin, min=0.0).pow(2)
        
        # Add L2 regularization on embeddings
        l2_reg = 0.01 * (torch.norm(z1_norm, p=2) + torch.norm(z2_norm, p=2))
        
        return (pos_loss + neg_loss).mean() + l2_reg
    
    @staticmethod
    def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, 
                     negative: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        return torch.relu(distance_positive - distance_negative + margin).mean()
    
    @staticmethod
    def balanced_accuracy_loss(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ba = balanced_accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        return torch.tensor(1 - ba, device=preds.device)

class ContrastiveTrainer:
    """Handles training and evaluation of contrastive models."""
    
    def __init__(self, model: ContrastiveModel, config: TrainConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        self.loss_computer = LossComputer()
    
    def _get_triplets(self, z1: torch.Tensor, z2: torch.Tensor, 
                      y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates triplets for training from embeddings."""
        if torch.all(y == 1) or torch.all(y == 0):
            return z1, z1, z1
        
        positive_indices = torch.where(y == 1)[0]
        negative_indices = torch.where(y == 0)[0]
        
        num_triplets = min(len(positive_indices), len(negative_indices), len(z1))
        anchor_indices = torch.randperm(len(z1))[:num_triplets]
        
        anchor = z1[anchor_indices]
        positive = z2[positive_indices[torch.randperm(len(positive_indices))[:num_triplets]]]
        negative = z2[negative_indices[torch.randperm(len(negative_indices))[:num_triplets]]]
        
        return anchor, positive, negative
    
    def compute_loss(self, z1: torch.Tensor, z2: torch.Tensor, y: torch.Tensor, 
                    logits: torch.Tensor) -> torch.Tensor:
        """Computes the combined loss."""
        similarity = nn.functional.cosine_similarity(z1, z2)
        preds = (similarity > 0.5).float()
        
        # Get triplets
        anchor, positive, negative = self._get_triplets(z1, z2, y)
        
        # Compute individual losses
        cl = self.loss_computer.contrastive_loss(z1, z2, y, self.config.temperature)
        tl = self.loss_computer.triplet_loss(anchor, positive, negative, self.config.margin)
        ce = nn.functional.cross_entropy(logits, y.long())
        bal = self.loss_computer.balanced_accuracy_loss(preds, y)
        
        # Combine losses with weights
        return (self.config.contrastive_weight * cl +
                self.config.triplet_weight * tl +
                self.config.crossentropy_weight * ce +
                self.config.balanced_acc_weight * bal)
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """Trains for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        
        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
            y = y.squeeze()
            
            self.optimizer.zero_grad()
            z1, z2 = self.model(x1, x2)
            logits = self.model.classify(x1)
            
            loss = self.compute_loss(z1, z2, y, logits)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            similarity = nn.functional.cosine_similarity(z1, z2)
            preds = (similarity > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy, balanced_acc

    # @torch.no_grad()
    # def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
    #     """Evaluates the model."""
    #     self.model.eval()
    #     all_preds, all_labels = [], []
        
    #     for x1, x2, y in dataloader:
    #         x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
    #         z1, z2 = self.model(x1, x2)
    #         similarity = nn.functional.cosine_similarity(z1, z2)
    #         preds = (similarity > 0.5).float()
            
    #         all_preds.extend(preds.cpu().numpy())
    #         all_labels.extend(y.cpu().numpy())
        
    #     accuracy = accuracy_score(all_labels, all_preds)
    #     balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    #     return accuracy, balanced_acc

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        all_sims, all_labels = [], []
        
        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
            z1, z2 = self.model(x1, x2)
            
            # Get normalized embeddings
            z1_norm = nn.functional.normalize(z1, dim=1)
            z2_norm = nn.functional.normalize(z2, dim=1)
            
            similarity = nn.functional.cosine_similarity(z1_norm, z2_norm)
            
            all_sims.extend(similarity.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        # Find best threshold on validation set
        thresholds = np.linspace(0, 1, 100)
        best_acc = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            preds = (np.array(all_sims) > threshold).astype(int)
            acc = balanced_accuracy_score(all_labels, preds)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        
        # Use best threshold for final predictions
        preds = (np.array(all_sims) > best_threshold).astype(int)
        accuracy = accuracy_score(all_labels, preds)
        balanced_acc = balanced_accuracy_score(all_labels, preds)
        
        return accuracy, balanced_acc

def create_model(
    encoder_type: str,
    input_dim: int,
    embedding_dim: int = 128,
    num_classes: int = 2,
    factory: Optional[ModelFactory] = None,
    **encoder_kwargs
) -> ContrastiveModel:
    """Helper function to create a complete model."""
    if factory is None:
        factory = ModelFactory()
    
    # Ensure all base config parameters are included
    base_config = {
        'input_dim': input_dim,
        'output_dim': embedding_dim,
    }
    
    # Update encoder_kwargs with base_config
    encoder_kwargs.update(base_config)
    
    encoder = factory.create_encoder(encoder_type, **encoder_kwargs)
    return ContrastiveModel(encoder)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    config = DataConfig()
    print("\nStarting data preparation...")
    train_loader, val_loader = prepare_dataset(config)

    # Create factory and model
    factory = ModelFactory()
    input_dim = 2080
    
    # Example: Create transformer-based model
    model = create_model(
        encoder_type="transformer",
        input_dim = input_dim,
        factory=factory
    ).to(device)
    
    # Training setup
    train_config = TrainConfig()
    trainer = ContrastiveTrainer(model, train_config, device)
    
    # Training loop
    best_val_accuracy = 0
    for epoch in tqdm(range(train_config.num_epochs), desc="Training"):
        train_loss, train_acc, train_bal_acc = trainer.train_epoch(train_loader)
        train_acc, train_bal_acc = trainer.evaluate(train_loader)
        val_acc, val_bal_acc = trainer.evaluate(val_loader)
        
        print(f"Epoch {epoch+1}/{train_config.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Train Balanced Accuracy: {train_bal_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Validation Balanced Accuracy: {val_bal_acc:.4f}")
        print("-" * 24)
        
        if val_bal_acc > best_val_accuracy:
            best_val_accuracy = val_bal_acc
            torch.save(model.state_dict(), "best_model.pth")

        # Update learning rate based on validation performance
        trainer.scheduler.step(val_bal_acc)  
    
    print(f"Best Validation Balanced Accuracy: {best_val_accuracy:.4f}")

if __name__ == "__main__":
    main()