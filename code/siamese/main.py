from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Type
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score

from lstm import LSTM
from transformer import Transformer
from cnn import CNN
from rcnn import RCNN
from mamba import Mamba
from kan import KAN
from vae import VAE
from MOE import MOE
from dense import Dense
from ode import ODE 
from rwkv import RWKV
from tcn import TCN
from wavenet import WaveNet
from util import prepare_dataset, DataConfig

@dataclass
class SimCLRConfig:
    """Configuration for SimCLR model with default values."""
    temperature: float = 0.5
    projection_dim: int = 128
    embedding_dim: int = 512
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
    """Non-linear projection head for SimCLR."""
    
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
    """SimCLR model combining encoder and projection head."""
    
    def __init__(self, encoder: nn.Module, config: SimCLRConfig):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(
            input_dim=config.embedding_dim,
            hidden_dim=config.embedding_dim,
            output_dim=config.projection_dim
        )
    
    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z1 = self.encoder(x1)
        h1 = self.projector(z1)
        
        if x2 is not None:
            z2 = self.encoder(x2)
            h2 = self.projector(z2)
            return h1, h2
        
        return h1, None

class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2, labels):
        batch_size = z1.shape[0]
        features = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs
        pos_mask = torch.zeros((2 * batch_size, 2 * batch_size), device=z1.device)
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size)
        
        # Mask out self-contrast cases
        mask = torch.eye(2 * batch_size, device=z1.device)
        mask = 1 - mask
        
        # Compute NT-Xent loss
        exp_sim = torch.exp(similarity) * mask
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-7)
        loss = -(pos_mask * log_prob).sum() / (2 * batch_size)
        
        return loss


class SimCLRTrainer:
    def __init__(self, model: SimCLRModel, config: SimCLRConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        
        # Separate learning rates for encoder and projector
        self.optimizer = optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': config.learning_rate},
            {'params': model.projector.parameters(), 'lr': config.learning_rate * 10},
        ], weight_decay=config.weight_decay)
        
        # OneCycleLR scheduler with warmup
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[config.learning_rate, config.learning_rate * 10],
            epochs=config.num_epochs,
            steps_per_epoch=500,  # Adjust based on your dataset size
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        self.contrastive_loss = SimCLRLoss()
        self.scaler = torch.cuda.amp.GradScaler()
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        all_embeddings_1, all_embeddings_2, all_labels = [], [], []
        
        for batch_idx, (x1, x2, labels) in enumerate(train_loader):
            x1, x2 = x1.float().to(self.device), x2.float().to(self.device)
            labels = labels.float().to(self.device)
            
            with torch.cuda.amp.autocast():
                z1, z2 = self.model(x1, x2)
                loss = self.contrastive_loss(z1, z2, labels)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            all_embeddings_1.append(z1.detach())
            all_embeddings_2.append(z2.detach())
            all_labels.append(labels)
            total_loss += loss.item()
        
        embeddings_1 = torch.cat(all_embeddings_1)
        embeddings_2 = torch.cat(all_embeddings_2)
        labels = torch.cat(all_labels)
        
        accuracy = self._compute_accuracy(embeddings_1, embeddings_2, labels)
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, accuracy
    
    def evaluate_model(self, loader: DataLoader) -> Tuple[float, float]:
        """Evaluate the model on given data loader."""
        self.model.eval()
        total_loss = 0
        all_embeddings_1, all_embeddings_2, all_labels = [], [], []
        
        with torch.no_grad():
            for x1, x2, labels in loader:
                x1, x2 = x1.float().to(self.device), x2.float().to(self.device)
                labels = labels.float().to(self.device)
                
                z1, z2 = self.model(x1, x2)
                loss = self.contrastive_loss(z1, z2, labels)
                
                all_embeddings_1.append(z1)
                all_embeddings_2.append(z2)
                all_labels.append(labels)
                total_loss += loss.item()
        
        embeddings_1 = torch.cat(all_embeddings_1)
        embeddings_2 = torch.cat(all_embeddings_2)
        labels = torch.cat(all_labels)
        
        accuracy = self._compute_accuracy(embeddings_1, embeddings_2, labels)
        avg_loss = total_loss / len(loader)
        
        return accuracy, avg_loss
    
    @staticmethod
    def _compute_accuracy(embeddings_1: torch.Tensor, embeddings_2: torch.Tensor, 
                         labels: torch.Tensor) -> float:
        """Compute balanced classification accuracy based on embedding similarity."""
        with torch.no_grad():
            z1 = F.normalize(embeddings_1, dim=1)
            z2 = F.normalize(embeddings_2, dim=1)
            labels = torch.argmax(labels, dim=1).cpu().numpy()
            
            similarity = F.cosine_similarity(z1, z2).cpu().numpy()
            threshold = np.mean(similarity)
            predictions = (similarity > threshold).astype(int)
            
            return balanced_accuracy_score(labels, predictions)

def create_encoder(config: SimCLRConfig, encoder_type: str) -> nn.Module:
    """Factory function to create encoder based on type with appropriate parameters."""
    encoder_mapping = {
        'transformer': lambda: Transformer(
            input_dim=config.input_dim,
            output_dim=config.embedding_dim,
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout
        ),
        'cnn': lambda: CNN(
            input_dim=config.input_dim,
            output_dim=config.embedding_dim,
            d_model=128,
            input_channels=1,
            dropout=config.dropout,
        ),
        'rcnn': lambda: RCNN(
            input_dim=config.input_dim,
            output_dim=config.embedding_dim,
            dropout=config.dropout,
        ),
        'lstm': lambda: LSTM(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.embedding_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        ),
        'mamba': lambda: Mamba(
            input_dim=config.input_dim,
            output_dim=config.embedding_dim,
            d_state=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        ),
        'kan': lambda: KAN(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.embedding_dim,
            num_inner_functions=10,
            dropout=config.dropout,
            num_layers=config.num_layers,
        ),
        'vae': lambda: VAE(
            input_dim=config.input_dim,
            output_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.hidden_dim,
            dropout=config.dropout,
        ),
        'moe': lambda: MOE(
            input_dim=config.input_dim,
            output_dim=config.embedding_dim,
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_experts=4,
            k=2,
            dropout=config.dropout,
        ),
        'dense': lambda: Dense(
            input_dim=config.input_dim,
            output_dim=config.embedding_dim,
            dropout=config.dropout,
        ),
        'ode': lambda: ODE(
            input_dim=config.input_dim,
            output_dim=config.embedding_dim,
            dropout=config.dropout,
        ),
        'rwkv': lambda: RWKV(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.embedding_dim,
            dropout=config.dropout,
        ),
        'tcn': lambda: TCN(
            input_dim=config.input_dim,
            output_dim=config.embedding_dim,
            dropout=config.dropout,
        ),
        'wavenet': lambda: WaveNet(
            input_dim=config.input_dim,
            output_dim=config.embedding_dim,
            dropout=config.dropout,
        ),
    }
    
    if encoder_type not in encoder_mapping:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")
    
    return encoder_mapping[encoder_type]()

def train_simclr(config: SimCLRConfig, encoder_type: str = 'vae') -> Tuple[SimCLRModel, Dict, Dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = prepare_dataset(DataConfig())
    
    encoder = create_encoder(config, encoder_type)
    model = SimCLRModel(encoder=encoder, config=config).to(device)
    trainer = SimCLRTrainer(model, config, device)
    
    best_val_acc = 0
    best_model_state = None
    best_metrics = None
    patience = 1000 # Early stopping disabled.
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_acc, val_loss = trainer.evaluate_model(val_loader)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
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
            patience_counter = 0
            torch.save(best_model_state, f"best_model_{encoder_type}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
    return model, best_model_state, best_metrics

if __name__ == "__main__":
    config = SimCLRConfig()
    model, best_state, best_metrics = train_simclr(config)
    # Print the best metrics.,]
    print(best_metrics)