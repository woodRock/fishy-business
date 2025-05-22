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

import copy
import logging
import matplotlib.pyplot as plt
import os
import math

@dataclass
class SimCLRConfig:
    """Configuration for SimCLR model with default values."""
    # Optuna hyperparameters
    # {
    # 'learning_rate': 3.523812306701682e-05, 
    # 'weight_decay': 5.279302459232037e-05, 
    # 'temperature': 0.5497380139231042, 
    # 'projection_dim': 256, 
    # 'embedding_dim': 256, 
    # 'hidden_dim': 256, 
    # 'num_layers': 6, 
    # 'dropout': 0.17848238562756857, 
    # 'batch_size': 32, 
    # 'num_heads': 8, 
    # }
    temperature: float = 0.5497380139231042
    projection_dim: int = 256
    embedding_dim: int = 256
    learning_rate: float = 3.523812306701682e-05
    weight_decay: float = 1e-6
    batch_size: int = 32
    num_epochs: int = 1000
    input_dim: int = 2080
    num_heads: int = 8
    hidden_dim: int = 256
    num_layers: int = 6
    dropout: float = 0.17848238562756857
    num_runs: int = 30  # Number of independent runs


class ProjectionHead(nn.Module):
    """Non-linear projection head for SimCLR."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            # nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            # nn.Dropout(dropout),
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
            output_dim=config.projection_dim,
            dropout=config.dropout
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
        
        self.contrastive_loss = SimCLRLoss(temperature=config.temperature)
        self.scaler = torch.amp.GradScaler()
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        all_embeddings_1, all_embeddings_2, all_labels = [], [], []
        
        for batch_idx, (x1, x2, labels) in enumerate(train_loader):
            x1, x2 = x1.float().to(self.device), x2.float().to(self.device)
            labels = labels.float().to(self.device)
            
            with torch.amp.autocast(self.device.type):
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

def train_simclr_single_run(config: SimCLRConfig, encoder_type: str, run_id: int, device: torch.device, 
                           train_loader: DataLoader, val_loader: DataLoader, 
                           base_model_copy: nn.Module) -> Tuple[nn.Module, Dict]:
    """Train SimCLR model for a single run."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting run {run_id + 1}/{config.num_runs}")
    
    # Create a new model from the base copy for this run
    model = copy.deepcopy(base_model_copy).to(device)
    trainer = SimCLRTrainer(model, config, device)
    
    best_val_acc = 0
    best_model_state = None
    best_metrics = None
    patience = 1000  # Early stopping disabled
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_acc, val_loss = trainer.evaluate_model(val_loader)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
            # Save the best model state
            torch.save(model.state_dict(), f"model_{encoder_type}_run_{run_id}.pth")

            best_model_state = {
                'epoch': epoch,
                'filename': f"model_{encoder_type}_run_{run_id}.pth",
                # 'optimizer_state_dict': copy.deepcopy(trainer.optimizer.state_dict()),
            }
            
            best_metrics = {
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            patience_counter = 0
            # Save only the best overall model from all runs
            if run_id == 0 or val_acc > best_val_acc:
                # torch.save(best_model_state, f"best_model_{encoder_type}_run_{run_id}.pth")
                pass 
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
                
        # Log progress every 10 epochs to avoid excessive output
        if (epoch + 1) % 10 == 0:
            logger.info(f"Run {run_id + 1}, Epoch {epoch+1}/{config.num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Load the best model from the filepath saved during training
    if best_model_state:
        model.load_state_dict(torch.load(best_model_state['filename']))
        logger.info(f"Loaded best model from epoch {best_model_state['epoch'] + 1}")
    else:   
        logger.warning("No best model found during training.")
    
    logger.info(f"Run {run_id + 1} completed. Best val accuracy: {best_val_acc:.4f}")
    
    return model, best_metrics

def calculate_stats(metrics_list: List[Dict]) -> Dict:
    """Calculate mean and standard deviation for each metric across runs."""
    all_metrics = {}
    
    # Initialize with first run's metrics
    for key in metrics_list[0].keys():
        all_metrics[key] = []
    
    # Collect metrics from all runs
    for run_metrics in metrics_list:
        for key, value in run_metrics.items():
            all_metrics[key].append(value)
    
    # Calculate mean and std for each metric
    stats = {}
    for key, values in all_metrics.items():
        stats[key] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    return stats

def train_simclr(config: SimCLRConfig, encoder_type: str = 'transformer') -> Tuple[SimCLRModel, Dict, Dict]:
    """Train SimCLR model with multiple independent runs for statistical evaluation."""
    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Prepare dataset
    data_config = DataConfig(batch_size=config.batch_size)
    train_loader, val_loader = prepare_dataset(data_config)
    
    # Create base model to be copied for each run
    encoder = create_encoder(config, encoder_type)
    base_model = SimCLRModel(encoder=encoder, config=config)
    base_model_copy = copy.deepcopy(base_model)
    
    # Create a directory for saving models and visualizations
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    # Storage for all runs
    all_runs_metrics = []
    best_overall_model = None
    best_overall_val_acc = float('-inf')
    
    logger.info(f"Starting {config.num_runs} independent runs")
    
    # Perform independent runs
    for run_id in range(config.num_runs):
        model, run_metrics = train_simclr_single_run(
            config=config,
            encoder_type=encoder_type,
            run_id=run_id,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            base_model_copy=base_model_copy
        )
        
        all_runs_metrics.append(run_metrics)
        
        # Update best overall model if this run performed better
        if run_metrics['val_accuracy'] > best_overall_val_acc:
            best_overall_val_acc = run_metrics['val_accuracy']
            best_overall_model = copy.deepcopy(model)
            torch.save({
                'model_state_dict': model.state_dict(),
                'run_id': run_id,
                'val_accuracy': best_overall_val_acc
            }, f"best_model_{encoder_type}_overall.pth")
            logger.info(f"New best overall model from run {run_id + 1} with val accuracy: {best_overall_val_acc:.4f}")
    
    # Calculate statistics across all runs
    stats = calculate_stats(all_runs_metrics)
    
    # Print overall statistics
    logger.info("\nStatistics across all runs:")
    for metric, values in stats.items():
        logger.info(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    
    # Save statistics to file
    import json
    with open(f"results/stats_{encoder_type}.json", 'w') as f:
        json.dump(stats, f, indent=4)
    
    # Create visualization for the best model
    logger.info("Creating visualizations for the best model")
    visualize_batch_thresholds(best_overall_model, train_loader, device, save_path=f"figures/train_contrastive_pairs_{encoder_type}.png")
    visualize_batch_thresholds(best_overall_model, val_loader, device, save_path=f"figures/val_contrastive_pairs_{encoder_type}.png")
    
    # Create box plots of metrics across runs
    plot_runs_metrics(all_runs_metrics, save_path=f"figures/runs_metrics_{encoder_type}.png")
    
    return best_overall_model, stats, all_runs_metrics

def visualize_batch_thresholds(model, loader, device, save_path="figures/batch_thresholds.png"):
    """Visualize all batches with their individual thresholds."""
    import os
    import matplotlib.pyplot as plt
    import math
    
    # Create figures directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    
    model.eval()
    
    # First pass to count batches
    num_batches = len(loader)
    
    # Calculate grid dimensions
    grid_size = math.ceil(math.sqrt(num_batches))
    
    # Create a grid of subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(4*grid_size, 4*grid_size))
    # The validation set is a single batch, so we need to handle that.
    axes = axes.ravel() if num_batches > 1 else [axes]
    
    with torch.no_grad():
        for batch_idx, (x1, x2, label) in enumerate(loader):
            x1, x2 = x1.float().to(device), x2.float().to(device)
            
            # Get projected embeddings (like in training)
            h1, h2 = model(x1, x2)  # Get projector outputs
            
            # Compute cosine similarity
            similarities = F.cosine_similarity(h1, h2).cpu().numpy()
            labels = torch.argmax(label, dim=1).cpu().numpy()
            
            # Compute batch threshold
            threshold = np.mean(similarities)
            predictions = (similarities > threshold).astype(int)
            accuracy = balanced_accuracy_score(labels, predictions)
            
            # Plot for this batch
            ax = axes[batch_idx]
            
            # Create scatter plot for this batch
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                ax.scatter(np.random.normal(0, 0.1, size=mask.sum()), 
                         similarities[mask],
                         alpha=0.6,
                         label=f'Class {label}')
            
            # Plot threshold
            ax.axhline(y=threshold, color='r', linestyle='--', 
                      label=f'Threshold: {threshold:.3f}')
            
            ax.set_ylabel('Cosine Similarity')
            ax.set_xlabel('Jittered x-axis')
            ax.set_title(f'Batch {batch_idx+1}\nAccuracy: {accuracy:.3f}')
            
            # Add stats text
            stats_text = (f'Mean: {np.mean(similarities):.3f}\n'
                         f'Std: {np.std(similarities):.3f}\n'
                         f'Threshold: {threshold:.3f}')
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.legend()
    
    # Remove empty subplots
    for idx in range(batch_idx + 1, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Cosine Similarities and Thresholds per Batch', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_runs_metrics(all_runs_metrics, save_path="figures/runs_metrics.png"):
    """Create box plots of metrics across runs."""
    import matplotlib.pyplot as plt
    
    # Extract metrics
    metrics = {}
    for key in all_runs_metrics[0].keys():
        metrics[key] = [run[key] for run in all_runs_metrics]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Plot each metric
    for i, (metric, values) in enumerate(metrics.items()):
        axes[i].boxplot(values)
        axes[i].set_title(f'{metric} across {len(all_runs_metrics)} runs')
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # Add individual points
        x = np.random.normal(1, 0.1, size=len(values))
        axes[i].scatter(x, values, alpha=0.6, c='r')
        
        # Add mean and std annotation
        mean_val = np.mean(values)
        std_val = np.std(values)
        axes[i].axhline(y=mean_val, color='g', linestyle='-', alpha=0.8)
        axes[i].text(0.05, 0.95, f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}',
                   transform=axes[i].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    config = SimCLRConfig()
    model, stats, all_runs_metrics = train_simclr(config)
    
    print("\nFinal Statistics:")
    for metric, values in stats.items():
        print(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")