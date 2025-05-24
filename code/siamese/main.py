import copy
import logging
import math
import os
import json # Moved from train_simclr
from collections import defaultdict # Not in original, but helpful if I were changing calculate_stats. Sticking to original.

import matplotlib.pyplot as plt # Moved from visualization functions
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict # Type was unused, but keeping imports as is mostly

# Assuming these local modules are in the same directory or PYTHONPATH
from lstm import LSTM
from transformer import Transformer
from cnn import CNN
from rcnn import RCNN
from mamba import Mamba
from kan import KAN
from vae import VAE
from MOE import MOE # Corrected casing if it was 'MOE' in user's files
from dense import Dense
from ode import ODE
from rwkv import RWKV
from tcn import TCN
from wavenet import WaveNet
from util import prepare_dataset, DataConfig

# Setup basic logging (can be configured further if needed)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
# Using original logging setup from train_simclr's main part

@dataclass
class SimCLRConfig:
    """Configuration for SimCLR model with default values."""
    # Optuna hyperparameters (original comments retained for context if they were from Optuna)
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
    weight_decay: float = 1e-6 # Original was 5.27e-05 from Optuna, script default 1e-6
    batch_size: int = 16 # Original was 32, but changed to 16 for better memory usage
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
            # nn.Dropout(dropout), # Original was commented out
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            # nn.Dropout(dropout), # Original was commented out
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)

class SimCLRModel(nn.Module):
    """SimCLR model combining encoder and projection head."""
    def __init__(self, encoder: nn.Module, config: SimCLRConfig):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(
            input_dim=config.embedding_dim, # This should be encoder's output feature dim
            hidden_dim=config.embedding_dim, # Projector's hidden dim
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
        
        return h1, None # If only x1 is provided

class SimCLRLoss(nn.Module):
    """NT-Xent loss for SimCLR."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, labels: Optional[torch.Tensor] = None): # labels from original, though not used in this NT-Xent
        batch_size = z1.shape[0]
        features = torch.cat([z1, z2], dim=0)
        
        similarity = torch.matmul(features, features.T) / self.temperature
        
        pos_mask = torch.zeros((2 * batch_size, 2 * batch_size), device=z1.device, dtype=torch.bool) # dtype=bool for direct masking
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size, device=z1.device, dtype=torch.bool)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size, device=z1.device, dtype=torch.bool)
        
        # Mask out self-contrast cases (diagonal)
        identity_mask = torch.eye(2 * batch_size, device=z1.device, dtype=torch.bool)
        
        # Numerator: positive pairs
        numerator = torch.exp(similarity[pos_mask]) # Shape (2*batch_size)

        # Denominator: sum over all non-self pairs
        # exp_sim for denominator should exclude diagonal elements
        exp_similarity_no_diag = torch.exp(similarity.masked_fill(identity_mask, -float('inf'))) # effectively removes diagonal for sum
        denominator = exp_similarity_no_diag.sum(dim=1) # Shape (2*batch_size)
        
        # Compute NT-Xent loss
        # Original calculation:
        # log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-7)
        # loss = -(pos_mask * log_prob).sum() / (2 * batch_size)
        # This can be simplified using the numerator/denominator logic derived from InfoNCE:
        # -log(positive_sim / sum_of_all_other_sims)
        # For each of the 2N samples, its positive pair is at a specific location.
        # The `labels` argument was in the original `SimCLRLoss.forward` signature but not used.
        # Replicating the original's sum and division logic more directly:

        # Mask for exp_sim calculation (exclude diagonal)
        exp_sim_mask = torch.ones((2 * batch_size, 2 * batch_size), device=z1.device)
        exp_sim_mask.fill_diagonal_(0)
        
        exp_sim = torch.exp(similarity) * exp_sim_mask # Masked to exclude diagonal from sum for log_prob
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-7) # 1e-7 for numerical stability

        # Original pos_mask was boolean. For multiplication, it might implicitly cast or use float version.
        # To be safe, if it was intended for element-wise mult with log_prob (float):
        # Re-using the integer pos_mask from original for indexing or direct use.
        # The original code had `pos_mask` for `torch.zeros`, not bool, for arithmetic.
        # Reverting to original pos_mask type for arithmetic:
        pos_mask_arith = torch.zeros((2 * batch_size, 2 * batch_size), device=z1.device)
        pos_mask_arith[:batch_size, batch_size:] = torch.eye(batch_size, device=z1.device)
        pos_mask_arith[batch_size:, :batch_size] = torch.eye(batch_size, device=z1.device)

        loss = -(pos_mask_arith * log_prob).sum() / (2 * batch_size)
        
        return loss

class SimCLRTrainer:
    def __init__(self, model: SimCLRModel, config: SimCLRConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        
        self.optimizer = optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': config.learning_rate},
            {'params': model.projector.parameters(), 'lr': config.learning_rate * 10},
        ], weight_decay=config.weight_decay)
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[config.learning_rate, config.learning_rate * 10],
            epochs=config.num_epochs,
            steps_per_epoch=500,  # Hardcoded in original, adjust based on your dataset size
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        self.contrastive_loss = SimCLRLoss(temperature=config.temperature)
        self.scaler = torch.amp.GradScaler(enabled=self.device.type == 'cuda') # Enable only for CUDA

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        all_embeddings_1, all_embeddings_2, all_labels_list = [], [], []
        
        for batch_idx, (x1, x2, labels_batch) in enumerate(train_loader): # labels_batch from loader
            x1, x2 = x1.float().to(self.device), x2.float().to(self.device)
            labels_batch = labels_batch.float().to(self.device) # labels_batch from loader to device
            
            with torch.amp.autocast(self.device.type):
                h1, h2 = self.model(x1, x2) # Projector outputs
                loss = self.contrastive_loss(h1, h2, labels_batch) # Pass labels_batch to loss (original behavior)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer) # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            all_embeddings_1.append(h1.detach()) # Using h1, h2 (projector outputs) for accuracy
            all_embeddings_2.append(h2.detach())
            all_labels_list.append(labels_batch) # Storing original labels for accuracy calculation
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        # Accuracy computation based on collected projected embeddings and labels
        accuracy = self._compute_accuracy(torch.cat(all_embeddings_1), torch.cat(all_embeddings_2), torch.cat(all_labels_list))
        return avg_loss, accuracy

    def evaluate_model(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        all_embeddings_1, all_embeddings_2, all_labels_list = [], [], []
        
        with torch.no_grad():
            for x1, x2, labels_batch in loader: # labels_batch from loader
                x1, x2 = x1.float().to(self.device), x2.float().to(self.device)
                labels_batch = labels_batch.float().to(self.device) # labels_batch from loader to device
                
                h1, h2 = self.model(x1, x2) # Projector outputs
                loss = self.contrastive_loss(h1, h2, labels_batch) # Pass labels_batch to loss
                
                all_embeddings_1.append(h1) # Using h1, h2 (projector outputs)
                all_embeddings_2.append(h2)
                all_labels_list.append(labels_batch)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        accuracy = self._compute_accuracy(torch.cat(all_embeddings_1), torch.cat(all_embeddings_2), torch.cat(all_labels_list))
        return accuracy, avg_loss # Original returned (accuracy, loss)

    @staticmethod
    def _compute_accuracy(embeddings_1: torch.Tensor, embeddings_2: torch.Tensor, 
                          labels: torch.Tensor) -> float:
        """Compute balanced classification accuracy based on embedding similarity."""
        # This method's logic for what 'labels' represent (e.g. pair-wise or sample-wise)
        # and how 'predictions' are derived is kept as per the original script.
        with torch.no_grad():
            # Normalizing again here, though projector output should already be normalized.
            # This was in the original, so keeping it.
            z1 = F.normalize(embeddings_1, dim=1)
            z2 = F.normalize(embeddings_2, dim=1)
            
            # Assuming labels are one-hot encoded class labels for the original samples.
            # The accuracy tries to see if cosine_similarity(z1_i, z2_i) > threshold
            # aligns with argmax(labels_i). This is an unconventional accuracy for SimCLR.
            # If labels are pair-wise as clarified ([1,0] neg, [0,1] pos):
            # argmax(labels, dim=1) would give 0 for neg, 1 for pos.
            true_labels_for_score = torch.argmax(labels, dim=1).cpu().numpy()
            
            similarity = F.cosine_similarity(z1, z2).cpu().numpy()
            threshold = np.mean(similarity) # Using mean as in original script
            predictions = (similarity > threshold).astype(int) # Predict 1 if > threshold, 0 otherwise
            
            return balanced_accuracy_score(true_labels_for_score, predictions)

def create_encoder(config: SimCLRConfig, encoder_type: str) -> nn.Module:
    """Factory function to create encoder based on type."""
    # This structure with many specific model parameters is kept as is.
    # To make this more concise, one might use **kwargs or a base class for encoders,
    # but that would be a larger refactoring possibly changing logic/initialization.
    encoder_mapping = {
        'transformer': lambda: Transformer(
            input_dim=config.input_dim, output_dim=config.embedding_dim, num_heads=config.num_heads,
            hidden_dim=config.hidden_dim, num_layers=config.num_layers, dropout=config.dropout),
        'cnn': lambda: CNN(
            input_dim=config.input_dim, output_dim=config.embedding_dim, d_model=128, # d_model hardcoded
            input_channels=1, dropout=config.dropout),
        'rcnn': lambda: RCNN(
            input_dim=config.input_dim, output_dim=config.embedding_dim, dropout=config.dropout),
        'lstm': lambda: LSTM(
            input_dim=config.input_dim, hidden_dim=config.hidden_dim, output_dim=config.embedding_dim,
            num_layers=config.num_layers, dropout=config.dropout),
        'mamba': lambda: Mamba(
            input_dim=config.input_dim, output_dim=config.embedding_dim, d_state=config.hidden_dim,
            num_layers=config.num_layers, dropout=config.dropout),
        'kan': lambda: KAN(
            input_dim=config.input_dim, hidden_dim=config.hidden_dim, output_dim=config.embedding_dim,
            num_inner_functions=10, dropout=config.dropout, num_layers=config.num_layers),
        'vae': lambda: VAE(
            input_dim=config.input_dim, output_dim=config.embedding_dim, hidden_dim=config.hidden_dim,
            latent_dim=config.hidden_dim, dropout=config.dropout),
        'moe': lambda: MOE(
            input_dim=config.input_dim, output_dim=config.embedding_dim, num_heads=config.num_heads,
            hidden_dim=config.hidden_dim, num_layers=config.num_layers, num_experts=4, k=2, dropout=config.dropout),
        'dense': lambda: Dense(input_dim=config.input_dim, output_dim=config.embedding_dim, dropout=config.dropout),
        'ode': lambda: ODE(input_dim=config.input_dim, output_dim=config.embedding_dim, dropout=config.dropout),
        'rwkv': lambda: RWKV(
            input_dim=config.input_dim, hidden_dim=config.hidden_dim, output_dim=config.embedding_dim, dropout=config.dropout),
        'tcn': lambda: TCN(input_dim=config.input_dim, output_dim=config.embedding_dim, dropout=config.dropout),
        'wavenet': lambda: WaveNet(input_dim=config.input_dim, output_dim=config.embedding_dim, dropout=config.dropout),
    }
    if encoder_type not in encoder_mapping:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")
    return encoder_mapping[encoder_type]()

def train_simclr_single_run(config: SimCLRConfig, encoder_type: str, run_id: int, device: torch.device,
                            train_loader: DataLoader, val_loader: DataLoader,
                            base_model_copy: nn.Module) -> Tuple[nn.Module, Dict]:
    logger = logging.getLogger(__name__) # Get logger instance
    logger.info(f"Starting run {run_id + 1}/{config.num_runs}")

    model = copy.deepcopy(base_model_copy).to(device)
    trainer = SimCLRTrainer(model, config, device)

    best_val_acc_this_run = 0.0 # Renamed for clarity, was best_val_acc
    best_model_state_file_this_run = None # Store only the filename
    best_metrics_this_run = None
    patience = 1000  # Original: Early stopping disabled
    patience_counter = 0

    for epoch in range(config.num_epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_acc, val_loss = trainer.evaluate_model(val_loader) # val_acc is first element

        if val_acc > best_val_acc_this_run:
            best_val_acc_this_run = val_acc
            
            current_model_filename = f"model_{encoder_type}_run_{run_id}.pth"
            torch.save(model.state_dict(), current_model_filename)
            best_model_state_file_this_run = current_model_filename # Keep track of the best file

            best_metrics_this_run = {
                'train_accuracy': train_acc, 'val_accuracy': val_acc,
                'train_loss': train_loss, 'val_loss': val_loss,
                'epoch': epoch # Store epoch for reference
            }
            patience_counter = 0
            # The original script had an inner `if run_id == 0 or val_acc > best_val_acc:` with `pass`.
            # This was likely a placeholder or dead code, as overall best model tracking
            # is handled in the `train_simclr` function. Removed for conciseness as it had no effect.
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 1 == 0: # Log progress every 10 epochs
            logger.info(f"Run {run_id + 1}, Epoch {epoch+1}/{config.num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%") # Assuming acc is 0-1
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

    if best_model_state_file_this_run: # If a best model was found and saved for this run
        model.load_state_dict(torch.load(best_model_state_file_this_run)) # Load the best state for this run
        # The epoch of this best model is in best_metrics_this_run['epoch']
        logger.info(f"Loaded best model for run {run_id+1} from epoch {best_metrics_this_run['epoch'] + 1} (file: {best_model_state_file_this_run})")
    else:
        logger.warning(f"No best model found or saved during training for run {run_id + 1}.")

    logger.info(f"Run {run_id + 1} completed. Best val accuracy for this run: {best_val_acc_this_run:.4f}")
    return model, best_metrics_this_run # Return model in its best state for this run, and its metrics

def calculate_stats(metrics_list: List[Dict]) -> Dict:
    """Calculate mean and standard deviation for each metric across runs."""
    if not metrics_list or not isinstance(metrics_list[0], dict): # Ensure list is not empty and contains dicts
        return {} 
        
    all_collected_metrics = {}
    # Initialize keys from the first valid run's metrics
    # Filter out None entries in metrics_list that might come from failed runs
    valid_metrics_list = [m for m in metrics_list if m is not None]
    if not valid_metrics_list:
        return {}

    for key in valid_metrics_list[0].keys():
        all_collected_metrics[key] = []
    
    for run_metrics in valid_metrics_list:
        for key, value in run_metrics.items():
            if key in all_collected_metrics: # Ensure key exists (it should if all dicts have same structure)
                 all_collected_metrics[key].append(value)
            # else: # Handle case where a run might have different metric keys (less likely with fixed structure)
            #     logger.warning(f"Metric key {key} not found in all_collected_metrics dict, skipping for this run.")
    
    stats = {}
    for key, values in all_collected_metrics.items():
        if not values: continue # Skip if no values were collected for a key
        stats[key] = {'mean': np.mean(values), 'std': np.std(values)}
    return stats

def train_simclr(config: SimCLRConfig, encoder_type: str = 'transformer') -> Tuple[Optional[SimCLRModel], Dict, List[Dict]]:
    logger = logging.getLogger(__name__) # Get logger instance
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Configure logger
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    data_config = DataConfig(batch_size=config.batch_size) # Assuming DataConfig is in util
    train_loader, val_loader = prepare_dataset(data_config)
    
    encoder = create_encoder(config, encoder_type)
    base_model = SimCLRModel(encoder=encoder, config=config)
    # This base_model_copy is passed to single runs; they deepcopy it again.
    # This is to ensure that if base_model_copy itself were modified (it's not here),
    # subsequent runs get a fresh version of the *original* base_model.
    base_model_for_runs = copy.deepcopy(base_model) 
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    # Potentially a models directory too, as done by train_simclr_single_run
    # os.makedirs(f"models/{encoder_type}", exist_ok=True) # Example if used

    all_runs_metrics_list = []
    best_overall_model = None
    best_overall_val_acc = float('-inf')
    
    logger.info(f"Starting {config.num_runs} independent runs for encoder: {encoder_type}")
    
    for run_id in range(config.num_runs):
        # Pass base_model_for_runs, which will be deepcopied inside train_simclr_single_run
        model_from_run, run_metrics = train_simclr_single_run(
            config=config, encoder_type=encoder_type, run_id=run_id, device=device,
            train_loader=train_loader, val_loader=val_loader, base_model_copy=base_model_for_runs
        )
        
        if run_metrics: # Check if run was successful and returned metrics
            all_runs_metrics_list.append(run_metrics)
            if run_metrics['val_accuracy'] > best_overall_val_acc:
                best_overall_val_acc = run_metrics['val_accuracy']
                best_overall_model = copy.deepcopy(model_from_run) # Save the best model object
                
                # Save the state dict of the best overall model
                best_overall_model_path = f"best_model_{encoder_type}_overall.pth"
                torch.save({
                    'model_state_dict': best_overall_model.state_dict(), # Save state_dict
                    'run_id': run_id,
                    'val_accuracy': best_overall_val_acc,
                    'encoder_type': encoder_type, # Store encoder type for reference
                    'config': vars(config) # Store config for reproducibility
                }, best_overall_model_path)
                logger.info(f"New best overall model from run {run_id + 1} saved to {best_overall_model_path} with val_accuracy: {best_overall_val_acc:.4f}")
        else:
            logger.warning(f"Run {run_id+1} for encoder {encoder_type} did not return metrics, possibly failed or was skipped.")

    stats = calculate_stats(all_runs_metrics_list)
    
    logger.info(f"\n--- Statistics for {encoder_type} across all runs ---")
    for metric, values in stats.items():
        logger.info(f"{metric.replace('_', ' ').capitalize()}: {values['mean']:.4f} ± {values['std']:.4f}")
    
    stats_file_path = f"results/stats_{encoder_type}.json"
    with open(stats_file_path, 'w') as f:
        # Save config along with stats for better record-keeping
        json.dump({'config': vars(config), 'stats': stats, 'all_runs_metrics': all_runs_metrics_list}, f, indent=4)
    logger.info(f"Saved detailed statistics to {stats_file_path}")

    if best_overall_model:
        logger.info("Creating visualizations for the best overall model...")
        visualize_batch_thresholds(best_overall_model, train_loader, device, title_prefix=f"Train ({encoder_type})", save_path=f"figures/train_contrastive_pairs_{encoder_type}.png")
        visualize_batch_thresholds(best_overall_model, val_loader, device, title_prefix=f"Val ({encoder_type})", save_path=f"figures/val_contrastive_pairs_{encoder_type}.png")
        if all_runs_metrics_list: # Only plot if there are metrics
            plot_runs_metrics(all_runs_metrics_list, encoder_type=encoder_type, save_path=f"figures/runs_metrics_{encoder_type}.png")
    else:
        logger.warning(f"No best overall model found for encoder {encoder_type} to visualize.")
        
    return best_overall_model, stats, all_runs_metrics_list

def visualize_batch_thresholds(model, loader, device, title_prefix="", save_path="figures/batch_thresholds.png"):
    """Visualize all batches with their individual thresholds."""
    # os and math imports moved to top
    os.makedirs("figures", exist_ok=True)
    
    model.eval()
    num_batches = len(loader)
    if num_batches == 0:
        logger.warning(f"Skipping visualization for {title_prefix}: DataLoader is empty.")
        return

    grid_size = math.ceil(math.sqrt(num_batches))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(max(8, 4*grid_size), max(8, 4*grid_size)), squeeze=False)
    axes = axes.ravel() # Ensure axes is always a flat array

    batch_idx = 0 # Initialize batch_idx before loop in case loader is empty after check
    with torch.no_grad():
        for batch_idx, (x1, x2, label_batch) in enumerate(loader):
            if batch_idx >= len(axes): break # Safety break if too many batches for grid

            x1, x2 = x1.float().to(device), x2.float().to(device)
            h1, h2 = model(x1, x2) # Get projector outputs
            
            similarities = F.cosine_similarity(h1, h2).cpu().numpy()
            # Assuming label_batch is one-hot for pair type or class for coloring
            true_labels_for_viz = torch.argmax(label_batch, dim=1).cpu().numpy() 
            
            threshold = np.mean(similarities) # Original used mean
            predictions = (similarities > threshold).astype(int)
            
            # Accuracy calculation here is for visualization per batch, might differ from main eval
            try:
                accuracy_viz = balanced_accuracy_score(true_labels_for_viz, predictions)
            except ValueError: # Handle cases where score cannot be computed (e.g. single class in true_labels_for_viz)
                accuracy_viz = np.nan # Not a number if score fails

            ax = axes[batch_idx]
            unique_display_labels = np.unique(true_labels_for_viz)
            for display_label in unique_display_labels:
                mask = true_labels_for_viz == display_label
                # Jitter x-axis based on the display_label for separation if multiple classes shown
                ax.scatter(np.random.normal(loc=display_label, scale=0.1, size=mask.sum()), 
                           similarities[mask], alpha=0.6, label=f'Class/PairType {display_label}')
            
            ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold (Mean): {threshold:.3f}')
            ax.set_ylabel('Cosine Similarity')
            ax.set_xlabel('Jittered X-axis (by True Label/PairType)')
            title = f'Batch {batch_idx+1}'
            if not np.isnan(accuracy_viz): title += f'\nAcc: {accuracy_viz:.3f}'
            ax.set_title(title)
            
            stats_text = (f'Mean Sim: {np.mean(similarities):.3f}\nStd Sim: {np.std(similarities):.3f}\nThreshold: {threshold:.3f}')
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            ax.legend(fontsize='small')
            ax.grid(True, linestyle=':', alpha=0.5)

    # Remove empty subplots if any
    for idx_empty in range(batch_idx + 1, len(axes)):
        fig.delaxes(axes[idx_empty])
    
    fig.suptitle(f'{title_prefix} Cosine Similarities & Thresholds per Batch'.strip(), fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    plt.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved batch threshold visualization to {save_path}")

def plot_runs_metrics(all_runs_metrics: List[Dict], encoder_type:str, save_path="figures/runs_metrics.png"):
    """Create box plots of metrics across runs."""
    # plt import moved to top
    valid_metrics_list = [m for m in all_runs_metrics if m is not None and isinstance(m, dict)]
    if not valid_metrics_list:
        logger.warning(f"No valid metrics to plot for {encoder_type}.")
        return

    # Extract metrics, assuming all dicts have same keys (take from first valid one)
    metric_keys = [k for k in valid_metrics_list[0].keys() if k != 'epoch'] # Exclude epoch from boxplot
    if not metric_keys:
        logger.warning(f"No metric keys (excluding 'epoch') found to plot for {encoder_type}.")
        return

    num_metrics = len(metric_keys)
    # Determine grid size for subplots
    ncols = 2
    nrows = math.ceil(num_metrics / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes = axes.flatten() # Flatten for easy indexing

    for i, metric_name in enumerate(metric_keys):
        values = [run_data[metric_name] for run_data in valid_metrics_list if metric_name in run_data]
        if not values: continue # Skip if no values for this metric

        axes[i].boxplot(values, patch_artist=True, medianprops={'color':'black'})
        axes[i].set_title(f"{metric_name.replace('_', ' ').capitalize()} ({encoder_type})")
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # Add individual points with jitter
        jitter = np.random.normal(1, 0.04, size=len(values)) # Jitter around position 1
        axes[i].scatter(jitter, values, alpha=0.6, color='red', zorder=3)
        
        mean_val, std_val = np.mean(values), np.std(values)
        axes[i].axhline(y=mean_val, color='green', linestyle='-', alpha=0.8, label=f'Mean: {mean_val:.4f}')
        axes[i].text(0.05, 0.95, f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}',
                       transform=axes[i].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        # axes[i].legend() # Legend for axhline might be too cluttered

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f'Metrics Distribution across {len(valid_metrics_list)} Runs for {encoder_type}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved runs metrics plot to {save_path}")

if __name__ == "__main__":
    # Ensure logger is configured for the main script execution
    # This basicConfig will apply if no handlers are already configured by `train_simclr`
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    config = SimCLRConfig() # Use default config values
    
    # Example: Train for a specific encoder or a list of them
    encoder_to_train = 'transformer' # Default from original function signature
    # To train multiple:
    # encoder_types = ['transformer', 'cnn', 'lstm']
    # for enc_type in encoder_types:
    #     logger.info(f"\n{'='*20} TRAINING ENCODER: {enc_type.upper()} {'='*20}")
    #     model, stats, all_runs_metrics = train_simclr(config, encoder_type=enc_type)
    #     if stats: # Check if stats were successfully computed
    #         logger.info(f"\nFinal Statistics for {enc_type}:")
    #         for metric, values in stats.items():
    #             logger.info(f"{metric.replace('_', ' ').capitalize()}: {values['mean']:.4f} ± {values['std']:.4f}")
    #     else:
    #         logger.info(f"No statistics computed for {enc_type}.")

    # Original main trains only one encoder type
    model, stats, all_runs_metrics = train_simclr(config, encoder_type=encoder_to_train)
    
    if stats: # Check if stats were successfully computed
        print("\nFinal Overall Statistics:")
        for metric, values in stats.items():
            print(f"{metric.replace('_', ' ').capitalize()}: {values['mean']:.4f} ± {values['std']:.4f}")
    else:
        print("No overall statistics computed (e.g., if all runs failed or no metrics were returned).")