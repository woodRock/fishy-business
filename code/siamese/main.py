# -*- coding: utf-8 -*-
"""
Main entry point for training SimCLR models with various encoders.

This script orchestrates the training and evaluation of a Siamese network using the SimCLR
framework. It supports multiple encoder architectures, handles data preparation, runs
training for a specified number of iterations, and provides detailed statistics and
visualizations.

Example Usage:
--------------
# Train a SimCLR model with a Transformer encoder
python -m siamese.main --encoder_type transformer --num_runs 10

# Train with a CNN encoder and different hyperparameters
python -m siamese.main --encoder_type cnn --num_runs 30 --batch_size 32 --learning_rate 1e-4

"""

# ## 1. Imports and Setup
# -----------------------
# Standard library, third-party, and local module imports.

import argparse
import copy
import json
import logging
import math
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Type

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score

from models import (
    Transformer,
    CNN,
    RCNN,
    LSTM,
    Mamba,
    KAN,
    VAE,
    MOE,
    Dense,
    ODE,
    RWKV,
    TCN,
    WaveNet,
)
from .util import prepare_dataset, DataConfig

# ## 2. Configuration
# --------------------
# Centralized configuration for the SimCLR model and training process.


@dataclass
class SimCLRConfig:
    """Configuration for SimCLR model training."""

    temperature: float = 0.55
    projection_dim: int = 256
    embedding_dim: int = 256
    learning_rate: float = 3.5e-5
    weight_decay: float = 1e-6
    batch_size: int = 16
    num_epochs: int = 1000
    input_dim: int = 2080
    num_heads: int = 8
    hidden_dim: int = 256
    num_layers: int = 6
    dropout: float = 0.18
    num_runs: int = 30
    patience: int = 1000
    encoder_type: str = "transformer"


# ## 3. Core SimCLR Components
# -----------------------------
# These classes define the main components of the SimCLR framework:
# the projection head, the combined model, and the NT-Xent loss function.


class ProjectionHead(nn.Module):
    """A non-linear projection head for mapping encoder outputs to a latent space."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ) -> None:
        """ Initializes the projection head with a sequence of layers.
        
        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Dimension of the output features.
            dropout (float): Dropout rate to apply after the hidden layer.

        The projection head consists of:
            - Layer normalization to stabilize training.
            - A linear layer to project input features to a hidden dimension.       
            - Batch normalization to normalize the hidden layer outputs.
            - ReLU activation for non-linearity.
            - A final linear layer to project to the output dimension.
            - The output is normalized to unit length.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the projection head.
        
        Args: 
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Normalized output tensor of shape (batch_size, output_dim).
        """
        return F.normalize(self.net(x), dim=1)


class SimCLRModel(nn.Module):
    """Combines an encoder with a projection head to form the full SimCLR model."""

    def __init__(self, encoder: nn.Module, config: SimCLRConfig) -> None:
        """ Initializes the SimCLR model with an encoder and a projection head.

        The model consists of:
            - An encoder to process input data and extract features.
            - A projection head to map the encoder outputs to a latent space.   
        The projection head includes:
            - Layer normalization to stabilize training.
            - A linear layer to project input features to a hidden dimension.
            - Batch normalization to normalize the hidden layer outputs.
            - ReLU activation for non-linearity.
            - A final linear layer to project to the output dimension.
            - The output is normalized to unit length.  
        
        Args: 
            encoder (nn.Module): The encoder network to extract features.
            config (SimCLRConfig): Configuration object containing model parameters.
        """
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(
            input_dim=config.embedding_dim,
            hidden_dim=config.embedding_dim,
            output_dim=config.projection_dim,
            dropout=config.dropout,
        )

    def forward(
        self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """ Forward pass through the SimCLR model.
        
        Args: 
            x1 (torch.Tensor): First input tensor of shape (batch_size, input_dim).
            x2 (Optional[torch.Tensor]): Second input tensor of shape (batch_size, input_dim).
                If provided, the model computes embeddings for both inputs.
       
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - h1: Projected output for x1 of shape (batch_size, projection_dim).
                - h2: Projected output for x2 of shape (batch_size, projection_dim).
                  If x2 is None, h2 will be None.
        """
        z1 = self.encoder(x1)
        h1 = self.projector(z1)
        if x2 is not None:
            z2 = self.encoder(x2)
            h2 = self.projector(z2)
            return h1, h2
        return h1, None


class SimCLRLoss(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy loss (NT-Xent)."""

    def __init__(self, temperature: float):
        """ Initializes the NT-Xent loss function with a temperature parameter.

        The loss function computes the cosine similarity between pairs of embeddings,
        applies temperature scaling, and calculates the cross-entropy loss.
        
        Args: 
            temperature (float): Temperature scaling factor for the loss function.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        """ Forward pass to compute the NT-Xent loss.

        This function computes the cosine similarity between the concatenated embeddings
        of two sets of inputs, applies temperature scaling, and calculates the loss.   
        
        Args: 
            z1 (torch.Tensor): First set of embeddings of shape (batch_size, projection_dim
            z2 (torch.Tensor): Second set of embeddings of shape (batch_size, projection_dim).

        Returns:
            torch.Tensor: Computed NT-Xent loss value.
        """
        batch_size = z1.shape[0]
        features = torch.cat([z1, z2], dim=0)
        similarity = F.cosine_similarity(
            features.unsqueeze(1), features.unsqueeze(0), dim=2
        )

        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
        similarity = similarity[~mask].view(2 * batch_size, -1)

        positives = torch.cat(
            [F.cosine_similarity(z1, z2, dim=1), F.cosine_similarity(z2, z1, dim=1)],
            dim=0,
        )
        positives = positives.view(2 * batch_size, 1)

        numerator = torch.exp(positives / self.temperature)
        denominator = torch.sum(
            torch.exp(similarity / self.temperature), dim=1, keepdim=True
        )

        loss = -torch.log(numerator / denominator).mean()
        return loss


# ## 4. Encoder Factory
# ----------------------
# A factory function to create different encoder architectures.

ENCODER_REGISTRY: Dict[str, Type[nn.Module]] = {
    "transformer": Transformer,
    "cnn": CNN,
    "rcnn": RCNN,
    "lstm": LSTM,
    "mamba": Mamba,
    "kan": KAN,
    "vae": VAE,
    "moe": MOE,
    "dense": Dense,
    "ode": ODE,
    "rwkv": RWKV,
    "tcn": TCN,
    "wavenet": WaveNet,
}


def create_encoder(config: SimCLRConfig) -> nn.Module:
    """Creates an encoder instance based on the type specified in the config.
    
    Args: 
        config (SimCLRConfig): Configuration object containing encoder type and parameters.

    Returns:
        nn.Module: An instance of the specified encoder class initialized with the provided parameters.
    """
    encoder_class = ENCODER_REGISTRY.get(config.encoder_type)
    if not encoder_class:
        raise ValueError(f"Unsupported encoder type: {config.encoder_type}")

    # Define arguments for each model type
    args = {
        "input_dim": config.input_dim,
        "output_dim": config.embedding_dim,
        "dropout": config.dropout,
    }
    if config.encoder_type in ["transformer", "lstm", "moe", "rwkv"]:
        args.update(
            {
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
            }
        )
    if config.encoder_type in ["transformer", "moe"]:
        args["num_heads"] = config.num_heads
    if config.encoder_type == "cnn":
        args.update({"d_model": 128, "input_channels": 1})
    if config.encoder_type == "mamba":
        args.update({"d_state": config.hidden_dim, "num_layers": config.num_layers})
    if config.encoder_type == "kan":
        args.update(
            {
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "num_inner_functions": 10,
            }
        )
    if config.encoder_type == "vae":
        args.update({"hidden_dim": config.hidden_dim, "latent_dim": config.hidden_dim})
    if config.encoder_type == "moe":
        args.update({"num_experts": 4, "k": 2})

    # For models that only need input_dim, output_dim, and dropout
    simple_models = ["rcnn", "dense", "ode", "tcn", "wavenet"]
    if config.encoder_type in simple_models:
        return encoder_class(
            input_dim=args["input_dim"],
            output_dim=args["output_dim"],
            dropout=args["dropout"],
        )

    return encoder_class(**args)


# ## 5. Training and Evaluation
# ------------------------------
# The SimCLRTrainer class encapsulates the training and evaluation logic.


class SimCLRTrainer:
    """Manages the training and evaluation of the SimCLR model."""

    def __init__(self, model: SimCLRModel, config: SimCLRConfig, device: torch.device) -> None:
        """ Initializes the SimCLRTrainer with a model, configuration, and device.
        
        Args: 
            model (SimCLRModel): The SimCLR model to train and evaluate.
            config (SimCLRConfig): Configuration object containing training parameters.
            device (torch.device): The device (CPU or GPU) to run the model on
        """
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.num_epochs,
            steps_per_epoch=500,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=10000.0,
        )
        self.contrastive_loss = SimCLRLoss(temperature=config.temperature)
        self.scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")

    def _run_epoch(
        self, data_loader: DataLoader, is_training: bool
    ) -> Tuple[float, float]:
        """Runs a single epoch of training or evaluation.

        This method processes the data loader, computes the model outputs, applies the contrastive loss,
        and updates the model parameters if in training mode. It also computes the average loss and accuracy
        for the epoch.
        
        Args: 
            data_loader (DataLoader): DataLoader for the current epoch.
            is_training (bool): Flag indicating whether this is a training epoch.

        Returns:
            Tuple[float, float]: Average loss and accuracy for the epoch.
        """
        self.model.train(is_training)
        total_loss = 0.0
        all_h1, all_h2, all_labels = [], [], []

        context = torch.no_grad() if not is_training else torch.enable_grad()
        with context:
            for x1, x2, labels in data_loader:
                x1, x2, labels = (
                    x1.float().to(self.device),
                    x2.float().to(self.device),
                    labels.to(self.device),
                )

                with torch.amp.autocast(self.device.type):
                    h1, h2 = self.model(x1, x2)
                    loss = self.contrastive_loss(h1, h2)

                if is_training:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()

                total_loss += loss.item()
                all_h1.append(h1.detach())
                all_h2.append(h2.detach())
                all_labels.append(labels.detach())

        avg_loss = total_loss / len(data_loader)
        accuracy = self._compute_accuracy(
            torch.cat(all_h1), torch.cat(all_h2), torch.cat(all_labels)
        )
        return avg_loss, accuracy

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """ Runs a single training epoch.
        
        Args: 
            train_loader (DataLoader): DataLoader for the training data.

        Returns:
            Tuple[float, float]: Average loss and accuracy for the training epoch.
        """
        return self._run_epoch(train_loader, is_training=True)

    def evaluate_model(self, val_loader: DataLoader) -> Tuple[float, float]:
        """ Evaluates the model on the validation set.

        This method computes the model outputs, applies the contrastive loss,
        and calculates the average loss and accuracy for the validation set.    
        
        Args: 
            val_loader (DataLoader): DataLoader for the validation data.

        Returns:
            Tuple[float, float]: Average loss and accuracy for the validation epoch.
        """
        return self._run_epoch(val_loader, is_training=False)

    @staticmethod
    def _compute_accuracy(
        h1: torch.Tensor, h2: torch.Tensor, labels: torch.Tensor
    ) -> float:
        """Computes balanced accuracy based on cosine similarity.
        
        Args: 
            h1 (torch.Tensor): First set of projected embeddings.
            h2 (torch.Tensor): Second set of projected embeddings.
            labels (torch.Tensor): True labels for the pairs.

        Returns:
            float: Balanced accuracy score for the predictions.
        """
        true_labels = torch.argmax(labels, dim=1).cpu().numpy()
        similarity = F.cosine_similarity(h1, h2).cpu().numpy()
        threshold = np.mean(similarity)
        predictions = (similarity > threshold).astype(int)
        return balanced_accuracy_score(true_labels, predictions)


# ## 6. Visualization
# --------------------
# Functions for creating and saving visualizations.


def visualize_batch_thresholds(
    model: SimCLRModel,
    loader: DataLoader,
    device: torch.device,
    title_prefix: str,
    save_path: str,
) -> None:
    """Visualizes cosine similarities and decision thresholds for each batch.
    
    This function evaluates the model on the provided DataLoader, computes cosine similarities
    between pairs of embeddings, and visualizes the results in a grid of scatter plots.
    Each plot shows the cosine similarities for a batch of pairs, colored by their true labels.
    It also computes a decision threshold based on the mean similarity and displays the accuracy
    of predictions based on this threshold.
    The resulting figure is saved to the specified path.
    If the DataLoader is empty, the function exits without creating a plot.

    Args: 
        model (SimCLRModel): The trained SimCLR model to evaluate.
        loader (DataLoader): DataLoader for the validation or test set.
        device (torch.device): The device (CPU or GPU) to run the model on.
        title_prefix (str): Prefix for the plot title.
        save_path (str): Path to save the visualization figure. 
    """
    model.eval()
    num_batches = len(loader)
    if num_batches == 0:
        return

    grid_size = math.ceil(math.sqrt(num_batches))
    fig, axes = plt.subplots(
        grid_size,
        grid_size,
        figsize=(max(8, 4 * grid_size), max(8, 4 * grid_size)),
        squeeze=False,
    )
    axes = axes.ravel()

    with torch.no_grad():
        for i, (x1, x2, labels) in enumerate(loader):
            if i >= len(axes):
                break
            h1, h2 = model(x1.float().to(device), x2.float().to(device))
            similarities = F.cosine_similarity(h1, h2).cpu().numpy()
            true_labels = torch.argmax(labels, dim=1).cpu().numpy()
            threshold = np.mean(similarities)
            predictions = (similarities > threshold).astype(int)
            accuracy = balanced_accuracy_score(true_labels, predictions)

            ax = axes[i]
            for label_val in np.unique(true_labels):
                mask = true_labels == label_val
                ax.scatter(
                    np.random.normal(loc=label_val, scale=0.1, size=mask.sum()),
                    similarities[mask],
                    alpha=0.6,
                    label=f"Pair Type {label_val}",
                )
            ax.axhline(
                y=threshold,
                color="r",
                linestyle="--",
                label=f"Threshold: {threshold:.3f}",
            )
            ax.set_title(f"Batch {i+1}\nAcc: {accuracy:.3f}")
            ax.legend()

    fig.suptitle(f"{title_prefix} Cosine Similarities per Batch", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)


def plot_runs_metrics(all_runs_metrics: List[Dict], encoder_type: str, save_path: str) -> None:
    """Creates box plots of metrics across multiple runs.
    
    This function takes a list of dictionaries containing metrics from multiple training runs,
    extracts the relevant metrics, and creates box plots for each metric. Each plot shows the distribution
    of the metric values across all runs, with jittered points for individual run values.   

    Args: 
        all_runs_metrics (List[Dict]): List of dictionaries containing metrics for each run.
        encoder_type (str): Type of encoder used in the runs, used for plot titles.
        save_path (str): Path to save the box plot figure.  
    """
    if not all_runs_metrics:
        return
    metric_keys = [k for k in all_runs_metrics[0].keys() if k != "epoch"]
    ncols = 2
    nrows = math.ceil(len(metric_keys) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
    )
    axes = axes.flatten()

    for i, metric in enumerate(metric_keys):
        values = [run[metric] for run in all_runs_metrics]
        axes[i].boxplot(values, patch_artist=True)
        axes[i].set_title(f"{metric.replace('_', ' ').capitalize()} ({encoder_type})")
        jitter = np.random.normal(1, 0.04, size=len(values))
        axes[i].scatter(jitter, values, alpha=0.6)

    fig.suptitle(f"Metrics Distribution for {encoder_type}", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ## 7. Main Orchestration
# -------------------------
# The main function that orchestrates the entire training and evaluation process.


def run_single_training(
    config: SimCLRConfig,
    run_id: int,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    base_model: SimCLRModel,
) -> Tuple[SimCLRModel, Dict]:
    """Executes a single training run.

    This function initializes the model, sets up the trainer, and runs the training and evaluation
    for a specified number of epochs. It tracks the best validation accuracy and saves the model state
    if the validation accuracy improves. It also implements early stopping based on a patience parameter.   
    
    Args: 
        config (SimCLRConfig): Configuration object containing training parameters.
        run_id (int): Identifier for the current run.
        device (torch.device): The device (CPU or GPU) to run the model on.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        base_model (SimCLRModel): The base model to train.

    Returns:
        Tuple[SimCLRModel, Dict]: The trained model and a dictionary containing the best metrics
        for the run, including training and validation loss and accuracy.
    """
    logging.info(f"Starting run {run_id + 1}/{config.num_runs}")
    model = copy.deepcopy(base_model).to(device)
    trainer = SimCLRTrainer(model, config, device)
    best_val_acc = 0.0
    best_metrics = {}
    patience_counter = 0

    for epoch in range(config.num_epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate_model(val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "epoch": epoch,
            }
            torch.save(
                model.state_dict(), f"model_{config.encoder_type}_run_{run_id}.pth"
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            logging.info(f"Run {run_id+1}, Epoch {epoch+1}: Val Acc: {val_acc:.2f}%")

    model.load_state_dict(torch.load(f"model_{config.encoder_type}_run_{run_id}.pth"))
    return model, best_metrics


def main(config: SimCLRConfig):
    """Main function to run the SimCLR training and evaluation.
    
    Args: 
        config (SimCLRConfig): Configuration object containing training parameters.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logging.info(f"Using device: {device}")

    data_config = DataConfig(batch_size=config.batch_size)
    train_loader, val_loader = prepare_dataset(data_config)

    encoder = create_encoder(config)
    base_model = SimCLRModel(encoder=encoder, config=config)

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    all_runs_metrics = []
    best_overall_model = None
    best_overall_val_acc = float("-inf")

    for i in range(config.num_runs):
        model, metrics = run_single_training(
            config, i, device, train_loader, val_loader, base_model
        )
        if metrics:
            all_runs_metrics.append(metrics)
            if metrics["val_accuracy"] > best_overall_val_acc:
                best_overall_val_acc = metrics["val_accuracy"]
                best_overall_model = copy.deepcopy(model)
                torch.save(
                    best_overall_model.state_dict(),
                    f"best_model_{config.encoder_type}_overall.pth",
                )

    if all_runs_metrics:
        stats = {
            key: {
                "mean": np.mean([m[key] for m in all_runs_metrics]),
                "std": np.std([m[key] for m in all_runs_metrics]),
            }
            for key in all_runs_metrics[0]
        }
        logging.info(f"Statistics for {config.encoder_type}: {stats}")
        with open(f"results/stats_{config.encoder_type}.json", "w") as f:
            json.dump(
                {"config": asdict(config), "stats": stats, "runs": all_runs_metrics},
                f,
                indent=4,
            )

        plot_runs_metrics(
            all_runs_metrics,
            config.encoder_type,
            f"figures/runs_metrics_{config.encoder_type}.png",
        )

    if best_overall_model:
        visualize_batch_thresholds(
            best_overall_model,
            val_loader,
            device,
            f"Val ({config.encoder_type})",
            f"figures/val_contrastive_pairs_{config.encoder_type}.png",
        )


if __name__ == "__main__":
    """ Entry point for the script.
    Parses command line arguments, initializes the configuration, and calls the main function."""
    parser = argparse.ArgumentParser(description="Train SimCLR models.")
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="transformer",
        choices=ENCODER_REGISTRY.keys(),
    )
    parser.add_argument("--num_runs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3.5e-5)
    # Add other config arguments as needed
    args = parser.parse_args()

    config = SimCLRConfig(
        encoder_type=args.encoder_type,
        num_runs=args.num_runs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    main(config)
