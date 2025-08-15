# -*- coding: utf-8 -*-
"""
Main script for training and evaluating contrastive models using contrastive learning methods.

This script supports various contrastive learning methods such as SimCLR, MoCo, BYOL, SimSiam, and Barlow Twins.
It includes model definitions, loss functions, and training routines.
It is designed to be flexible and extensible, allowing for easy integration of new models and methods.
It also includes utilities for data preprocessing, model evaluation, and visualization of results.

Example usage:
    python3 -m contrastive.main --encoder_type transformer --contrastive_method simclr --num_runs 3 --batch_size 16

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
from sklearn.model_selection import StratifiedGroupKFold

from models import (
    Transformer,
    Ensemble,
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
    Hybrid,
    # Longformer,
    Performer,
    SimCLRModel,
    SimCLRLoss,
    MoCoModel,
    MoCoLoss,
    BYOLModel,
    BYOLLoss,
    SimSiamModel,
    SimSiamLoss,
    BarlowTwinsModel,
    BarlowTwinsLoss,
)
from deep_learning.util import AugmentationConfig, DataAugmenter
from .util import DataConfig, DataPreprocessor, SiameseDataset, BalancedBatchSampler

# ## 2. Configuration
# --------------------
# Centralized configuration for the SimCLR model and training process.


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive model training."""

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
    num_inner_functions: int = 10  # Added
    dropout: float = 0.18
    num_runs: int = 30
    patience: int = 100
    encoder_type: str = "transformer"
    contrastive_method: str = "simclr"  # New: specifies the contrastive learning method

    # MoCo specific
    moco_dim: int = 256
    moco_k: int = 65536
    moco_m: float = 0.999
    moco_mlp: bool = False

    # BYOL specific
    byol_projection_dim: int = 256
    byol_hidden_dim: int = 4096
    byol_m: float = 0.996

    # SimSiam specific
    simsiam_projection_dim: int = 2048
    simsiam_hidden_dim: int = 512

    # Barlow Twins specific
    barlow_twins_projection_dim: int = 8192
    barlow_twins_lambda: float = 5e-3

    # Augmentation parameters
    noise_enabled: bool = False
    shift_enabled: bool = False
    scale_enabled: bool = False
    crop_enabled: bool = False
    flip_enabled: bool = False
    permutation_enabled: bool = False
    noise_level: float = 0.1
    crop_size: float = 0.8  # Added
    trial_number: Optional[int] = None # For unique model saving in Optuna trials


class VAEEncoderWrapper(nn.Module):
    def __init__(self, vae_model: nn.Module):
        super().__init__()
        self.vae_model = vae_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # VAE.forward returns recon_x, mu, logvar, class_probs
        # We need the latent representation 'z' for contrastive learning
        mu, logvar = self.vae_model.encode(x)
        z = self.vae_model.reparameterize(mu, logvar)
        return z


# ## 3. Core SimCLR Components
# -----------------------------
# These classes define the main components of the SimCLR framework:
# the projection head, the combined model, and the NT-Xent loss function.


# SimCLRModel, SimCLRLoss, and ProjectionHead moved to models/simclr.py


# ## 4. Encoder Factory
# ----------------------
# A factory function to create different encoder architectures.

ENCODER_REGISTRY: Dict[str, Type[nn.Module]] = {
    "transformer": Transformer,
    "ensemble": Ensemble,
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
    "hybrid": Hybrid,
    # "longformer": Longformer,
    "performer": Performer,
}

CONTRASTIVE_MODEL_REGISTRY: Dict[str, Tuple[Type[nn.Module], Type[nn.Module]]] = {
    "simclr": (SimCLRModel, SimCLRLoss),
    "moco": (MoCoModel, MoCoLoss),
    "byol": (BYOLModel, BYOLLoss),
    "simsiam": (SimSiamModel, SimSiamLoss),
    "barlow_twins": (BarlowTwinsModel, BarlowTwinsLoss),
}


def create_backbone_encoder(config: ContrastiveConfig) -> nn.Module:
    """Creates an encoder instance based on the type specified in the config."""
    encoder_class = ENCODER_REGISTRY.get(config.encoder_type)
    if not encoder_class:
        raise ValueError(f"Unsupported encoder type: {config.encoder_type}")

    args = {
        "dropout": config.dropout,
    }

    if config.encoder_type == "cnn":
        return encoder_class(
            input_dim=config.input_dim,
            output_dim=config.embedding_dim,
            dropout=config.dropout,
        )
    elif config.encoder_type == "mamba":
        args.update(
            {
                "input_dim": config.input_dim,  # Add input_dim
                "d_model": config.embedding_dim,  # Output dimension of the encoder
                "d_state": config.hidden_dim,
                "d_conv": 4,  # Default value for d_conv
                "expand": 2,  # Default value for expand
                "depth": config.num_layers,
            }
        )
    elif config.encoder_type == "kan":
        args.update(
            {
                "input_dim": config.input_dim,
                "output_dim": config.embedding_dim,
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "num_inner_functions": config.num_inner_functions,
                "dropout_rate": args.pop("dropout"),  # Rename dropout to dropout_rate
            }
        )
    elif config.encoder_type == "vae":
        args["input_size"] = config.input_dim  # Rename input_dim to input_size
        args["num_classes"] = config.embedding_dim  # Rename output_dim to num_classes
        args["latent_dim"] = config.hidden_dim  # Map hidden_dim to latent_dim
    elif config.encoder_type == "moe":
        args.update(
            {
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
                "num_experts": 4,
                "k": 2,
            }
        )
    elif config.encoder_type in ["transformer", "lstm", "rwkv"]:
        args.update(
            {
                "input_dim": config.input_dim,
                "output_dim": config.embedding_dim,
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
            }
        )
        if config.encoder_type == "transformer":
            args["num_heads"] = config.num_heads
    elif config.encoder_type == "ensemble":
        args.update(
            {
                "input_dim": config.input_dim,
                "output_dim": config.embedding_dim,
                "hidden_dim": config.hidden_dim,
            }
        )
    elif config.encoder_type in ["hybrid", "longformer", "performer"]:
        args.update(
            {
                "input_dim": config.input_dim,
                "output_dim": config.embedding_dim,
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
            }
        )
        if config.encoder_type == "longformer":
            args["attention_window"] = 32  # Default attention window for Longformer
            args["hidden_dim"] = 64
            args["num_layers"] = 1
            args["num_heads"] = 4
        elif config.encoder_type == "performer":
            args["num_random_features"] = 256  # Default for Performer
    else:  # For simple_models and others that take input_dim, output_dim, dropout
        args.update(
            {
                "input_dim": config.input_dim,
                "output_dim": config.embedding_dim,
            }
        )

    encoder = encoder_class(**args)

    if config.encoder_type == "vae":
        return VAEEncoderWrapper(encoder)
    return encoder


def create_contrastive_model(
    config: ContrastiveConfig,
) -> Tuple[nn.Module, nn.Module]:
    """Creates the full contrastive learning model and its loss function."""
    model_class, loss_class = CONTRASTIVE_MODEL_REGISTRY.get(config.contrastive_method)
    if not model_class:
        raise ValueError(f"Unsupported contrastive method: {config.contrastive_method}")

    backbone_encoder = create_backbone_encoder(config)

    if config.contrastive_method == "simclr":
        model = SimCLRModel(encoder=backbone_encoder, config=config)
        loss_fn = SimCLRLoss(temperature=config.temperature)
    elif config.contrastive_method == "moco":
        # MoCoModel expects dim, K, m, T, mlp
        model = MoCoModel(
            encoder=backbone_encoder,
            encoder_output_dim=config.embedding_dim,
            dim=config.moco_dim,
            K=config.moco_k,
            m=config.moco_m,
            T=config.temperature,
            mlp=config.moco_mlp,
        )
        loss_fn = MoCoLoss(T=config.temperature)
    elif config.contrastive_method == "byol":
        # BYOLModel expects projection_dim, hidden_dim, m
        model = BYOLModel(
            encoder=backbone_encoder,
            encoder_output_dim=config.embedding_dim,
            projection_dim=config.byol_projection_dim,
            hidden_dim=config.byol_hidden_dim,
            m=config.byol_m,
        )
        loss_fn = BYOLLoss()
    elif config.contrastive_method == "simsiam":
        # SimSiamModel expects projection_dim, hidden_dim
        model = SimSiamModel(
            encoder=backbone_encoder,
            encoder_output_dim=config.embedding_dim,
            projection_dim=config.simsiam_projection_dim,
            hidden_dim=config.simsiam_hidden_dim,
        )
        loss_fn = SimSiamLoss()
    elif config.contrastive_method == "barlow_twins":
        # BarlowTwinsModel expects projection_dim
        model = BarlowTwinsModel(
            encoder=backbone_encoder,
            encoder_output_dim=config.embedding_dim,
            projection_dim=config.barlow_twins_projection_dim,
        )
        loss_fn = BarlowTwinsLoss(lambda_param=config.barlow_twins_lambda)
    else:
        raise ValueError(f"Unhandled contrastive method: {config.contrastive_method}")

    return model, loss_fn


# ## 5. Training and Evaluation
# ------------------------------
# The SimCLRTrainer class encapsulates the training and evaluation logic.


class ContrastiveTrainer:
    """Manages the training and evaluation of the contrastive model."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: ContrastiveConfig,
        device: torch.device,
    ) -> None:
        """Initializes the ContrastiveTrainer."""
        self.model = model
        self.loss_fn = loss_fn
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
        self.scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")
        self.best_threshold = 0.5

    def _run_epoch(
        self, data_loader: DataLoader, is_training: bool
    ) -> Tuple[float, float]:
        """Runs a single epoch of training or evaluation."""
        if len(data_loader) == 0:
            logging.warning("Data loader is empty, skipping epoch.")
            return 0.0, 0.0

        self.model.train(is_training)
        total_loss = 0.0
        all_h1, all_h2, all_labels = [], [], []

        # Create a DataAugmenter instance with the current config
        aug_config = AugmentationConfig(
            enabled=True,  # Always enabled for contrastive learning
            noise_enabled=self.config.noise_enabled,
            shift_enabled=self.config.shift_enabled,
            scale_enabled=self.config.scale_enabled,
            crop_enabled=self.config.crop_enabled,
            flip_enabled=self.config.flip_enabled,
            permutation_enabled=self.config.permutation_enabled,
            noise_level=self.config.noise_level,
            crop_size=self.config.crop_size,
        )
        data_augmenter = DataAugmenter(aug_config)

        context = torch.no_grad() if not is_training else torch.enable_grad()
        with context:
            for x1_raw, x2_raw, labels in data_loader:
                # Apply augmentations to x1_raw and x2_raw
                # DataAugmenter methods expect numpy arrays, so convert
                x1_np = x1_raw.cpu().numpy()
                x2_np = x2_raw.cpu().numpy()

                # Apply augmentations to create two views
                x1_aug = (
                    torch.from_numpy(
                        data_augmenter._apply_augmentations_to_batch(x1_np)
                    )
                    .float()
                    .to(self.device)
                )
                x2_aug = (
                    torch.from_numpy(
                        data_augmenter._apply_augmentations_to_batch(x2_np)
                    )
                    .float()
                    .to(self.device)
                )
                labels = labels.float().to(self.device)

                with torch.amp.autocast(self.device.type):
                    # Forward pass depends on the contrastive method
                    if self.config.contrastive_method == "simclr":
                        h1, h2 = self.model(x1_aug, x2_aug)
                        loss = self.loss_fn(h1, h2)
                    elif self.config.contrastive_method == "moco":
                        q, k, queue = self.model(x1, x2)
                        loss = self.loss_fn(q, k, queue)
                    elif self.config.contrastive_method == "byol":
                        p1, z2, p2, z1 = self.model(x1, x2)
                        loss = self.loss_fn(p1, z2, p2, z1)
                    elif self.config.contrastive_method == "simsiam":
                        p1, z2, p2, z1 = self.model(x1, x2)
                        loss = self.loss_fn(p1, z2, p2, z1)
                    elif self.config.contrastive_method == "barlow_twins":
                        z1, z2 = self.model(x1, x2)
                        loss = self.loss_fn(z1, z2)
                    else:
                        raise ValueError(
                            f"Unhandled contrastive method for forward pass: {self.config.contrastive_method}"
                        )

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
                # Collect embeddings for accuracy calculation
                if self.config.contrastive_method == "simclr":
                    all_h1.append(h1.detach())
                    all_h2.append(h2.detach())
                elif self.config.contrastive_method == "moco":
                    all_h1.append(q.detach())
                    all_h2.append(k.detach())
                elif self.config.contrastive_method == "byol":
                    # For BYOL, use the projected outputs (z1, z2) from the target network
                    # The model's forward returns (p1, z2, p2, z1)
                    all_h1.append(z1.detach())
                    all_h2.append(z2.detach())
                elif self.config.contrastive_method == "simsiam":
                    # For SimSiam, use the projected outputs (z1, z2)
                    # The model's forward returns (p1, z2, p2, z1)
                    all_h1.append(z1.detach())
                    all_h2.append(z2.detach())
                elif self.config.contrastive_method == "barlow_twins":
                    # For Barlow Twins, use the normalized outputs (z1, z2)
                    all_h1.append(z1.detach())
                    all_h2.append(z2.detach())
                else:
                    logging.warning(
                        f"Unhandled contrastive method for embedding collection: {self.config.contrastive_method}"
                    )
                    # Fallback to encoder output if available, though not ideal for all methods
                    if hasattr(self.model, "encoder"):
                        all_h1.append(self.model.encoder(x1).detach())
                        all_h2.append(self.model.encoder(x2).detach())
                    else:
                        logging.warning(
                            "Could not collect embeddings for accuracy calculation."
                        )

                all_labels.append(labels.detach())

        avg_loss = total_loss / len(data_loader)

        # Only compute accuracy if embeddings were collected
        accuracy = 0.0
        if len(all_h1) > 0:
            all_h1 = torch.cat(all_h1)
            all_h2 = torch.cat(all_h2)
            all_labels = torch.cat(all_labels)

            if is_training:
                self.best_threshold = self._find_best_threshold(
                    all_h1, all_h2, all_labels
                )

            accuracy = self._compute_accuracy(
                all_h1, all_h2, all_labels, self.best_threshold
            )
        else:
            logging.warning("Could not collect embeddings for accuracy calculation.")

        return avg_loss, accuracy

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Runs a single training epoch."""
        return self._run_epoch(train_loader, is_training=True)

    def evaluate_model(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Evaluates the model on the validation set."""
        return self._run_epoch(val_loader, is_training=False)

    @staticmethod
    def _find_best_threshold(
        h1: torch.Tensor, h2: torch.Tensor, labels: torch.Tensor
    ) -> float:
        """Finds the best cosine similarity threshold for classification."""
        similarities = F.cosine_similarity(h1, h2).cpu().numpy()
        true_labels = torch.argmax(labels, dim=1).cpu().numpy()

        best_acc = 0
        best_thresh = 0
        for threshold in np.arange(0, 1, 0.01):
            preds = (similarities > threshold).astype(int)
            acc = balanced_accuracy_score(true_labels, preds)
            if acc > best_acc:
                best_acc = acc
                best_thresh = threshold
        return best_thresh

    @staticmethod
    def _compute_accuracy(
        h1: torch.Tensor, h2: torch.Tensor, labels: torch.Tensor, threshold: float
    ) -> float:
        """Computes balanced accuracy based on cosine similarity."""
        true_labels = torch.argmax(labels, dim=1).cpu().numpy()
        similarity = F.cosine_similarity(h1, h2).cpu().numpy()
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
    threshold: float,
) -> None:
    """Visualizes cosine similarities and decision thresholds for each batch."""
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


def plot_runs_metrics(
    all_runs_metrics: List[Dict], encoder_type: str, save_path: str
) -> None:
    """Creates box plots of metrics across multiple runs."""
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
    config: ContrastiveConfig,
    run_id: int,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    base_model: SimCLRModel,
    loss_fn: Optional[nn.Module] = None,
) -> Tuple[SimCLRModel, Dict, float]:
    """Executes a single training run for a given fold."""
    logging.info(f"Starting training for fold {run_id + 1}/{config.num_runs}")
    model = copy.deepcopy(base_model).to(device)
    trainer = ContrastiveTrainer(
        model=model, loss_fn=loss_fn, config=config, device=device
    )
    best_val_acc = 0.0
    best_metrics = {}
    patience_counter = 0
    best_threshold = 0.5

    # Initialize best_model_state_path to None
    best_model_state_path = None

    for epoch in range(config.num_epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate_model(val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_threshold = trainer.best_threshold
            best_metrics = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "epoch": epoch,
            }
            # Save the model state only if it's the best so far
            if config.trial_number is not None:
                best_model_state_path = f"model_{config.encoder_type}_run_{run_id}_trial_{config.trial_number}.pth"
            else:
                best_model_state_path = f"model_{config.encoder_type}_run_{run_id}.pth"
            torch.save(model.state_dict(), best_model_state_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            logging.info(f"Fold {run_id+1}, Epoch {epoch+1}: Val Acc: {val_acc:.2f}%")

    # Load the best model state if it was saved, otherwise return the last state
    if best_model_state_path and os.path.exists(best_model_state_path):
        model.load_state_dict(torch.load(best_model_state_path))
        # Clean up the temporary model file
        os.remove(best_model_state_path)
    else:
        logging.warning(
            f"No best model saved for run {run_id}. Returning model from last epoch."
        )

    return model, best_metrics, best_threshold


def main(config: ContrastiveConfig) -> Dict:
    """
    Orchestrates the training, validation, and testing of contrastive learning models.

    This function implements a robust evaluation methodology by first splitting the data
    into a training/validation set and a held-out test set. It then performs
    k-fold cross-validation on the training/validation set to find the best model,
    which is finally evaluated on the test set for an unbiased performance estimate.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load and preprocess data
    data_config = DataConfig(
        batch_size=config.batch_size,
        data_path="/Users/woodj/Desktop/fishy-business/data/REIMS.xlsx",   
    )
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data(data_config)
    filtered_data = preprocessor.filter_data(data, data_config.dataset_name)
    features = filtered_data.drop("m/z", axis=1).to_numpy()
    labels = preprocessor.encode_labels(filtered_data, data_config.dataset_name)
    groups = preprocessor.extract_groups(filtered_data)

    # --- Split into (Train + Val) and Test sets ---
    sgkf_test_split = StratifiedGroupKFold(n_splits=2)  # 80/20 split
    train_val_indices, test_indices = next(sgkf_test_split.split(features, np.argmax(labels, axis=1), groups=groups))

    X_train_val, X_test = features[train_val_indices], features[test_indices]
    y_train_val, y_test = labels[train_val_indices], labels[test_indices]
    groups_train_val = groups[train_val_indices]

    logging.info(f"Data split: {len(X_train_val)} train/val samples, {len(X_test)} test samples.")

    # --- Cross-validation on the (Train + Val) set ---
    k_folds = config.num_runs
    sgkf_cv = StratifiedGroupKFold(n_splits=k_folds)
    base_model, loss_fn = create_contrastive_model(config)

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    all_fold_metrics = []
    best_overall_model = None
    best_overall_val_acc = float("-inf")

    for fold, (train_index, val_index) in enumerate(sgkf_cv.split(X_train_val, np.argmax(y_train_val, axis=1), groups=groups_train_val)):
        logging.info(f"--- Starting Fold {fold + 1}/{k_folds} ---")
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]

        train_dataset = SiameseDataset(X_train, y_train)
        val_dataset = SiameseDataset(X_val, y_val)
        
        # ... (data loader setup as before)
        train_sampler = BalancedBatchSampler(train_dataset.pair_labels, config.batch_size)
        val_sampler = BalancedBatchSampler(val_dataset.pair_labels, config.batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)

        model, metrics, threshold = run_single_training(config, fold, device, train_loader, val_loader, base_model, loss_fn)

        if metrics:
            all_fold_metrics.append(metrics)
            if metrics["val_accuracy"] > best_overall_val_acc:
                best_overall_val_acc = metrics["val_accuracy"]
                best_overall_model = copy.deepcopy(model)
                torch.save(best_overall_model.state_dict(), f"best_model_{config.contrastive_method}_{config.encoder_type}_overall.pth")

    # --- Final Evaluation on the Test Set ---
    test_loss, test_accuracy = 0.0, 0.0
    if best_overall_model:
        test_dataset = SiameseDataset(X_test, y_test)
        if len(test_dataset) > 0:
            test_sampler = BalancedBatchSampler(test_dataset.pair_labels, config.batch_size)
            if len(test_sampler) > 0:
                test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)
                trainer = ContrastiveTrainer(best_overall_model, loss_fn, config, device)
                test_loss, test_accuracy = trainer.evaluate_model(test_loader)
                logging.info(f"Final Test Set Evaluation - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

    # --- Save results ---
    if all_fold_metrics:
        stats = {
            key: {"mean": np.mean([m[key] for m in all_fold_metrics]), "std": np.std([m[key] for m in all_fold_metrics])}
            for key in all_fold_metrics[0]
        }
        stats['test_loss'] = test_loss
        stats['test_accuracy'] = test_accuracy
        
        logging.info(f"Cross-validation statistics for {config.encoder_type}: {stats}")
        with open(f"results/stats_{config.contrastive_method}_{config.encoder_type}.json", "w") as f:
            json.dump({"config": asdict(config), "stats": stats, "folds": all_fold_metrics}, f, indent=4)
        return stats
    else:
        logging.warning("No folds completed successfully to aggregate metrics.")
        return {}


if __name__ == "__main__":
    """Entry point for the script."""
    parser = argparse.ArgumentParser(description="Train SimCLR models.")
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="transformer",
        choices=ENCODER_REGISTRY.keys(),
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Number of folds for cross-validation (k).",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3.5e-5)
    parser.add_argument(
        "--contrastive_method",
        type=str,
        default="simclr",
        choices=CONTRASTIVE_MODEL_REGISTRY.keys(),
        help="Contrastive learning method to use (e.g., simclr, moco, byol)",
    )
    # Add other config arguments as needed
    args = parser.parse_args()

    config = ContrastiveConfig(
        encoder_type=args.encoder_type,
        num_runs=args.num_runs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        contrastive_method=args.contrastive_method,
    )
    main(config)
