# -*- coding: utf-8 -*-
"""
Contrastive learning experiments with comprehensive pair-wise similarity metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from fishy._core.config import TrainingConfig
from fishy._core.utils import RunContext, get_device
from fishy.data.module import create_data_module, make_pairwise_test_split
from fishy._core.constants import DatasetName
from fishy.data.datasets import SiameseDataset, BalancedBatchSampler
from fishy._core.config_loader import load_config
from fishy._core.factory import create_model, get_model_class

from fishy.models.contrastive.simclr import SimCLRModel, SimCLRLoss
from fishy.models.contrastive.simsiam import SimSiamModel, SimSiamLoss
from fishy.models.contrastive.moco import MoCoModel, MoCoLoss
from fishy.models.contrastive.byol import BYOLModel, BYOLLoss
from fishy.models.contrastive.barlow_twins import BarlowTwinsModel, BarlowTwinsLoss

logger = logging.getLogger(__name__)


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive learning."""

    contrastive_method: str = "simclr"
    dataset: str = "species"
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    file_path: Optional[str] = None
    encoder_type: str = "dense"
    embedding_dim: int = 128
    projection_dim: int = 128
    temperature: float = 0.55
    moco_k: int = 4096
    moco_m: float = 0.999
    moco_t: float = 0.07
    byol_m: float = 0.996
    barlow_lambda: float = 5e-3
    wandb_project: Optional[str] = "fishy-business"
    wandb_entity: Optional[str] = "victoria-university-of-wellington"
    wandb_log: bool = False
    run: int = 0  # seed / run index for reproducible splits


class ContrastiveTrainer:
    """Trainer focused on Pair-wise Similarity Metrics for self-supervised learning."""

    def __init__(
        self,
        config: ContrastiveConfig,
        wandb_run: Optional[Any] = None,
        ctx: Optional[RunContext] = None,
    ) -> None:
        self.config = config
        self.wandb_run = wandb_run
        self.ctx = (
            ctx
            if ctx
            else RunContext(
                dataset=config.dataset,
                method="contrastive",
                model_name=config.contrastive_method,
                wandb_run=self.wandb_run,
            )
        )
        self.logger = self.ctx.logger
        self.device = get_device()
        self.metrics = {}
        self.best_threshold = 0.5

    def setup(self) -> None:
        self.data_module = create_data_module(
            file_path=self.config.file_path,
            dataset_name=self.config.dataset,
            batch_size=self.config.batch_size,
        )
        self.data_module.setup()
        self.input_dim = self.data_module.get_input_dim()

        t_cfg = TrainingConfig(
            model=self.config.encoder_type,
            hidden_dim=self.config.embedding_dim,
            num_layers=4,
            num_heads=4,
        )
        encoder = create_model(t_cfg, self.input_dim, self.config.embedding_dim).to(
            self.device
        )

        method = self.config.contrastive_method.lower()
        contrastive_cfg = load_config("models")["contrastive_models"]
        info = contrastive_cfg[method]
        model_class = get_model_class(info["model"])
        loss_class = get_model_class(info["loss"])

        self.model = model_class(
            backbone=encoder,
            embedding_dim=self.config.embedding_dim,
            projection_dim=self.config.projection_dim,
            dropout=0.1,
        ).to(self.device)

        if method == "simclr":
            self.criterion = loss_class(temperature=self.config.temperature)
        elif method == "barlow_twins":
            self.criterion = loss_class(lambda_param=self.config.barlow_lambda)
        else:
            self.criterion = loss_class()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        full_samples, full_labels = self.data_module.get_numpy_data()

        # Hold out a fixed test set — same split used by all method types for batch-detection
        if DatasetName.BATCH_DETECTION in self.config.dataset:
            train_X, self._test_X, train_y, self._test_y = make_pairwise_test_split(
                full_samples, full_labels, self.config.run
            )
        else:
            train_X, train_y = full_samples, full_labels
            self._test_X, self._test_y = None, None

        self._train_X, self._train_y = train_X, train_y
        self.siamese_dataset = SiameseDataset(train_X, train_y)
        self.train_sampler = BalancedBatchSampler(
            self.siamese_dataset.paired_labels, self.config.batch_size
        )
        self.train_loader = DataLoader(
            self.siamese_dataset, batch_sampler=self.train_sampler
        )

    def train(self) -> None:
        self.model.train()
        history = {"loss": [], "accuracy": []}
        from rich.progress import track

        for epoch in track(
            range(self.config.num_epochs), description="Contrastive Training..."
        ):
            total_loss = 0
            all_sims, all_labels = [], []

            for x1, x2, pair_labels, y1, y2 in self.train_loader:
                x1, x2 = x1.to(self.device), x2.to(self.device)
                pair_labels = pair_labels.to(self.device)

                # Filter for positive pairs if the method only supports them
                method = self.config.contrastive_method.lower()
                if method in ["byol", "simsiam", "barlow_twins"]:
                    mask = (pair_labels == 1).flatten()
                    if not mask.any():
                        continue
                    x1, x2 = x1[mask], x2[mask]
                    pair_labels = pair_labels[mask]

                self.optimizer.zero_grad()
                outputs = self.model(x1, x2)

                loss = (
                    self.criterion(*outputs)
                    if isinstance(outputs, tuple)
                    else self.criterion(outputs)
                )
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                # Dynamic Accuracy Check (Training set)
                with torch.no_grad():
                    z1, z2 = outputs[0], (
                        outputs[1] if isinstance(outputs, tuple) else (outputs, outputs)
                    )
                    sim = F.cosine_similarity(z1, z2).cpu().numpy()
                    all_sims.extend(sim)
                    all_labels.extend(pair_labels.cpu().numpy().flatten())

            avg_loss = (
                total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
            )
            history["loss"].append(avg_loss)

            # Epoch-level accuracy for progress bar
            if all_sims:
                self.best_threshold = self._optimize_threshold(
                    np.array(all_sims), np.array(all_labels)
                )
                acc = accuracy_score(
                    np.array(all_labels),
                    (np.array(all_sims) > self.best_threshold).astype(int),
                )
            else:
                acc = 0.0
            history["accuracy"].append(acc)

        self.metrics = {
            "history": history,
            "epoch_metrics": history,
            "val_loss": history["loss"][-1] if history["loss"] else 0,
        }

        # Comprehensive Pair-wise evaluation
        self.evaluate_pairwise_performance()

        self.ctx.save_results(self.metrics)

        # Add non-serializable objects for notebook/in-memory use after saving
        self.metrics["model"] = self.model
        self.metrics["data_module"] = self.data_module

        if self.ctx.wandb_run:
            self.log_contrastive_visualizations()

    def _optimize_threshold(
        self, similarities: np.ndarray, labels: np.ndarray
    ) -> float:
        """Finds threshold maximizing balanced accuracy."""
        best_acc, best_thresh = 0, 0.5
        for threshold in np.arange(0, 1, 0.05):
            acc = balanced_accuracy_score(
                labels, (similarities > threshold).astype(int)
            )
            if acc > best_acc:
                best_acc, best_thresh = acc, threshold
        return best_thresh

    def _calculate_pairwise_metrics(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        prefix: str,
    ):
        """Calculates all requested metrics for a set of similarities."""
        preds = (similarities > threshold).astype(int)
        self.metrics.update(
            {
                f"{prefix}_accuracy": accuracy_score(labels, preds),
                f"{prefix}_balanced_accuracy": balanced_accuracy_score(labels, preds),
                f"{prefix}_mae": mean_absolute_error(labels, similarities),
                f"{prefix}_mse": mean_squared_error(labels, similarities),
                f"{prefix}_precision": precision_score(labels, preds, zero_division=0),
                f"{prefix}_recall": recall_score(labels, preds, zero_division=0),
                f"{prefix}_f1": f1_score(labels, preds, zero_division=0),
            }
        )
        # Standardize for CLI display and align key names with other methods
        if prefix == "val":
            self.metrics["accuracy"] = self.metrics["val_accuracy"]
            self.metrics["balanced_accuracy"] = self.metrics["val_balanced_accuracy"]
            self.metrics["mae"] = self.metrics["val_mae"]
            self.metrics["mse"] = self.metrics["val_mse"]
            self.metrics["precision"] = self.metrics["val_precision"]
            self.metrics["recall"] = self.metrics["val_recall"]
            self.metrics["f1"] = self.metrics["val_f1"]
            # Mirror as test_ keys so all methods report the same wandb metric names
            self.metrics["test_balanced_accuracy"] = self.metrics[
                "val_balanced_accuracy"
            ]
            self.metrics["test_accuracy"] = self.metrics["val_accuracy"]
            self.metrics["test_f1"] = self.metrics["val_f1"]

    def evaluate_pairwise_performance(self) -> None:
        """Evaluates pair-wise similarity on the held-out test set."""
        self.logger.info("Calculating comprehensive pair-wise metrics...")
        self.model.eval()

        def get_sims(X, y):
            ds = SiameseDataset(X, y)
            ldr = DataLoader(ds, batch_size=self.config.batch_size)
            sims, lbls = [], []
            with torch.no_grad():
                for x1, x2, pair_lbl, _, _ in ldr:
                    outputs = self.model(x1.to(self.device), x2.to(self.device))
                    z1, z2 = outputs[0], (
                        outputs[1] if isinstance(outputs, tuple) else (outputs, outputs)
                    )
                    sims.extend(F.cosine_similarity(z1, z2).cpu().numpy())
                    lbls.extend(pair_lbl.cpu().numpy().flatten())
            return np.array(sims), np.array(lbls)

        # Use the pre-stored train/test split (set in setup()) for batch-detection;
        # fall back to a fresh split for other datasets.
        if self._test_X is not None:
            X_tr, y_tr = self._train_X, self._train_y
            X_val, y_val = self._test_X, self._test_y
        else:
            full_samples, full_labels = self.data_module.get_numpy_data()
            try:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    full_samples,
                    full_labels,
                    test_size=0.3,
                    stratify=np.argmax(full_labels, axis=1),
                    random_state=self.config.run,
                )
            except ValueError:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    full_samples,
                    full_labels,
                    test_size=0.3,
                    random_state=self.config.run,
                )

        train_sims, train_lbls = get_sims(X_tr, y_tr)
        val_sims, val_lbls = get_sims(X_val, y_val)

        self.best_threshold = self._optimize_threshold(train_sims, train_lbls)
        self._calculate_pairwise_metrics(
            train_sims, train_lbls, self.best_threshold, "train"
        )
        self._calculate_pairwise_metrics(val_sims, val_lbls, self.best_threshold, "val")

    def log_contrastive_visualizations(self) -> None:
        table = wandb.Table(columns=["id", "pair_1", "pair_2", "relationship"])
        import matplotlib.pyplot as plt

        for i in range(min(len(self.siamese_dataset), 20)):
            x1, x2, label, _, _ = self.siamese_dataset[i]
            plt.figure(figsize=(4, 3))
            plt.plot(x1.numpy())
            plt.axis("off")
            img1 = wandb.Image(plt)
            plt.close()
            plt.figure(figsize=(4, 3))
            plt.plot(x2.numpy())
            plt.axis("off")
            img2 = wandb.Image(plt)
            plt.close()
            table.add_data(
                i, img1, img2, "Same Class" if label == 1 else "Different Class"
            )
        self.ctx.wandb_run.log({"contrastive_pairs_sample": table}, commit=False)


def run_contrastive_experiment(config, wandb_run=None, ctx=None):
    started_wandb = False
    if wandb_run is None and config.wandb_log:
        started_wandb = True
    trainer = ContrastiveTrainer(config, wandb_run=wandb_run, ctx=ctx)
    try:
        trainer.setup()
        trainer.train()
        return trainer.metrics
    finally:
        if started_wandb and trainer.wandb_run:
            trainer.wandb_run.finish()
