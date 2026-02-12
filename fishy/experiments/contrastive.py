# -*- coding: utf-8 -*-
"""
Contrastive learning experiments with restored balanced sampling and similarity evaluation.
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from fishy._core.config import TrainingConfig
from fishy._core.utils import RunContext, get_device
from fishy.data.module import create_data_module
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
    file_path: str = "data/REIMS.xlsx"
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


class ContrastiveTrainer:
    """Restored Contrastive Trainer with Balanced Sampling and Similarity Metrics."""

    def __init__(
        self,
        config: ContrastiveConfig,
        wandb_run: Optional[Any] = None,
        ctx: Optional[RunContext] = None,
    ) -> None:
        self.config = config
        self.wandb_run = wandb_run
        self.ctx = ctx if ctx else RunContext(dataset=config.dataset, method="contrastive", model_name=config.contrastive_method, wandb_run=self.wandb_run)
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
            hidden_dimension=self.config.embedding_dim,
            num_layers=4,
            num_heads=4,
        )
        encoder = create_model(t_cfg, self.input_dim, self.config.embedding_dim).to(self.device)
        
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

        if method == "simclr": self.criterion = loss_class(temperature=self.config.temperature)
        elif method == "barlow_twins": self.criterion = loss_class(lambda_param=self.config.barlow_lambda)
        else: self.criterion = loss_class()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        # Restored Balanced Sampling
        full_samples, full_labels = self.data_module.get_numpy_data()
        self.siamese_dataset = SiameseDataset(full_samples, full_labels)
        self.train_sampler = BalancedBatchSampler(self.siamese_dataset.paired_labels, self.config.batch_size)
        self.train_loader = DataLoader(self.siamese_dataset, batch_sampler=self.train_sampler)

    def train(self) -> None:
        self.model.train()
        history = {"loss": [], "accuracy": []}
        from rich.progress import track

        for epoch in track(range(self.config.num_epochs), description="Contrastive Training..."):
            total_loss = 0
            all_h1, all_h2, all_pair_labels = [], [], []
            
            for x1, x2, pair_labels, y1, y2 in self.train_loader:
                x1, x2 = x1.to(self.device), x2.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x1, x2)
                
                loss = self.criterion(*outputs) if isinstance(outputs, tuple) else self.criterion(outputs)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
                # Collect for threshold/accuracy (SimCLR style similarity)
                if self.config.contrastive_method == "simclr":
                    z1, z2 = outputs
                    all_h1.append(z1.detach()); all_h2.append(z2.detach())
                    all_pair_labels.append(pair_labels.detach())

            avg_loss = total_loss / len(self.train_loader)
            history["loss"].append(avg_loss)
            
            # Restored: Find best threshold and calculate pair-wise accuracy
            if all_h1:
                h1, h2 = torch.cat(all_h1), torch.cat(all_h2)
                lbls = torch.cat(all_pair_labels)
                self.best_threshold = self._find_best_threshold(h1, h2, lbls)
                acc = self._compute_pair_accuracy(h1, h2, lbls, self.best_threshold)
                history["accuracy"].append(acc)

        self.metrics = {
            "history": history,
            "val_loss": history["loss"][-1] if history["loss"] else 0,
            "train_accuracy": history["accuracy"][-1] if history["accuracy"] else 0
        }
        
        self.evaluate_classification()
        self.ctx.save_results(self.metrics)
        if self.ctx.wandb_run: self.log_contrastive_visualizations()

    def _find_best_threshold(self, h1, h2, labels) -> float:
        similarities = F.cosine_similarity(h1, h2).cpu().numpy()
        true_labels = labels.cpu().numpy().flatten()
        best_acc, best_thresh = 0, 0.5
        for threshold in np.arange(0, 1, 0.05):
            preds = (similarities > threshold).astype(int)
            acc = balanced_accuracy_score(true_labels, preds)
            if acc > best_acc: best_acc, best_thresh = acc, threshold
        return best_thresh

    def _compute_pair_accuracy(self, h1, h2, labels, threshold) -> float:
        similarities = F.cosine_similarity(h1, h2).cpu().numpy()
        true_labels = labels.cpu().numpy().flatten()
        preds = (similarities > threshold).astype(int)
        return balanced_accuracy_score(true_labels, preds)

    def evaluate_classification(self) -> None:
        """Evaluates learned representations via a linear probe."""
        self.logger.info("Evaluating representations via linear probe...")
        self.model.eval()
        X, y = self.data_module.get_numpy_data(labels_as_indices=True)
        embeddings = []
        with torch.no_grad():
            eval_loader = DataLoader(torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y)), batch_size=64)
            for batch_x, _ in eval_loader:
                backbone = self.model.encoder if hasattr(self.model, "encoder") else (self.model.online_encoder if hasattr(self.model, "online_encoder") else self.model)
                emb = backbone(batch_x.to(self.device))
                embeddings.append(emb.cpu().numpy())
        
        X_emb = np.concatenate(embeddings, axis=0)
        
        # Robust split for small datasets
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X_emb, y, test_size=0.3, stratify=y, random_state=42)
        except ValueError:
            # Fallback to non-stratified if stratification fails
            X_tr, X_te, y_tr, y_te = train_test_split(X_emb, y, test_size=0.5, random_state=42)
            
        clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
        tr_acc, te_acc = balanced_accuracy_score(y_tr, clf.predict(X_tr)), balanced_accuracy_score(y_te, clf.predict(X_te))
        self.metrics.update({"val_balanced_accuracy": te_acc, "train_balanced_accuracy": tr_acc, "accuracy": te_acc})

    def log_contrastive_visualizations(self) -> None:
        table = wandb.Table(columns=["id", "pair_1", "pair_2", "relationship"])
        import matplotlib.pyplot as plt
        for i in range(min(len(self.siamese_dataset), 20)):
            x1, x2, label, _, _ = self.siamese_dataset[i]
            plt.figure(figsize=(4, 3)); plt.plot(x1.numpy()); plt.axis("off"); img1 = wandb.Image(plt); plt.close()
            plt.figure(figsize=(4, 3)); plt.plot(x2.numpy()); plt.axis("off"); img2 = wandb.Image(plt); plt.close()
            table.add_data(i, img1, img2, "Same Class" if label == 1 else "Different Class")
        self.ctx.wandb_run.log({"contrastive_pairs_sample": table}, commit=False)


def run_contrastive_experiment(config, wandb_run=None, ctx=None):
    started_wandb = False
    if wandb_run is None and config.wandb_log: started_wandb = True
    trainer = ContrastiveTrainer(config, wandb_run=wandb_run, ctx=ctx)
    try:
        trainer.setup(); trainer.train()
        return trainer.metrics
    finally:
        if started_wandb and trainer.wandb_run: trainer.wandb_run.finish()
