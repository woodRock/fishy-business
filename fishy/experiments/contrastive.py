# -*- coding: utf-8 -*-
"""
Contrastive learning experiments with comprehensive pair-wise similarity metrics.
Integrated with 3-Fold Stratified Cross-Validation for fair benchmark alignment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
import warnings
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
from sklearn.model_selection import train_test_split, StratifiedKFold

from fishy._core.config import TrainingConfig
from fishy._core.utils import RunContext, get_device
from fishy.data.module import create_data_module, make_pairwise_test_split
from fishy._core.constants import DatasetName
from fishy.data.datasets import SiameseDataset, BalancedBatchSampler, CustomDataset
from fishy.data.augmentation import DataAugmenter, AugmentationConfig
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
    k_folds: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    file_path: Optional[str] = None
    encoder_type: str = "dense"
    embedding_dim: int = 128
    projection_dim: int = 128
    temperature: float = 0.55
    learnable_temperature: bool = True
    use_stop_grad: bool = False
    moco_k: int = 4096
    moco_m: float = 0.999
    moco_t: float = 0.07
    byol_m: float = 0.996
    barlow_lambda: float = 5e-3
    wandb_project: Optional[str] = "fishy-business"
    wandb_entity: Optional[str] = "victoria-university-of-wellington"
    wandb_log: bool = False
    run: int = 0  # seed / run index for reproducible splits
    random_projection: bool = (False,)
    quantize: bool = (False,)
    turbo_quant: bool = (False,)
    polar: bool = (False,)
    normalize: bool = (False,)
    snv: bool = (False,)
    minmax: bool = (False,)
    log_transform: bool = (False,)
    savgol: bool = (False,)
    # New augmentation params for contrastive on-the-fly
    data_augmentation: bool = True
    noise_level: float = 0.05
    shift_range: float = 0.05
    scale_range: float = 0.1


class SupervisedContrastiveDataset(torch.utils.data.Dataset):
    """
    Returns pairs of different physical samples from the same batch
    with on-the-fly augmentations for both.
    """
    def __init__(self, samples: torch.Tensor, labels: torch.Tensor, augmenter: DataAugmenter):
        self.samples = samples
        self.labels = labels.argmax(dim=1) if labels.dim() > 1 else labels
        self.augmenter = augmenter
        self.unique_labels = torch.unique(self.labels)
        
        # Pre-group indices by label for fast sampling
        self.label_to_idx = {int(l): torch.where(self.labels == l)[0] for l in self.unique_labels}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label = int(self.labels[idx])
        # Pick another sample from the same batch (class)
        pos_idxs = self.label_to_idx[label]
        if len(pos_idxs) > 1:
            other_idx = pos_idxs[torch.randint(0, len(pos_idxs), (1,)).item()]
        else:
            other_idx = idx # Fallback to same sample if batch has only 1 sample
            
        x1 = self.samples[idx].unsqueeze(0)
        x2 = self.samples[other_idx].unsqueeze(0)
        
        # Apply independent random augmentations to both physical samples
        x1_aug = self.augmenter._apply_augmentations_to_batch(x1).squeeze(0)
        x2_aug = self.augmenter._apply_augmentations_to_batch(x2).squeeze(0)
        
        return x1_aug, x2_aug, torch.tensor(label)


class ContrastiveTrainer:
    """Trainer focused on Pair-wise Similarity Metrics with robust K-Fold support."""

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
        self.all_fold_metrics = []

    def setup(self) -> None:
        self.data_module = create_data_module(
            dataset_name=self.config.dataset,
            file_path=self.config.file_path,
            batch_size=self.config.batch_size,
            random_projection=self.config.random_projection,
            quantize=self.config.quantize,
            turbo_quant=self.config.turbo_quant,
            polar=self.config.polar,
            normalize=self.config.normalize,
            snv=self.config.snv,
            minmax=self.config.minmax,
            log_transform=self.config.log_transform,
            savgol=self.config.savgol,
            run_id=self.config.run,
        )
        self.data_module.setup()
        self.input_dim = self.data_module.get_input_dim()

    def _create_fresh_model(self) -> Tuple[nn.Module, nn.Module, optim.Optimizer]:
        """Creates a new model, criterion, and optimizer instance for each fold."""
        t_cfg = TrainingConfig(
            model=self.config.encoder_type,
            hidden_dim=self.config.embedding_dim,
            num_layers=4,
            num_heads=4,
        )
        encoder = create_model(t_cfg, self.input_dim, self.config.embedding_dim).to(self.device)

        method = self.config.contrastive_method.lower()
        contrastive_cfg = load_config("models")["contrastive_models"]
        info = contrastive_cfg[method]
        model_class = get_model_class(info["model"])
        loss_class = get_model_class(info["loss"])

        model = model_class(
            backbone=encoder,
            embedding_dim=self.config.embedding_dim,
            projection_dim=self.config.projection_dim,
            dropout=0.1,
            use_stop_grad=self.config.use_stop_grad if method == "simclr" else False,
        ).to(self.device)

        if method == "simclr":
            criterion = loss_class(temperature=self.config.temperature)
        elif method == "barlow_twins":
            criterion = loss_class(lambda_param=self.config.barlow_lambda)
        else:
            criterion = loss_class()

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        return model, criterion, optimizer

    def train(self) -> None:
        full_samples, full_labels = self.data_module.get_numpy_data()
        
        if DatasetName.BATCH_DETECTION in self.config.dataset:
            train_X, test_X, train_y, test_y = make_pairwise_test_split(
                full_samples, full_labels, self.config.run
            )
        else:
            train_X, test_X, train_y, test_y = full_samples, None, full_labels, None

        # Implementation of K-Fold with stratification fallback
        try:
            strat_labels = np.argmax(train_y, axis=1) if train_y.ndim > 1 else train_y.flatten()
            unique_labels, counts = np.unique(strat_labels, return_counts=True)
            if np.min(counts) < self.config.k_folds:
                raise ValueError("Too few members for stratification")
            skf = StratifiedKFold(n_splits=self.config.k_folds, shuffle=True, random_state=self.config.run)
            folds = list(skf.split(train_X, strat_labels))
        except (ValueError, TypeError):
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=self.config.k_folds, shuffle=True, random_state=self.config.run)
            folds = list(kf.split(train_X))

        for fold, (tr_idx, val_idx) in enumerate(folds):
            self.logger.info(f"--- Contrastive Fold {fold+1}/{self.config.k_folds} ---")
            
            X_tr_fold, y_tr_fold = train_X[tr_idx], train_y[tr_idx]
            X_val_fold, y_val_fold = train_X[val_idx], train_y[val_idx]
            
            model, criterion, optimizer = self._create_fresh_model()
            
            method = self.config.contrastive_method.lower()
            if method == "simclr" and self.config.data_augmentation:
                aug_cfg = AugmentationConfig(
                    enabled=True,
                    noise_enabled=True,
                    noise_level=self.config.noise_level,
                    shift_enabled=True,
                    shift_range=self.config.shift_range,
                    scale_enabled=True,
                    scale_range=self.config.scale_range,
                )
                augmenter = DataAugmenter(aug_cfg)
                train_ds = SupervisedContrastiveDataset(
                    torch.from_numpy(X_tr_fold).to(self.device),
                    torch.from_numpy(y_tr_fold).to(self.device),
                    augmenter
                )
                train_ldr = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
            else:
                siamese_ds = SiameseDataset(X_tr_fold, y_tr_fold)
                sampler = BalancedBatchSampler(siamese_ds.paired_labels, self.config.batch_size)
                train_ldr = DataLoader(siamese_ds, batch_sampler=sampler)

            history = self._train_fold(model, criterion, optimizer, train_ldr)
            fold_metrics = self._evaluate_pairwise_fold(model, X_tr_fold, y_tr_fold, X_val_fold, y_val_fold)
            fold_metrics["history"] = history
            self.all_fold_metrics.append(fold_metrics)
            self.model = model
        
        self._aggregate_metrics()
        
        if test_X is not None:
            self._evaluate_final_test(test_X, test_y)

        self.ctx.save_results(self.metrics)
        self.metrics["model"] = self.model
        self.metrics["data_module"] = self.data_module

        if self.ctx.wandb_run:
            self.log_contrastive_visualizations()

    def _train_fold(self, model, criterion, optimizer, loader) -> Dict[str, List]:
        model.train()
        history = {"loss": [], "accuracy": []}
        for epoch in range(self.config.num_epochs):
            total_loss = 0
            for batch in loader:
                if len(batch) == 5:
                    x1, x2, pair_labels, _, _ = batch
                    labels = None
                elif len(batch) == 3:
                    x1, x2, labels = batch
                    pair_labels = torch.ones(x1.size(0), 1).to(self.device)
                else:
                    x1, x2 = batch
                    pair_labels = torch.ones(x1.size(0), 1).to(self.device)
                    labels = None

                x1, x2 = x1.to(self.device), x2.to(self.device)
                pair_labels = pair_labels.to(self.device)

                method = self.config.contrastive_method.lower()
                if method in ["byol", "simsiam", "barlow_twins"]:
                    mask = (pair_labels == 1).flatten()
                    if not mask.any(): continue
                    x1, x2, pair_labels = x1[mask], x2[mask], pair_labels[mask]

                optimizer.zero_grad()
                outputs = model(x1, x2)
                
                if method == "simclr":
                    loss = criterion(*outputs, labels=labels)
                    # Update queue with keys (outputs[1] is momentum encoder output)
                    if hasattr(model, "_dequeue_and_enqueue"):
                        model._dequeue_and_enqueue(outputs[1])
                else:
                    loss = criterion(*outputs) if isinstance(outputs, tuple) else criterion(outputs)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            history["loss"].append(total_loss / len(loader) if len(loader) > 0 else 0)
        return history

    def _evaluate_pairwise_fold(self, model, X_tr, y_tr, X_val, y_val) -> Dict[str, float]:
        model.eval()
        def get_sims(X, y):
            ds = SiameseDataset(X, y)
            ldr = DataLoader(ds, batch_size=self.config.batch_size)
            sims, lbls = [], []
            with torch.no_grad():
                for x1, x2, pair_lbl, _, _ in ldr:
                    outputs = model(x1.to(self.device), x2.to(self.device))
                    # Standard Siamese uses z1, z2 directly. 
                    # MoCLR uses query and momentum-key.
                    z1, z2 = outputs[0], outputs[1]
                    sims.extend(F.cosine_similarity(z1, z2).cpu().numpy())
                    lbls.extend(pair_lbl.cpu().numpy().flatten())
            return np.array(sims), np.array(lbls)

        tr_sims, tr_lbls = get_sims(X_tr, y_tr)
        val_sims, val_lbls = get_sims(X_val, y_val)
        
        thresh = self._optimize_threshold(tr_sims, tr_lbls)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            tr_res = self._calc_raw_metrics(tr_sims, tr_lbls, thresh, "train")
            val_res = self._calc_raw_metrics(val_sims, val_lbls, thresh, "val")
            
            res = {**tr_res, **val_res}
            res["threshold"] = thresh
            return res

    def _calc_raw_metrics(self, sims, labels, thresh, prefix):
        preds = (sims > thresh).astype(int)
        unique_labels = len(np.unique(labels))
        
        if unique_labels < 2:
            bal_acc = accuracy_score(labels, preds)
        else:
            bal_acc = balanced_accuracy_score(labels, preds)
            
        return {
            f"{prefix}_accuracy": accuracy_score(labels, preds),
            f"{prefix}_balanced_accuracy": bal_acc,
            f"{prefix}_f1": f1_score(labels, preds, zero_division=0),
            f"{prefix}_mae": mean_absolute_error(labels, sims),
            f"{prefix}_mse": mean_squared_error(labels, sims),
            f"{prefix}_precision": precision_score(labels, preds, zero_division=0),
            f"{prefix}_recall": recall_score(labels, preds, zero_division=0),
        }

    def _aggregate_metrics(self) -> None:
        agg_keys = [
            "accuracy", "balanced_accuracy", "f1", "mae", "mse", "precision", "recall"
        ]
        self.metrics = {}
        for k in agg_keys:
            # Average Validation Metrics
            val_vals = [m[f"val_{k}"] for m in self.all_fold_metrics]
            self.metrics[f"val_{k}"] = float(np.mean(val_vals))
            self.metrics[k] = self.metrics[f"val_{k}"] # Root key for display
            
            # Average Training Metrics
            tr_vals = [m[f"train_{k}"] for m in self.all_fold_metrics]
            self.metrics[f"train_{k}"] = float(np.mean(tr_vals))

        self.best_threshold = float(np.mean([m["threshold"] for m in self.all_fold_metrics]))
        self.metrics["history"] = self.all_fold_metrics[-1]["history"]
        self.metrics["epoch_metrics"] = self.metrics["history"]
        self.metrics["folds"] = self.all_fold_metrics

    def _evaluate_final_test(self, X_te, y_te) -> None:
        """One-time evaluation on held-out samples."""
        self.model.eval()
        ds = SiameseDataset(X_te, y_te)
        ldr = DataLoader(ds, batch_size=self.config.batch_size)
        sims, lbls = [], []
        with torch.no_grad():
            for x1, x2, pair_lbl, _, _ in ldr:
                outputs = self.model(x1.to(self.device), x2.to(self.device))
                z1, z2 = outputs[0], outputs[1]
                sims.extend(F.cosine_similarity(z1, z2).cpu().numpy())
                lbls.extend(pair_lbl.cpu().numpy().flatten())
        
        sims, lbls = np.array(sims), np.array(lbls)
        thresh = self.best_threshold
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            preds = (sims > thresh).astype(int)
            unique_labels = len(np.unique(lbls))
            bal_acc = balanced_accuracy_score(lbls, preds) if unique_labels > 1 else accuracy_score(lbls, preds)
            
            self.metrics.update({
                "test_accuracy": accuracy_score(lbls, preds),
                "test_balanced_accuracy": bal_acc,
                "test_f1": f1_score(lbls, preds, zero_division=0)
            })

    def _optimize_threshold(self, similarities: np.ndarray, labels: np.ndarray) -> float:
        if len(np.unique(labels)) < 2: return 0.5
        best_acc, best_thresh = 0, 0.5
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            for threshold in np.arange(0, 1, 0.05):
                acc = balanced_accuracy_score(labels, (similarities > threshold).astype(int))
                if acc > best_acc: best_acc, best_thresh = acc, threshold
        return best_thresh

    def log_contrastive_visualizations(self) -> None:
        full_X, full_y = self.data_module.get_numpy_data()
        vis_ds = SiameseDataset(full_X, full_y)
        table = wandb.Table(columns=["id", "pair_1", "pair_2", "relationship"])
        import matplotlib.pyplot as plt
        for i in range(min(len(vis_ds), 20)):
            x1, x2, label, _, _ = vis_ds[i]
            plt.figure(figsize=(4, 3))
            plt.plot(x1.numpy()); plt.axis("off")
            img1 = wandb.Image(plt); plt.close()
            plt.figure(figsize=(4, 3))
            plt.plot(x2.numpy()); plt.axis("off")
            img2 = wandb.Image(plt); plt.close()
            table.add_data(i, img1, img2, "Same Class" if label == 1 else "Different Class")
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
