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
from sklearn.cluster import AgglomerativeClustering

from fishy._core.config import TrainingConfig
from fishy._core.utils import RunContext, get_device
from fishy.data.module import create_data_module, make_pairwise_test_split, make_all_pairwise_folds, make_group_pairwise_folds
from fishy._core.constants import DatasetName
from fishy.data.datasets import SiameseDataset, BalancedBatchSampler, CustomDataset, PrecomputedPairDataset
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
    random_projection: bool = False
    quantize: bool = False
    turbo_quant: bool = False
    polar: bool = False
    normalize: bool = False
    snv: bool = False
    minmax: bool = False
    log_transform: bool = False
    savgol: bool = False
    use_groups: bool = False
    # New augmentation params for contrastive on-the-fly
    data_augmentation: bool = True
    noise_level: float = 0.05
    shift_range: float = 0.05
    scale_range: float = 0.1
    
    # Transitive / Clustering parameters
    use_transitive_clustering: bool = False
    clustering_threshold: float = 0.7  # Initial threshold
    learned_threshold: bool = False   # If true, the threshold is optimized
    clustering_step: int = 5          # Re-cluster every N epochs


class SupervisedContrastiveDataset(torch.utils.data.Dataset):
    """
    Returns pairs of different physical samples from the same batch
    with on-the-fly augmentations for both.
    """

    def __init__(
        self, samples: torch.Tensor, labels: torch.Tensor, augmenter: DataAugmenter
    ):
        self.samples = samples
        self.labels = labels.argmax(dim=1) if labels.dim() > 1 else labels
        self.augmenter = augmenter
        self.unique_labels = torch.unique(self.labels)

        # Pre-group indices by label for fast sampling
        self.label_to_idx = {
            int(l): torch.where(self.labels == l)[0] for l in self.unique_labels
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label = int(self.labels[idx])
        # Pick another sample from the same batch (class)
        pos_idxs = self.label_to_idx[label]
        if len(pos_idxs) > 1:
            other_idx = pos_idxs[torch.randint(0, len(pos_idxs), (1,)).item()]
        else:
            other_idx = idx  # Fallback to same sample if batch has only 1 sample

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
        encoder = create_model(t_cfg, self.input_dim, self.config.embedding_dim).to(
            self.device
        )

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
        
        augmenter = None
        if self.config.data_augmentation:
            aug_cfg = AugmentationConfig(
                enabled=True,
                num_augmentations=0,
                noise_enabled=True,
                noise_level=self.config.noise_level,
                shift_enabled=self.config.shift_range > 0,
                shift_range=self.config.shift_range,
                scale_enabled=self.config.scale_range > 0,
                scale_range=self.config.scale_range,
                normalize=self.config.normalize,
            )
            augmenter = DataAugmenter(aug_cfg)

        if DatasetName.BATCH_DETECTION in self.config.dataset:
            if self.config.use_groups:
                # Group-level split: each batch is kept whole in train or val.
                # Pairs formed within each split — no sample-level leakage.
                group_folds = make_group_pairwise_folds(
                    full_samples, full_labels,
                    n_splits=self.config.k_folds,
                    run_id=self.config.run,
                )
                for fold, (X1_tr, X2_tr, labels_tr, X1_val, X2_val, labels_val) in enumerate(group_folds):
                    self.logger.info(f"--- Contrastive Fold {fold+1}/{self.config.k_folds} (group split) ---")
                    model, criterion, optimizer = self._create_fresh_model()
                    train_ds = PrecomputedPairDataset(
                        X1_tr, X2_tr, labels_tr, augmenter=augmenter, train=True
                    )
                    sampler = BalancedBatchSampler(train_ds.paired_labels, self.config.batch_size)
                    train_ldr = DataLoader(train_ds, batch_sampler=sampler)
                    history = self._train_fold(model, criterion, optimizer, train_ldr)
                    fold_metrics = self._evaluate_precomputed_fold(
                        model, X1_tr, X2_tr, labels_tr, X1_val, X2_val, labels_val,
                    )
                    fold_metrics["history"] = history
                    self.all_fold_metrics.append(fold_metrics)
                    self.model = model
            else:
                # Default: pair-level K-fold (all C(N,2) pairs, split on pairs).
                # Same fold indices as deep and classic trainers for fair comparison.
                X1_all, X2_all, pair_labels_all, folds = make_all_pairwise_folds(
                    full_samples, full_labels,
                    n_splits=self.config.k_folds,
                    run_id=self.config.run,
                )
                for fold, (tr_idx, val_idx) in enumerate(folds):
                    self.logger.info(f"--- Contrastive Fold {fold+1}/{self.config.k_folds} ---")
                    model, criterion, optimizer = self._create_fresh_model()
                    train_ds = PrecomputedPairDataset(
                        X1_all[tr_idx], X2_all[tr_idx], pair_labels_all[tr_idx],
                        augmenter=augmenter, train=True
                    )
                    sampler = BalancedBatchSampler(train_ds.paired_labels, self.config.batch_size)
                    train_ldr = DataLoader(train_ds, batch_sampler=sampler)
                    history = self._train_fold(model, criterion, optimizer, train_ldr)
                    fold_metrics = self._evaluate_precomputed_fold(
                        model,
                        X1_all[tr_idx], X2_all[tr_idx], pair_labels_all[tr_idx],
                        X1_all[val_idx], X2_all[val_idx], pair_labels_all[val_idx],
                    )
                    fold_metrics["history"] = history
                    self.all_fold_metrics.append(fold_metrics)
                    self.model = model

        else:
            # Non-batch-detection: standard sample-level K-fold
            train_X, train_y = full_samples, full_labels
            try:
                strat_labels = (
                    np.argmax(train_y, axis=1) if train_y.ndim > 1 else train_y.flatten()
                )
                unique_labels, counts = np.unique(strat_labels, return_counts=True)
                if np.min(counts) < self.config.k_folds:
                    raise ValueError("Too few members for stratification")
                skf = StratifiedKFold(
                    n_splits=self.config.k_folds, shuffle=True, random_state=self.config.run
                )
                folds = list(skf.split(train_X, strat_labels))
            except (ValueError, TypeError):
                from sklearn.model_selection import KFold
                kf = KFold(
                    n_splits=self.config.k_folds, shuffle=True, random_state=self.config.run
                )
                folds = list(kf.split(train_X))

            for fold, (tr_idx, val_idx) in enumerate(folds):
                self.logger.info(f"--- Contrastive Fold {fold+1}/{self.config.k_folds} ---")
                X_tr_fold, y_tr_fold = train_X[tr_idx], train_y[tr_idx]
                X_val_fold, y_val_fold = train_X[val_idx], train_y[val_idx]

                model, criterion, optimizer = self._create_fresh_model()
                siamese_ds = SiameseDataset(X_tr_fold, y_tr_fold)
                sampler = BalancedBatchSampler(siamese_ds.paired_labels, self.config.batch_size)
                train_ldr = DataLoader(siamese_ds, batch_sampler=sampler)

                history = self._train_fold(model, criterion, optimizer, train_ldr)
                fold_metrics = self._evaluate_pairwise_fold(
                    model, X_tr_fold, y_tr_fold, X_val_fold, y_val_fold
                )
                fold_metrics["history"] = history
                self.all_fold_metrics.append(fold_metrics)
                self.model = model

        self._aggregate_metrics()
        self.ctx.save_results(self.metrics)
        self.metrics["model"] = self.model
        self.metrics["data_module"] = self.data_module

        if self.ctx.wandb_run:
            self.log_contrastive_visualizations()

    def _train_fold(self, model, criterion, optimizer, loader) -> Dict[str, List]:
        model.train()
        history = {"loss": [], "accuracy": []}
        
        # Keep track of current transitive assignments
        sample_to_cluster = {}
        
        # Extract unique samples for efficient clustering
        full_x = None
        if self.config.use_transitive_clustering:
            if hasattr(loader.dataset, "samples"):
                full_x = loader.dataset.samples.to(self.device)
            elif hasattr(loader.dataset, "X1"):
                # Efficiently find unique physical samples from the pair arrays
                all_samples = torch.cat([loader.dataset.X1, loader.dataset.X2], dim=0)
                # Use numpy for unique because it's faster for high-dim vectors than torch.unique
                unique_np = np.unique(all_samples.numpy(), axis=0)
                full_x = torch.from_numpy(unique_np).to(self.device)
                self.logger.info(f"Clustering setup: {len(full_x)} unique physical samples discovered.")

        for epoch in range(self.config.num_epochs):
            # 1. Update Transitive Assignments (Delayed start for stability)
            if self.config.use_transitive_clustering and epoch >= 10 and epoch % self.config.clustering_step == 0:
                with torch.no_grad():
                    model.eval()
                    embeddings = F.normalize(model.encoder(full_x), dim=1)
                    
                    # Adaptive strict threshold (top 0.5% of similarities)
                    sim_matrix = torch.matmul(embeddings, embeddings.T)
                    triu_idx = torch.triu_indices(len(embeddings), len(embeddings), offset=1)
                    flat_sims = sim_matrix[triu_idx[0], triu_idx[1]]
                    
                    k = max(1, int(len(flat_sims) * 0.005)) 
                    current_threshold = torch.topk(flat_sims, k)[0][-1].item()
                    current_threshold = max(0.95, current_threshold)
                    
                    clustering = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=max(0.0001, 1.0 - current_threshold),
                        metric="cosine",
                        linkage="average"
                    )
                    clusters = clustering.fit_predict(embeddings.cpu().numpy())
                    n_clusters = len(np.unique(clusters))
                    
                    # Positive Rate Guard: If too many positives are found, clustering likely collapsed
                    pos_rate = (1.0 / n_clusters) if n_clusters > 0 else 1.0
                    if 1 < n_clusters < len(full_x) and pos_rate < 0.2:
                        # Build efficient lookup map (using sample hash as key)
                        # We use the index in full_x as a proxy for the sample identity
                        sample_to_cluster = {i: c for i, c in enumerate(clusters)}
                        self._cluster_embeddings = embeddings.clone()
                        self.logger.info(f"Epoch {epoch}: Refined {n_clusters} transitive clusters (Thr: {current_threshold:.4f}).")
                    else:
                        self.logger.info(f"Epoch {epoch}: Clustering ignored (rate: {pos_rate:.2f}) to prevent collapse.")
                    
                    model.train()

            total_loss = 0
            for batch in loader:
                # Handle both Siamese and Precomputed datasets
                if len(batch) == 5:
                    x1, x2, pair_labels, _, _ = batch
                else:
                    x1, x2, pair_labels = batch[0], batch[1], batch[2]

                x1, x2 = x1.to(self.device), x2.to(self.device)
                pair_labels = pair_labels.to(self.device).float().view(-1, 1)

                optimizer.zero_grad()
                outputs = model(x1, x2)

                # 2. Apply Transitive Consistency (Soft approach)
                transitive_loss = 0.0
                if sample_to_cluster:
                    with torch.no_grad():
                        z1_enc = F.normalize(model.encoder(x1), dim=1)
                        z2_enc = F.normalize(model.encoder(x2), dim=1)
                        idx1 = torch.matmul(z1_enc, self._cluster_embeddings.T).argmax(dim=1)
                        idx2 = torch.matmul(z2_enc, self._cluster_embeddings.T).argmax(dim=1)
                        c1 = torch.tensor([sample_to_cluster[i.item()] for i in idx1], device=self.device)
                        c2 = torch.tensor([sample_to_cluster[i.item()] for i in idx2], device=self.device)
                        # Identify pairs that are in the same cluster but NOT ground-truth positives
                        is_transitive = (c1 == c2).float().view(-1, 1)
                        transitive_mask = (is_transitive > 0) & (pair_labels < 0.5)

                    if transitive_mask.any():
                        # Gently pull these transitive matches together
                        # We use the projection head outputs for the loss
                        z1_proj, z2_proj = outputs[0][transitive_mask.flatten()], outputs[1][transitive_mask.flatten()]
                        # MSE loss toward 1.0 similarity (gentle pull)
                        transitive_loss = 0.1 * F.mse_loss(F.cosine_similarity(z1_proj, z2_proj), torch.ones_like(F.cosine_similarity(z1_proj, z2_proj)))

                method = self.config.contrastive_method.lower()
                if method == "simclr":
                    loss = criterion(*outputs, pair_labels=pair_labels) + transitive_loss
                else:
                    loss = criterion(*outputs) + transitive_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            history["loss"].append(total_loss / len(loader) if len(loader) > 0 else 0)
        return history

    def _sims_from_loader(self, model, ldr) -> Tuple[np.ndarray, np.ndarray]:
        """Run model over a DataLoader of pairs, return (cosine_sims, pair_labels)."""
        sims, lbls = [], []
        with torch.no_grad():
            for x1, x2, pair_lbl, _, _ in ldr:
                outputs = model(x1.to(self.device), x2.to(self.device))
                z1, z2 = outputs[0], outputs[1]
                sims.extend(F.cosine_similarity(z1, z2).cpu().numpy())
                lbls.extend(pair_lbl.cpu().numpy().flatten())
        return np.array(sims), np.array(lbls)

    def _evaluate_precomputed_fold(
        self,
        model,
        X1_tr, X2_tr, y_tr,
        X1_val, X2_val, y_val,
    ) -> Dict[str, float]:
        """Evaluate on pre-indexed pair arrays (batch-detection pair-level CV)."""
        model.eval()
        tr_ldr = DataLoader(
            PrecomputedPairDataset(X1_tr, X2_tr, y_tr),
            batch_size=self.config.batch_size,
        )
        val_ldr = DataLoader(
            PrecomputedPairDataset(X1_val, X2_val, y_val),
            batch_size=self.config.batch_size,
        )
        tr_sims, tr_lbls = self._sims_from_loader(model, tr_ldr)
        val_sims, val_lbls = self._sims_from_loader(model, val_ldr)
        thresh = self._optimize_threshold(tr_sims, tr_lbls)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            res = {
                **self._calc_raw_metrics(tr_sims, tr_lbls, thresh, "train"),
                **self._calc_raw_metrics(val_sims, val_lbls, thresh, "val"),
                "threshold": thresh,
            }
        return res

    def _evaluate_pairwise_fold(
        self, model, X_tr, y_tr, X_val, y_val
    ) -> Dict[str, float]:
        model.eval()
        tr_ldr = DataLoader(SiameseDataset(X_tr, y_tr), batch_size=self.config.batch_size)
        val_ldr = DataLoader(SiameseDataset(X_val, y_val), batch_size=self.config.batch_size)
        tr_sims, tr_lbls = self._sims_from_loader(model, tr_ldr)
        val_sims, val_lbls = self._sims_from_loader(model, val_ldr)
        thresh = self._optimize_threshold(tr_sims, tr_lbls)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            res = {
                **self._calc_raw_metrics(tr_sims, tr_lbls, thresh, "train"),
                **self._calc_raw_metrics(val_sims, val_lbls, thresh, "val"),
                "threshold": thresh,
            }
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
            "accuracy",
            "balanced_accuracy",
            "f1",
            "mae",
            "mse",
            "precision",
            "recall",
        ]
        self.metrics = {}
        for k in agg_keys:
            # Average Validation Metrics
            val_vals = [m[f"val_{k}"] for m in self.all_fold_metrics]
            self.metrics[f"val_{k}"] = float(np.mean(val_vals))
            self.metrics[k] = self.metrics[f"val_{k}"]  # Root key for display

            # Average Training Metrics
            tr_vals = [m[f"train_{k}"] for m in self.all_fold_metrics]
            self.metrics[f"train_{k}"] = float(np.mean(tr_vals))

        self.best_threshold = float(
            np.mean([m["threshold"] for m in self.all_fold_metrics])
        )
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
            bal_acc = (
                balanced_accuracy_score(lbls, preds)
                if unique_labels > 1
                else accuracy_score(lbls, preds)
            )

            self.metrics.update(
                {
                    "test_accuracy": accuracy_score(lbls, preds),
                    "test_balanced_accuracy": bal_acc,
                    "test_f1": f1_score(lbls, preds, zero_division=0),
                }
            )

    def _optimize_threshold(
        self, similarities: np.ndarray, labels: np.ndarray
    ) -> float:
        if len(np.unique(labels)) < 2:
            return 0.5
        best_acc, best_thresh = 0, 0.5
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            for threshold in np.arange(0, 1, 0.05):
                acc = balanced_accuracy_score(
                    labels, (similarities > threshold).astype(int)
                )
                if acc > best_acc:
                    best_acc, best_thresh = acc, threshold
        return best_thresh

    def log_contrastive_visualizations(self) -> None:
        full_X, full_y = self.data_module.get_numpy_data()
        vis_ds = SiameseDataset(full_X, full_y)
        table = wandb.Table(columns=["id", "pair_1", "pair_2", "relationship"])
        import matplotlib.pyplot as plt

        for i in range(min(len(vis_ds), 20)):
            x1, x2, label, _, _ = vis_ds[i]
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
