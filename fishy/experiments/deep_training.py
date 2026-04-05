# -*- coding: utf-8 -*-
"""
Training orchestrator for the deep learning pipeline.
"""

import logging
from dataclasses import asdict
from typing import Dict, Optional, List, Any, Tuple
import wandb

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from fishy._core.config import TrainingConfig
from fishy._core.factory import create_model
from fishy._core.utils import RunContext, get_device
from fishy.experiments.pre_training import PreTrainingOrchestrator
from fishy.engine.trainer import DeepEngine
from fishy.data.module import create_data_module
from fishy.data.datasets import CustomDataset, SiameseDataset
from fishy.engine.losses import coral_loss, cumulative_link_loss


class ModelTrainer:
    """Orchestrates deep learning experiments."""

    def __init__(
        self,
        config: TrainingConfig,
        wandb_run: Optional[Any] = None,
        ctx: Optional[RunContext] = None,
    ):
        self.config = config
        self.wandb_run = wandb_run
        if self.wandb_run is None and self.config.wandb_log:
            import os
            os.environ["WANDB_START_METHOD"] = "thread"
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=asdict(self.config),
                reinit=True,
                group=f"{self.config.dataset}_{self.config.model}",
                job_type="training",
            )

        self.ctx = (
            ctx
            if ctx
            else RunContext(
                dataset=config.dataset,
                method="deep",
                model_name=config.model,
                wandb_run=self.wandb_run,
            )
        )
        self.logger = self.ctx.logger
        self.ctx.save_config(config)
        self.device = get_device()

        dataset_name = (
            "oil_regression"
            if (self.config.regression and self.config.dataset == "oil")
            else self.config.dataset
        )
        self.data_module = create_data_module(
            file_path=config.file_path,
            dataset_name=dataset_name,
            batch_size=config.batch_size,
            augmentation_config=config,
        )
        self.data_module.setup()
        self.n_features = self.data_module.get_input_dim()
        self.n_classes = self.data_module.get_num_classes()
        self.pre_train_orchestrator = PreTrainingOrchestrator(
            config=self.config,
            device=self.device,
            input_dim=self.n_features,
            ctx=self.ctx,
        )

    def pre_train(self) -> Optional[nn.Module]:
        self.logger.info("Evaluating pre-training phase")
        return self.pre_train_orchestrator.run_all(
            self.data_module.get_train_dataloader()
        )

    def train(
        self, pre_trained_model: Optional[nn.Module] = None
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        if "batch-detection" in self.config.dataset:
            return self._train_kfold_pairwise(pre_trained_model)
        return self._train_kfold(pre_trained_model)

    def _train_kfold_pairwise(
        self, pre_trained_model: Optional[nn.Module]
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Binary same/different classification on pairwise difference vectors."""
        from sklearn.model_selection import StratifiedKFold

        self.logger.info("Building pairwise difference-vector dataset for batch-detection.")
        full_samples, full_labels = self.data_module.get_numpy_data()
        siamese_ds = SiameseDataset(full_samples, full_labels)

        X_diff = (siamese_ds.X1 - siamese_ds.X2).cpu().numpy()
        y_pair = siamese_ds.paired_labels.cpu().numpy().flatten().astype(int)
        y_onehot = np.eye(2, dtype=np.float32)[y_pair]

        n_classes = 2
        k_folds = self.config.k_folds
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.config.run)
        all_fold_metrics = []
        last_model = None

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_diff, y_pair)):
            self.logger.info(f"--- Fold {fold + 1}/{k_folds} ---")
            # Balanced sampler to handle same/different class imbalance
            tr_labels = y_pair[tr_idx]
            class_counts = np.bincount(tr_labels)
            sample_weights = torch.tensor(
                (1.0 / class_counts)[tr_labels], dtype=torch.float32
            )
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

            tr_ldr = DataLoader(
                CustomDataset(X_diff[tr_idx], y_onehot[tr_idx]),
                batch_size=self.config.batch_size,
                sampler=sampler,
            )
            val_ldr = DataLoader(
                CustomDataset(X_diff[val_idx], y_onehot[val_idx]),
                batch_size=self.config.batch_size,
            )
            model = create_model(self.config, self.n_features, n_classes).to(self.device)
            if pre_trained_model:
                self.pre_train_orchestrator.adapt_for_finetuning(model, pre_trained_model)
            criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
            opt = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
            last_model, metrics = DeepEngine.train_model(
                model=model,
                train_loader=tr_ldr,
                val_loader=val_ldr,
                criterion=criterion,
                optimizer=opt,
                num_epochs=self.config.epochs,
                patience=self.config.early_stopping,
                is_augmented=self.config.data_augmentation,
                device=self.device,
                num_classes=n_classes,
                regression=False,
                ctx=self.ctx,
            )
            if fold == k_folds - 1 and self.ctx.wandb_run:
                self._log_advanced_visualizations(metrics, val_ldr)
            all_fold_metrics.append(metrics)

        stats = {}
        if all_fold_metrics:
            for k in all_fold_metrics[0].keys():
                vals = [
                    m[k]
                    for m in all_fold_metrics
                    if k in m and isinstance(m[k], (int, float, np.number))
                ]
                if vals:
                    stats[k] = float(np.mean(vals))
            stats["predictions"] = all_fold_metrics[-1].get("predictions")
            for m in reversed(all_fold_metrics):
                if m.get("epoch_metrics"):
                    stats["epoch_metrics"] = m["epoch_metrics"]
                    break
            else:
                stats["epoch_metrics"] = None
            stats["folds"] = all_fold_metrics

        self.ctx.save_results(
            {"stats": stats, "folds": all_fold_metrics},
            filename=f"aggregated_stats_{self.config.dataset}.json",
        )
        return last_model, stats

    def _train_single_split_siamese(
        self, pre_trained_model: Optional[nn.Module]
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        from sklearn.model_selection import train_test_split

        self.logger.info("Using Train/Validation/Test split for batch-detection.")
        full_samples, full_labels = self.data_module.get_numpy_data()
        class_indices = np.argmax(full_labels, axis=1)
        idx = np.arange(len(full_samples))
        n_classes_unique = len(np.unique(class_indices))
        test_size = max(0.2, n_classes_unique / len(full_samples))
        tr_val_idx, te_idx = train_test_split(
            idx, test_size=test_size, random_state=self.config.run, stratify=class_indices
        )
        val_size = max(0.25, n_classes_unique / len(tr_val_idx))
        tr_idx, val_idx = train_test_split(
            tr_val_idx,
            test_size=val_size,
            random_state=self.config.run,
            stratify=class_indices[tr_val_idx],
        )
        tr_ldr = DataLoader(
            CustomDataset(full_samples[tr_idx], full_labels[tr_idx]),
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_ldr = DataLoader(
            CustomDataset(full_samples[val_idx], full_labels[val_idx]),
            batch_size=self.config.batch_size,
        )
        te_ldr = DataLoader(
            CustomDataset(full_samples[te_idx], full_labels[te_idx]),
            batch_size=self.config.batch_size,
        )
        model = create_model(self.config, self.n_features, self.n_classes).to(
            self.device
        )
        if pre_trained_model:
            self.pre_train_orchestrator.adapt_for_finetuning(model, pre_trained_model)
        criterion = (
            nn.MSELoss()
            if self.config.regression
            else nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        )
        opt = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        trained_model, tr_val_met = DeepEngine.train_model(
            model=model,
            train_loader=tr_ldr,
            val_loader=val_ldr,
            criterion=criterion,
            optimizer=opt,
            num_epochs=self.config.epochs,
            patience=self.config.early_stopping,
            is_augmented=self.config.data_augmentation,
            device=self.device,
            num_classes=self.n_classes,
            regression=self.config.regression,
            ctx=self.ctx,
        )
        test_res = DeepEngine.evaluate_model(
            trained_model,
            te_ldr,
            criterion,
            self.device,
            num_classes=self.n_classes,
            regression=self.config.regression,
        )
        if self.ctx.wandb_run:
            self._log_advanced_visualizations(test_res, te_ldr)
        final_metrics = {
            "train_loss": tr_val_met.get("train_loss"),
            "val_loss": tr_val_met.get("val_loss"),
            "test_loss": test_res.get("loss"),
            "test_balanced_accuracy": test_res.get("metrics", {}).get(
                "balanced_accuracy"
            ),
            "predictions": test_res.get("predictions"),
            "epoch_metrics": tr_val_met.get("epoch_metrics"),
        }
        self.ctx.save_results(final_metrics, filename="final_metrics.json")
        return trained_model, final_metrics

    def _train_kfold(
        self, pre_trained_model: Optional[nn.Module]
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        from sklearn.model_selection import StratifiedKFold

        full_samples, full_labels = self.data_module.get_numpy_data()
        k_folds = self.config.k_folds
        skf = StratifiedKFold(
            n_splits=k_folds, shuffle=True, random_state=self.config.run
        )
        all_fold_metrics = []
        last_model = None
        for fold, (tr_idx, val_idx) in enumerate(
            skf.split(full_samples, np.argmax(full_labels, axis=1))
        ):
            self.logger.info(f"--- Fold {fold + 1}/{k_folds} ---")
            tr_ldr = DataLoader(
                CustomDataset(full_samples[tr_idx], full_labels[tr_idx]),
                batch_size=self.config.batch_size,
                shuffle=True,
            )
            val_ldr = DataLoader(
                CustomDataset(full_samples[val_idx], full_labels[val_idx]),
                batch_size=self.config.batch_size,
            )
            model = create_model(self.config, self.n_features, self.n_classes).to(
                self.device
            )
            if pre_trained_model:
                self.pre_train_orchestrator.adapt_for_finetuning(
                    model, pre_trained_model
                )
            criterion = (
                nn.MSELoss()
                if self.config.regression
                else nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
            )
            if self.config.ordinal_method == "coral":
                criterion = coral_loss
            elif self.config.ordinal_method == "clm":
                criterion = cumulative_link_loss
            opt = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
            last_model, metrics = DeepEngine.train_model(
                model=model,
                train_loader=tr_ldr,
                val_loader=val_ldr,
                criterion=criterion,
                optimizer=opt,
                num_epochs=self.config.epochs,
                patience=self.config.early_stopping,
                is_augmented=self.config.data_augmentation,
                device=self.device,
                use_coral=(self.config.ordinal_method == "coral"),
                use_cumulative_link=(self.config.ordinal_method == "clm"),
                num_classes=self.n_classes,
                regression=self.config.regression,
                ctx=self.ctx,
            )
            if fold == k_folds - 1 and self.ctx.wandb_run:
                self._log_advanced_visualizations(metrics, val_ldr)
            all_fold_metrics.append(metrics)

        stats = {}
        if all_fold_metrics:
            for k in all_fold_metrics[0].keys():
                vals = [
                    m[k]
                    for m in all_fold_metrics
                    if k in m and isinstance(m[k], (int, float, np.number))
                ]
                if vals:
                    stats[k] = float(np.mean(vals))
            # Include specific data for visualization from the last fold
            stats["predictions"] = all_fold_metrics[-1].get("predictions")
            # Find any non-None epoch metrics in the folds
            for m in reversed(all_fold_metrics):
                if m.get("epoch_metrics"):
                    stats["epoch_metrics"] = m["epoch_metrics"]
                    break
            else:
                stats["epoch_metrics"] = None

            stats["folds"] = all_fold_metrics

        self.ctx.save_results(
            {"stats": stats, "folds": all_fold_metrics},
            filename=f"aggregated_stats_{self.config.dataset}.json",
        )
        return last_model, stats

    def _log_advanced_visualizations(self, results: Dict, loader: DataLoader):
        class_names = self.data_module.get_class_names()
        preds_dict = results.get("predictions", results.get("best_val_predictions", {}))
        y_true, y_preds, y_probs = (
            preds_dict.get("labels"),
            preds_dict.get("preds"),
            preds_dict.get("probs"),
        )
        if not self.config.regression and y_probs is not None:
            self.ctx.log_summary_charts(y_true, y_probs, class_names)
        batch = next(iter(loader))
        spectra = batch[0]
        self.ctx.log_prediction_table(
            spectra=spectra.cpu().numpy(),
            preds=y_preds[: len(spectra)],
            targets=y_true[: len(spectra)],
            probs=y_probs[: len(spectra)] if y_probs is not None else None,
            class_names=class_names,
        )


def run_training_pipeline(
    config: TrainingConfig,
    wandb_run: Optional[Any] = None,
    ctx: Optional[RunContext] = None,
) -> Dict[str, Any]:
    started_wandb = False
    if wandb_run is None and config.wandb_log:
        started_wandb = True
    trainer = ModelTrainer(config, wandb_run=wandb_run, ctx=ctx)
    try:
        pre_trained_model = trainer.pre_train()
        model, stats = trainer.train(pre_trained_model)
        return stats
    finally:
        if started_wandb and trainer.wandb_run:
            trainer.wandb_run.finish()
