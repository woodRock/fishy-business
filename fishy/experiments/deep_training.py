# -*- coding: utf-8 -*-
"""
Training orchestrator for the deep learning pipeline.
"""

import logging
import time
from dataclasses import asdict
from typing import Dict, Optional, Tuple, List, Any, Callable
import wandb

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from fishy._core.config import TrainingConfig
from fishy._core.factory import create_model
from fishy._core.utils import RunContext, get_device
from fishy.experiments.pre_training_orchestrator import PreTrainingOrchestrator
from fishy.engine.training_loops import train_model, evaluate_model
from fishy.data.module import create_data_module
from fishy.data.datasets import CustomDataset, SiameseDataset
from fishy.engine.losses import coral_loss, cumulative_link_loss


def analyze_oil_predictions(
    predictions: Dict[str, np.ndarray], fold: int, config: TrainingConfig, ctx: RunContext
) -> None:
    """
    Analyzes and visualizes predictions specifically for the oil dataset.

    Generates confusion matrices and prediction error distributions, saving them
    via the RunContext.

    Args:
        predictions (Dict[str, np.ndarray]): Dictionary containing 'labels' and 'preds'.
        fold (int): The current cross-validation fold index.
        config (TrainingConfig): Experiment configuration.
        ctx (RunContext): Context for logging and saving figures.
    """
    logger = ctx.logger
    if not predictions:
        logger.warning(f"Fold {fold + 1}: No predictions to analyze.")
        return

    true_labels = predictions["labels"]
    pred_labels = predictions["preds"]

    if config.regression:
        from sklearn.metrics import r2_score

        mae = np.mean(np.abs(true_labels - pred_labels))
        logger.info(f"Fold {fold + 1} Regression MAE: {mae:.4f}")
        r2 = r2_score(true_labels, pred_labels)
        logger.info(f"Fold {fold + 1} Regression R2 Score: {r2:.4f}")
    else:
        mae = np.mean(np.abs(true_labels - pred_labels))
        logger.info(f"Fold {fold + 1} Ordinal MAE: {mae:.4f}")
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Fold {fold + 1} Confusion Matrix for Oil Dataset")
        ctx.save_figure(plt, f"oil_confusion_matrix_fold_{fold + 1}.png")
        plt.close()

    errors = pred_labels - true_labels
    plt.figure()
    plt.hist(errors, bins=np.arange(errors.min(), errors.max() + 2) - 0.5, rwidth=0.8)
    plt.xlabel("Prediction Error (Predicted - True)")
    plt.ylabel("Frequency")
    plt.title(f"Fold {fold + 1} Prediction Error Distribution for Oil Dataset")
    ctx.save_figure(plt, f"oil_prediction_error_fold_{fold + 1}.png")
    plt.close()


class ModelTrainer:
    """
    Orchestrates the model training pipeline, from data setup to pre-training and fine-tuning.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.wandb_run = None
        if self.config.wandb_log:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=asdict(self.config),
                reinit=True,
                group=f"{self.config.dataset}_{self.config.model}",
                job_type="training",
            )
        self.ctx = RunContext(
            dataset=config.dataset,
            method="deep",
            model_name=config.model,
            wandb_run=self.wandb_run,
        )
        self.logger = self.ctx.logger
        self.ctx.save_config(config)

        self.device = get_device()

        dataset_name = self.config.dataset
        if self.config.regression and self.config.dataset == "oil":
            dataset_name = "oil_regression"

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
        """
        Executes the pre-training phase using the orchestrator.

        Returns:
            Optional[nn.Module]: The pre-trained model, or None if pre-training is skipped.
        """
        self.logger.info("Evaluating pre-training phase")
        train_loader = self.data_module.get_train_dataloader()
        val_loader = (
            self.data_module.get_val_dataloader()
            if hasattr(self.data_module, "get_val_dataloader")
            else None
        )

        return self.pre_train_orchestrator.run_all(train_loader, val_loader)

    def train(self, pre_trained_model: Optional[nn.Module] = None) -> Dict[str, Any]:
        """
        Main training loop, supporting both single split and K-Fold CV.

        Args:
            pre_trained_model (Optional[nn.Module], optional): A pre-trained model to fine-tune. Defaults to None.

        Returns:
            Dict[str, Any]: Aggregated metrics from the training run(s).
        """
        from sklearn.model_selection import train_test_split

        if self.data_module is None:
            self.logger.error("Fine-tuning DataModule not set.")
            return {}

        if "instance-recognition" in self.config.dataset:
            return self._train_single_split_siamese(pre_trained_model)
        else:
            return self._train_kfold(pre_trained_model)

    def _train_single_split_siamese(self, pre_trained_model: Optional[nn.Module]) -> Dict[str, Any]:
        """Specialized training for Siamese datasets using a fixed split."""
        from sklearn.model_selection import train_test_split
        self.logger.info("Using Train/Validation/Test split on Siamese pairs for instance-recognition.")

        full_dataset_samples = self.data_module.get_dataset().samples.cpu().numpy()
        full_dataset_labels = self.data_module.get_dataset().labels.cpu().numpy()
        full_siamese_dataset = SiameseDataset(full_dataset_samples, full_dataset_labels)

        pair_indices = np.arange(len(full_siamese_dataset))
        pair_labels_for_stratify = full_siamese_dataset.paired_labels.cpu().numpy().flatten()

        train_val_indices, test_indices = train_test_split(
            pair_indices, test_size=0.2, random_state=self.config.run, stratify=pair_labels_for_stratify,
        )
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=0.25, random_state=self.config.run, stratify=pair_labels_for_stratify[train_val_indices],
        )

        train_loader = DataLoader(Subset(full_siamese_dataset, train_indices), batch_size=self.config.batch_size, shuffle=True, num_workers=4, pin_memory=(self.device.type == "cuda"))
        val_loader = DataLoader(Subset(full_siamese_dataset, val_indices), batch_size=self.config.batch_size, shuffle=False, num_workers=4, pin_memory=(self.device.type == "cuda"))
        test_loader = DataLoader(Subset(full_siamese_dataset, test_indices), batch_size=self.config.batch_size, shuffle=False, num_workers=4, pin_memory=(self.device.type == "cuda"))

        model = create_model(self.config, self.n_features, self.n_classes).to(self.device)
        if pre_trained_model:
            self.pre_train_orchestrator.adapt_for_finetuning(model, pre_trained_model)

        criterion = nn.MSELoss() if self.config.regression else nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        trained_model, train_val_metrics = train_model(
            model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer,
            num_epochs=self.config.epochs, patience=self.config.early_stopping, is_augmented=self.config.data_augmentation,
            device=self.device, use_coral=(self.config.ordinal_method == "coral"), num_classes=self.n_classes, regression=self.config.regression, ctx=self.ctx,
        )

        test_results = evaluate_model(trained_model, test_loader, criterion, self.device, (self.config.ordinal_method == "coral"), self.n_classes, regression=self.config.regression)
        
        if self.ctx.wandb_run:
            self._log_advanced_visualizations(test_results, test_loader)

        final_metrics = self._compile_final_metrics(train_val_metrics, test_results)
        self.ctx.save_results(final_metrics, filename="final_metrics.json")
        torch.save(trained_model.state_dict(), self.ctx.get_checkpoint_path("final_model.pth"))
        return final_metrics

    def _train_kfold(self, pre_trained_model: Optional[nn.Module]) -> Dict[str, Any]:
        """Standard K-Fold Cross-Validation training loop, with optional Group-Aware splitting."""
        from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
        
        full_dataset_samples = self.data_module.get_dataset().samples.cpu().numpy()
        full_dataset_labels = self.data_module.get_dataset().labels.cpu().numpy()
        
        k_folds = self.config.k_folds
        
        if self.config.use_groups:
            self.logger.info("Using StratifiedGroupKFold for cross-validation.")
            cv_splitter = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=self.config.run)
            groups = self.data_module.get_groups()
            split_args = (full_dataset_samples, np.argmax(full_dataset_labels, axis=1), groups)
        else:
            self.logger.info("Using standard StratifiedKFold for cross-validation.")
            cv_splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.config.run)
            split_args = (full_dataset_samples, np.argmax(full_dataset_labels, axis=1))

        all_fold_metrics = []
        for fold, (train_index, val_index) in enumerate(cv_splitter.split(*split_args)):
            self.logger.info(f"--- Starting Fold {fold + 1}/{k_folds} ---")
            
            train_loader = DataLoader(CustomDataset(full_dataset_samples[train_index], full_dataset_labels[train_index]), batch_size=self.config.batch_size, shuffle=True, num_workers=4, pin_memory=(self.device.type == "cuda"))
            val_loader = DataLoader(CustomDataset(full_dataset_samples[val_index], full_dataset_labels[val_index]), batch_size=self.config.batch_size, shuffle=False, num_workers=4, pin_memory=(self.device.type == "cuda"))

            model = create_model(self.config, self.n_features, self.n_classes).to(self.device)
            if pre_trained_model:
                self.pre_train_orchestrator.adapt_for_finetuning(model, pre_trained_model)

            criterion = nn.MSELoss() if self.config.regression else nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
            if self.config.ordinal_method == "coral": criterion = coral_loss
            elif self.config.ordinal_method == "clm": criterion = cumulative_link_loss
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

            trained_model, metrics = train_model(
                model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer,
                num_epochs=self.config.epochs, patience=self.config.early_stopping, is_augmented=self.config.data_augmentation,
                device=self.device, use_coral=(self.config.ordinal_method == "coral"), 
                use_cumulative_link=(self.config.ordinal_method == "clm"),
                num_classes=self.n_classes, regression=self.config.regression, ctx=self.ctx,
            )
            
            if "best_val_predictions" in metrics:
                if self.config.dataset == "oil":
                    analyze_oil_predictions(metrics["best_val_predictions"], fold, self.config, self.ctx)
                if fold == k_folds - 1 and self.ctx.wandb_run:
                    self._log_advanced_visualizations(metrics, val_loader)
                del metrics["best_val_predictions"]

            all_fold_metrics.append(metrics)
            torch.save(trained_model.state_dict(), self.ctx.get_checkpoint_path(f"model_fold_{fold+1}.pth"))

        stats = self._aggregate_fold_metrics(all_fold_metrics)
        self.ctx.save_results({"config": asdict(self.config), "stats": stats, "folds": all_fold_metrics}, filename=f"aggregated_stats_{self.config.dataset}.json")
        return stats

    def _log_advanced_visualizations(self, results: Dict, loader: DataLoader):
        """Helper to log W&B charts and prediction tables."""
        class_names = self.data_module.get_class_names() if hasattr(self.data_module, "get_class_names") else [str(i) for i in range(self.n_classes)]
        preds_dict = results.get("predictions", results.get("best_val_predictions", {}))
        
        y_true, y_preds, y_probs = preds_dict.get("labels"), preds_dict.get("preds"), preds_dict.get("probs")
        
        if not self.config.regression and y_probs is not None:
            self.ctx.log_summary_charts(y_true, y_probs, class_names)

        spectra, _ = next(iter(loader))
        self.ctx.log_prediction_table(spectra=spectra.cpu().numpy(), preds=y_preds[:len(spectra)], targets=y_true[:len(spectra)], probs=y_probs[:len(spectra)] if y_probs is not None else None, class_names=class_names)

    def _compile_final_metrics(self, train_val: Dict, test: Dict) -> Dict:
        return {
            "train_loss": train_val.get("train_loss"),
            "val_loss": train_val.get("val_loss"),
            "test_loss": test.get("loss"),
            "test_balanced_accuracy": test.get("metrics", {}).get("balanced_accuracy"),
        }

    def _aggregate_fold_metrics(self, all_fold_metrics: List[Dict]) -> Dict:
        if not all_fold_metrics: return {}
        return {k: np.mean([m[k] for m in all_fold_metrics if k in m and isinstance(m[k], (int, float))]) for k in all_fold_metrics[0].keys()}


def run_training_pipeline(config: TrainingConfig) -> Dict[str, Any]:
    """
    Executes the full training pipeline for a given configuration.

    Initializes the ModelTrainer, runs pre-training (if configured), and then
    runs the main training/fine-tuning loop.

    Args:
        config (TrainingConfig): The experiment configuration.

    Returns:
        Dict[str, Any]: The final results/metrics of the training pipeline.
    """
    trainer = ModelTrainer(config)
    trainer.logger.info("Starting training pipeline")
    try:
        pre_trained_model = trainer.pre_train()
        return trainer.train(pre_trained_model)
    finally:
        if trainer.wandb_run: trainer.wandb_run.finish()