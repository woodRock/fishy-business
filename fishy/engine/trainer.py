# -*- coding: utf-8 -*-
"""
Trainer module for encapsulating the training loop and related logic.
Supports PyTorch (deep), Scikit-Learn (classic), and DEAP (evolutionary) models.
"""

import copy
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from fishy._core.utils import RunContext
from fishy.data.datasets import SiameseDataset
from fishy.engine.losses import coral_loss, cumulative_link_loss, levels_from_labelbatch
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class Trainer:
    """
    Unified trainer for PyTorch models.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        num_epochs: int,
        scheduler: Optional[Any] = None,
        patience: int = 20,
        use_coral: bool = False,
        use_cumulative_link: bool = False,
        num_classes: Optional[int] = None,
        regression: bool = False,
        ctx: Optional[RunContext] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The PyTorch model to be trained.
            criterion (nn.Module): The loss function (e.g., nn.CrossEntropyLoss).
            optimizer (optim.Optimizer): The optimizer (e.g., optim.Adam).
            device (torch.device): The computing device ('cpu' or 'cuda').
            num_epochs (int): The maximum number of epochs to train.
            scheduler (Optional[Any], optional): Learning rate scheduler. Defaults to None.
            patience (int, optional): Early stopping patience in epochs. Defaults to 20.
            use_coral (bool, optional): Enable CORAL ordinal loss. Defaults to False.
            use_cumulative_link (bool, optional): Enable cumulative link ordinal loss. Defaults to False.
            num_classes (Optional[int], optional): Number of classes, required if use_coral is True. Defaults to None.
            regression (bool, optional): Set to True if performing regression. Defaults to False.
            ctx (Optional[RunContext], optional): Context for experiment tracking. Defaults to None.
            logger (Optional[logging.Logger], optional): Custom logger. If None, uses module logger. Defaults to None.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.scheduler = scheduler
        self.patience = patience
        self.use_coral = use_coral
        self.use_cumulative_link = use_cumulative_link
        self.num_classes = num_classes
        self.regression = regression
        self.ctx = ctx
        self.logger = logger if logger else logging.getLogger(__name__)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """
        Executes the full training process for PyTorch models.
        """
        best_val_accuracy = float("-inf")
        epochs_no_improve = 0
        best_model_state_cpu = None
        best_fold_metrics = None
        best_val_predictions = None
        
        epoch_log = {
            "train_losses": [], "val_losses": [], "train_metrics": [], "val_metrics": [], "learning_rates": [],
        }

        for epoch in tqdm(range(self.num_epochs), desc="Training", unit="epoch", leave=False):
            self.model.train()
            train_results = self._run_epoch(train_loader, is_training=True)

            val_results = {"loss": float("nan"), "metrics": {}}
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    val_results = self._run_epoch(val_loader, is_training=False)

            epoch_log["train_losses"].append(train_results["loss"])
            epoch_log["train_metrics"].append(train_results["metrics"])
            epoch_log["learning_rates"].append(self.optimizer.param_groups[0]["lr"])
            
            if val_loader:
                epoch_log["val_losses"].append(val_results["loss"])
                epoch_log["val_metrics"].append(val_results["metrics"])
                
                current_val_acc = val_results["metrics"].get("balanced_accuracy", float("-inf"))

                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(current_val_acc)
                    else:
                        self.scheduler.step()

                if self.ctx:
                    self._log_to_context(epoch, train_results, val_results)
                
                if current_val_acc > best_val_accuracy:
                    best_val_accuracy = current_val_acc
                    best_model_state_cpu = OrderedDict((k, v.clone().cpu()) for k, v in self.model.state_dict().items())
                    best_fold_metrics = self._compile_best_metrics(epoch, train_results, val_results)
                    best_val_predictions = val_results["predictions"]
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        break
            
            if (epoch + 1) % 10 == 0 or epoch == self.num_epochs - 1:
                val_loss_str = f"VL:{val_results['loss']:.3f}" if val_loader else ""
                val_acc_str = f"VAcur:{val_results['metrics'].get('balanced_accuracy', 0):.3f}" if val_loader else ""
                self.logger.info(f"E{epoch+1} TL:{train_results['loss']:.3f} {val_loss_str} {val_acc_str}")

        if not val_loader:
            best_model_state_cpu = OrderedDict((k, v.clone().cpu()) for k, v in self.model.state_dict().items())
            best_fold_metrics = self._compile_best_metrics(self.num_epochs - 1, train_results, val_results)

        if best_fold_metrics is None and val_loader:
             best_fold_metrics = self._compile_best_metrics(self.num_epochs - 1, train_results, val_results)

        return {
            "best_accuracy": best_val_accuracy,
            "best_model_state": best_model_state_cpu,
            "best_fold_metrics": best_fold_metrics,
            "epoch_metrics": epoch_log,
            "best_val_predictions": best_val_predictions,
            "predictions": best_val_predictions # Alias for easier access
        }

    def evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        """Evaluates the model on a given data loader."""
        self.model.eval()
        with torch.no_grad():
            results = self._run_epoch(loader, is_training=False)
        return results

    def _run_epoch(self, loader: DataLoader, is_training: bool) -> Dict[str, Any]:
        total_loss, all_labels_np, all_preds_np, all_probs_np = 0.0, [], [], []
        for batch in loader:
            inputs, labels_batch = self._unpack_batch(batch)
            inputs, labels_on_device = self._to_device(inputs, labels_batch)
            if is_training: self.optimizer.zero_grad()
            outputs = self._forward_pass(inputs)
            loss = self._compute_loss(outputs, labels_on_device)
            if is_training:
                loss.backward(); self.optimizer.step()
            batch_size = inputs[0].size(0) if isinstance(inputs, tuple) else inputs.size(0)
            total_loss += loss.item() * batch_size
            preds, probs, actuals = self._process_predictions(outputs, labels_on_device)
            all_labels_np.append(actuals.cpu().numpy())
            all_preds_np.append(preds.detach().cpu().numpy())
            if probs is not None: all_probs_np.append(probs.detach().cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        final_labels = np.concatenate(all_labels_np); final_preds = np.concatenate(all_preds_np)
        final_probs = np.concatenate(all_probs_np) if all_probs_np else None
        metrics = self._calculate_metrics(final_labels, final_preds)
        return {"loss": avg_loss, "metrics": metrics, "predictions": {"labels": final_labels, "preds": final_preds, "probs": final_probs}}

    def _unpack_batch(self, batch):
        if len(batch) == 3: return (batch[0], batch[1]), batch[2]
        return batch[0], batch[1]

    def _to_device(self, inputs, labels):
        if isinstance(inputs, tuple): return (inputs[0].to(self.device), inputs[1].to(self.device)), labels.to(self.device)
        return inputs.to(self.device), labels.to(self.device)

    def _forward_pass(self, inputs):
        if isinstance(inputs, tuple): return self.model(inputs[0], inputs[1])
        if hasattr(self.model, "decode") and hasattr(self.model, "reparameterize"):
             _, _, _, outputs = self.model(inputs); return outputs
        return self.model(inputs)

    def _compute_loss(self, outputs, labels):
        if self.regression:
            actual = labels.squeeze(-1).float()
            return self.criterion(outputs.squeeze(), actual)
        if labels.dim() > 1 and labels.shape[1] > 1: actual = labels.argmax(dim=1)
        elif labels.dim() > 1: actual = labels.squeeze(-1)
        else: actual = labels
        if self.use_coral:
            levels = levels_from_labelbatch(actual, num_classes=self.num_classes, dtype=torch.float32).to(self.device)
            return self.criterion(outputs, levels)
        return self.criterion(outputs, actual.long())

    def _process_predictions(self, outputs, labels):
        if self.regression:
            actual = labels.squeeze(-1).float(); return outputs.squeeze(), None, actual
        if labels.dim() > 1 and labels.shape[1] > 1: actual = labels.argmax(dim=1)
        elif labels.dim() > 1: actual = labels.squeeze(-1)
        else: actual = labels
        if self.use_coral or self.use_cumulative_link:
             probs = torch.sigmoid(outputs)
             preds = torch.sum((probs > 0.5), dim=1).long()
             return preds, probs, actual
        probs = torch.softmax(outputs, dim=1); preds = outputs.argmax(dim=1)
        return preds, probs, actual

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        if self.regression:
             metrics = {"mae": mean_absolute_error(y_true, y_pred), "mse": mean_squared_error(y_true, y_pred), "r2": r2_score(y_true, y_pred)}
             metrics["balanced_accuracy"] = balanced_accuracy_score(y_true.astype(int), np.round(y_pred).astype(int))
             return metrics
        labels_for_scoring = np.unique(np.concatenate([y_true, y_pred])).astype(int)
        return {
            "accuracy": np.mean(y_true == y_pred), "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred), "mse": mean_squared_error(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels_for_scoring),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels_for_scoring),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels_for_scoring),
        }

    def _log_to_context(self, epoch, train_results, val_results):
        epoch_metrics = {
            "epoch/train_loss": train_results["loss"], "epoch/val_loss": val_results["loss"],
            "epoch/train_balanced_accuracy": train_results["metrics"].get("balanced_accuracy", 0.0),
            "epoch/val_balanced_accuracy": val_results["metrics"].get("balanced_accuracy", 0.0),
        }
        self.ctx.log_metric(epoch + 1, epoch_metrics)

    def _compile_best_metrics(self, epoch, train_results, val_results):
        metrics = {"train_loss": train_results["loss"], "val_loss": val_results["loss"], "epoch": epoch}
        if val_results.get("metrics"):
            for m, v in val_results["metrics"].items(): metrics[f"val_{m}"] = v
        if train_results.get("metrics"):
            for m, v in train_results["metrics"].items(): metrics[f"train_{m}"] = v
        return metrics


class TrainingEngine:
    """
    High-level engine that dispatches training tasks to appropriate trainers (Deep, Classic, or GP).
    """

    @staticmethod
    def run_classic(config: Any, model_name: str, dataset_name: str, run_id: int, file_path: Optional[str] = None) -> Dict[str, float]:
        """Calls the ClassicTrainer."""
        from fishy.experiments.classic_training import run_classic_experiment
        return run_classic_experiment(config, model_name, dataset_name, run_id, file_path)

    @staticmethod
    def run_evolutionary(dataset: str, generations: int, population: int, run: int, data_file_path: Optional[str] = None, wandb_log: bool = False) -> Dict[str, float]:
        """Calls the Genetic Programming runner."""
        from fishy.experiments.evolutionary import run_gp_experiment
        return run_gp_experiment(dataset=dataset, generations=generations, population=population, run=run, data_file_path=data_file_path, wandb_log=wandb_log)

    @staticmethod
    def run_deep(config: Any) -> Dict[str, Any]:
        """Calls the Deep Learning training pipeline."""
        from fishy.experiments.deep_training import run_training_pipeline
        return run_training_pipeline(config)
