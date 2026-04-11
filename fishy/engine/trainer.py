# -*- coding: utf-8 -*-
"""
Trainer module for encapsulating the training loop and related logic.
"""

import copy
import logging
import time
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from sklearn.model_selection import StratifiedKFold

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.panel import Panel

from fishy._core.utils import RunContext, get_device, console
from fishy.data.datasets import SiameseDataset
from fishy.engine.losses import coral_loss, cumulative_link_loss, levels_from_labelbatch
from fishy.data.augmentation import AugmentationConfig, DataAugmenter

from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class Trainer:
    """Unified trainer for PyTorch models with Rich integration."""

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
        self.logger = (
            logger if logger else (ctx.logger if ctx else logging.getLogger(__name__))
        )

    def train(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        best_val_accuracy, epochs_no_improve = float("-inf"), 0
        best_model_state_cpu, best_fold_metrics, best_val_predictions = None, None, None
        epoch_log = {
            "train_losses": [],
            "val_losses": [],
            "train_metrics": [],
            "val_metrics": [],
            "learning_rates": [],
        }

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Training {self.num_epochs} Epochs...", total=self.num_epochs
            )
            for epoch in range(self.num_epochs):
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
                    cur_acc = val_results["metrics"].get(
                        "balanced_accuracy", float("-inf")
                    )
                    if self.scheduler:
                        if isinstance(
                            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                        ):
                            self.scheduler.step(cur_acc)
                        else:
                            self.scheduler.step()
                    if self.ctx:
                        self._log_to_context(epoch, train_results, val_results)
                    if cur_acc > best_val_accuracy:
                        best_val_accuracy = cur_acc
                        best_model_state_cpu = OrderedDict(
                            (k, v.clone().cpu())
                            for k, v in self.model.state_dict().items()
                        )
                        best_fold_metrics = self._compile_best_metrics(
                            epoch, train_results, val_results
                        )
                        best_val_predictions = val_results["predictions"]
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= self.patience:
                            break
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]E{epoch+1}[/] [dim]Loss: {train_results['loss']:.3f}[/]",
                )

        if not val_loader:
            best_model_state_cpu = OrderedDict(
                (k, v.clone().cpu()) for k, v in self.model.state_dict().items()
            )
            best_fold_metrics = self._compile_best_metrics(
                self.num_epochs - 1, train_results, val_results
            )
        return {
            "best_accuracy": best_val_accuracy,
            "best_model_state": best_model_state_cpu,
            "best_fold_metrics": best_fold_metrics,
            "epoch_metrics": epoch_log,
            "predictions": best_val_predictions,
        }

    def evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            return self._run_epoch(loader, is_training=False)

    def _run_epoch(self, loader: DataLoader, is_training: bool) -> Dict[str, Any]:
        total_loss, all_labels, all_preds, all_probs = 0.0, [], [], []
        for batch in loader:
            inputs, labels_batch = self._unpack_batch(batch)
            inputs, labels_dev = self._to_device(inputs, labels_batch)
            if is_training:
                self.optimizer.zero_grad()
            outputs = self._forward_pass(inputs)
            loss = self._compute_loss(outputs, labels_dev)
            if is_training:
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item() * (
                inputs[0].size(0) if isinstance(inputs, tuple) else inputs.size(0)
            )
            preds, probs, actuals = self._process_predictions(outputs, labels_dev)
            all_labels.append(actuals.cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())
            if probs is not None:
                all_probs.append(probs.detach().cpu().numpy())
        final_labels, final_preds = np.concatenate(all_labels), np.concatenate(
            all_preds
        )
        metrics = self._calculate_metrics(final_labels, final_preds)
        return {
            "loss": total_loss / len(loader.dataset),
            "metrics": metrics,
            "predictions": {
                "labels": final_labels,
                "preds": final_preds,
                "probs": (np.concatenate(all_probs) if all_probs else None),
            },
        }

    def _labels_to_indices(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert one-hot or multi-dim labels to flat class indices."""
        if labels.dim() > 1 and labels.shape[1] > 1:
            return labels.argmax(dim=1)
        if labels.dim() > 1:
            return labels.squeeze(-1)
        return labels

    def _unpack_batch(self, batch):
        if len(batch) == 3:
            return (batch[0], batch[1]), batch[2]
        return batch[0], batch[1]

    def _to_device(self, inputs, labels):
        if isinstance(inputs, tuple):
            return tuple(x.to(self.device) for x in inputs), labels.to(self.device)
        return inputs.to(self.device), labels.to(self.device)

    def _forward_pass(self, inputs):
        if isinstance(inputs, tuple):
            return self.model(*inputs)
        if hasattr(self.model, "decode") and hasattr(self.model, "reparameterize"):
            # Handle VAE or models with tuple returns
            out = self.model(inputs)
            return out[3] if isinstance(out, tuple) and len(out) > 3 else out
        return self.model(inputs)

    def _compute_loss(self, outputs, labels):
        if self.regression:
            loss = self.criterion(outputs.squeeze(), labels.squeeze(-1).float())
        elif self.use_coral:
            actual = self._labels_to_indices(labels)
            loss = self.criterion(
                outputs,
                levels_from_labelbatch(
                    actual, num_classes=self.num_classes, dtype=torch.float32
                ).to(self.device),
            )
        else:
            actual = self._labels_to_indices(labels)
            loss = self.criterion(outputs, actual.long())
        if self.model.training and hasattr(self.model, "binding_loss"):
            loss = loss + self.model.binding_loss()
        return loss

    def _process_predictions(self, outputs, labels):
        if self.regression:
            return outputs.squeeze(), None, labels.squeeze(-1).float()
        actual = self._labels_to_indices(labels)
        if self.use_coral or self.use_cumulative_link:
            probs = torch.sigmoid(outputs)
            return torch.sum((probs > 0.5), dim=1).long(), probs, actual
        return outputs.argmax(dim=1), torch.softmax(outputs, dim=1), actual

    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if self.regression:
                try:
                    bca = balanced_accuracy_score(
                        y_true.astype(int), np.round(y_pred).astype(int)
                    )
                except (ValueError, TypeError):
                    bca = 0.0
                return {
                    "mae": mean_absolute_error(y_true, y_pred),
                    "mse": mean_squared_error(y_true, y_pred),
                    "r2": r2_score(y_true, y_pred),
                    "balanced_accuracy": bca,
                }
            all_labels = np.arange(self.num_classes) if self.num_classes else None
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "precision": precision_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                    labels=all_labels,
                ),
                "recall": recall_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                    labels=all_labels,
                ),
                "f1": f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                    labels=all_labels,
                ),
                "mae": mean_absolute_error(y_true, y_pred),
                "mse": mean_squared_error(y_true, y_pred),
            }

    def _log_to_context(self, epoch, tr, val):
        self.ctx.log_metric(
            epoch + 1,
            {
                "epoch/train_loss": tr["loss"],
                "epoch/val_loss": val["loss"],
                "epoch/train_balanced_accuracy": tr["metrics"].get(
                    "balanced_accuracy", 0.0
                ),
                "epoch/val_balanced_accuracy": val["metrics"].get(
                    "balanced_accuracy", 0.0
                ),
            },
        )

    def _compile_best_metrics(self, epoch, tr, val):
        m = {"train_loss": tr["loss"], "val_loss": val["loss"], "epoch": epoch}
        for k, v in val.get("metrics", {}).items():
            m[f"val_{k}"] = v
        for k, v in tr.get("metrics", {}).items():
            m[f"train_{k}"] = v
        return m


class DeepEngine:
    """High-level engine for deep learning experiments."""

    @staticmethod
    def train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        num_epochs=100,
        patience=20,
        n_splits=5,
        n_runs=30,
        is_augmented=False,
        device=get_device(),
        val_loader=None,
        use_coral=False,
        use_cumulative_link=False,
        num_classes=None,
        regression=False,
        ctx=None,
    ):
        if isinstance(device, str):
            device = torch.device(device)
        if val_loader:
            trainer = Trainer(
                model,
                criterion,
                optimizer,
                device,
                num_epochs,
                patience=patience,
                use_coral=use_coral,
                use_cumulative_link=use_cumulative_link,
                num_classes=num_classes,
                regression=regression,
                ctx=ctx,
            )
            res = trainer.train(train_loader, val_loader)
            if res["best_model_state"]:
                model.load_state_dict(res["best_model_state"])
            m = res["best_fold_metrics"]
            m["predictions"] = res["predictions"]
            return model, m
        pristine = copy.deepcopy(model).cpu()
        augmenter = (
            DataAugmenter(AugmentationConfig(enabled=True)) if is_augmented else None
        )
        all_metrics = []
        best_overall_acc, best_state = float("-inf"), None
        for r_idx in range(n_runs):
            ds = train_loader.dataset
            lbls = DeepEngine._extract_labels(ds)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=r_idx)
            for f_idx, (tr_idx, val_idx) in enumerate(
                skf.split(np.zeros(len(ds)), lbls), 1
            ):
                f_model = copy.deepcopy(pristine).to(device)
                f_opt = type(optimizer)(
                    f_model.parameters(),
                    **{
                        k: v
                        for k, v in optimizer.defaults.items()
                        if k not in {"decoupled_weight_decay", "step_size", "gamma"}
                    },
                )
                tr_ldr = DataLoader(
                    Subset(ds, tr_idx), batch_size=train_loader.batch_size, shuffle=True
                )
                val_ldr = DataLoader(
                    Subset(ds, val_idx), batch_size=train_loader.batch_size
                )
                if augmenter:
                    tr_ldr = augmenter.augment(tr_ldr)
                trainer = Trainer(
                    f_model,
                    criterion,
                    f_opt,
                    device,
                    num_epochs,
                    patience=patience,
                    use_coral=use_coral,
                    use_cumulative_link=use_cumulative_link,
                    num_classes=num_classes,
                    regression=regression,
                    ctx=ctx,
                )
                res = trainer.train(tr_ldr, val_ldr)
                m = res["best_fold_metrics"]
                if res["best_accuracy"] > best_overall_acc:
                    best_overall_acc = res["best_accuracy"]
                    best_state = res["best_model_state"]
                all_metrics.append(m)
        final = copy.deepcopy(pristine).to(device)
        if best_state:
            final.load_state_dict(best_state)
        return final, {"best_accuracy": best_overall_acc, "all_metrics": all_metrics}

    @staticmethod
    def _extract_labels(ds):
        base = ds.dataset if isinstance(ds, Subset) else ds
        idxs = ds.indices if isinstance(ds, Subset) else range(len(base))
        return np.array(
            [
                (
                    lbl.item()
                    if isinstance(lbl, torch.Tensor) and lbl.numel() == 1
                    else (
                        lbl.argmax().item()
                        if isinstance(lbl, torch.Tensor)
                        else int(lbl)
                    )
                )
                for _, lbl in [base[i] for i in idxs]
            ]
        )

    @staticmethod
    def evaluate_model(model, loader, criterion, device=get_device(), **kwargs):
        if isinstance(device, str):
            device = torch.device(device)
        return Trainer(
            model, criterion, optim.Adam(model.parameters()), device, 1, **kwargs
        ).evaluate(loader)

    @staticmethod
    def transfer_learning(dataset_name, model, file_path):
        try:
            model.load_state_dict(
                torch.load(file_path, map_location="cpu"), strict=False
            )
            console.print(f"[success]Transferred weights from {file_path}[/]")
        except Exception as e:
            console.print(f"[error]Transfer failed: {e}[/]")
        return model
