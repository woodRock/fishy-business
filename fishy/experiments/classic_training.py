# -*- coding: utf-8 -*-
"""
Unified orchestrator for non-deep learning experiments (Classic ML & Evolutionary).
"""

import logging
import numpy as np
import pandas as pd
import warnings
import time
from typing import Dict, List, Optional, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)
from dataclasses import asdict

from fishy.data.module import create_data_module, make_all_pairwise_folds
from fishy._core.utils import RunContext, console
from fishy._core.config import TrainingConfig
from fishy._core.config_loader import load_config
from fishy._core.factory import get_model_class
from fishy._core.constants import DatasetName
import wandb

logger = logging.getLogger(__name__)


class SklearnTrainer:
    """Orchestrates training for sklearn-compatible models."""

    def __init__(
        self,
        config: TrainingConfig,
        model_name: str,
        dataset_name: str,
        run_id: int = 0,
        file_path: Optional[str] = None,
        wandb_run: Optional[Any] = None,
        ctx: Optional[RunContext] = None,
    ) -> None:
        self.config = config
        self.model_name = model_name.lower()
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.file_path = file_path
        all_cfg = load_config("models")
        self.classic_cfg = all_cfg.get("classic_models", {})
        self.evolutionary_cfg = all_cfg.get("evolutionary_models", {})
        self.probabilistic_cfg = all_cfg.get("probabilistic_models", {})
        self.method = (
            "classic"
            if self.model_name in self.classic_cfg
            else (
                "evolutionary"
                if self.model_name in self.evolutionary_cfg
                else (
                    "probabilistic"
                    if self.model_name in self.probabilistic_cfg
                    else "classic"
                )
            )
        )
        self.model_entry = (
            self.classic_cfg.get(self.model_name)
            or self.evolutionary_cfg.get(self.model_name)
            or self.probabilistic_cfg.get(self.model_name)
        )
        self.wandb_run = wandb_run
        if self.wandb_run is None and self.config.wandb_log:
            # Force thread start method for W&B to prevent CUDA 12 hangs during CPU tasks
            import os
            os.environ["WANDB_START_METHOD"] = "thread"
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=asdict(self.config),
                reinit=True,
                group=f"{self.config.dataset}_{self.model_name}",
                job_type=f"{self.method}_training",
            )
        self.ctx = (
            ctx
            if ctx
            else RunContext(
                dataset=dataset_name,
                method=self.method,
                model_name=self.model_name,
                wandb_run=self.wandb_run,
            )
        )
        self.logger = self.ctx.logger

    def run(self) -> Tuple[Any, Dict[str, Any]]:
        start_time = time.time()
        self.data_module = create_data_module(
            dataset_name=self.dataset_name,
            file_path=self.file_path,
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
        X, y = self.data_module.get_numpy_data(labels_as_indices=True)
        self.num_classes = self.data_module.get_num_classes()

        # For batch-detection: all C(N,2) pairs with stratified pair-level K-fold.
        # Identical fold indices are used by deep, classic, and contrastive trainers
        # for a fair, consistent evaluation.
        if DatasetName.BATCH_DETECTION in self.dataset_name:
            full_samples, full_labels = self.data_module.get_numpy_data()
            X1_all, X2_all, pair_labels_all, folds = make_all_pairwise_folds(
                full_samples, full_labels,
                n_splits=self.config.k_folds,
                run_id=self.config.run,
            )
            # Use difference vectors for classic ML models
            X_diff_all = X1_all - X2_all
            
            all_fold_metrics = []
            last_model = None
            
            for fold, (tr_idx, val_idx) in enumerate(folds):
                self.logger.info(f"--- Classic Fold {fold+1}/{self.config.k_folds} ---")
                
                # Fit scaler on train split only
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_diff_all[tr_idx])
                X_val_scaled = scaler.transform(X_diff_all[val_idx])
                
                model = self._get_model_instance()
                # Train binary classifier (Same: 1, Different: 0)
                model.fit(X_tr_scaled, pair_labels_all[tr_idx])
                
                # Evaluate
                y_tr_pred = model.predict(X_tr_scaled)
                y_val_pred = model.predict(X_val_scaled)
                
                # Use binary metrics for batch detection pairs
                tr_met = {
                    "accuracy": accuracy_score(pair_labels_all[tr_idx], y_tr_pred),
                    "balanced_accuracy": balanced_accuracy_score(pair_labels_all[tr_idx], y_tr_pred),
                    "f1": f1_score(pair_labels_all[tr_idx], y_tr_pred, zero_division=0),
                }
                val_met = {
                    "accuracy": accuracy_score(pair_labels_all[val_idx], y_val_pred),
                    "balanced_accuracy": balanced_accuracy_score(pair_labels_all[val_idx], y_val_pred),
                    "f1": f1_score(pair_labels_all[val_idx], y_val_pred, zero_division=0),
                }
                
                fold_res = {f"train_{k}": v for k, v in tr_met.items()}
                fold_res.update({f"val_{k}": v for k, v in val_met.items()})
                all_fold_metrics.append(fold_res)
                last_model = model

            # Aggregate results
            stats = {
                k: float(np.mean([m[k] for m in all_fold_metrics]))
                for k in all_fold_metrics[0].keys()
            }
            # For consistency with other methods, ensure top-level keys point to val metrics
            for k in ["accuracy", "balanced_accuracy", "f1"]:
                if f"val_{k}" in stats:
                    stats[k] = stats[f"val_{k}"]
            
            # Map val_ metrics to test_ for legacy test compatibility
            stats["test_balanced_accuracy"] = stats.get("val_balanced_accuracy", 0.0)
            stats["test_accuracy"] = stats.get("val_accuracy", 0.0)
            stats["test_f1"] = stats.get("val_f1", 0.0)
            stats["val_balanced_accuracy"] = stats.get("test_balanced_accuracy", 0.0)
            stats["val_accuracy"] = stats.get("test_accuracy", 0.0)
            stats["val_f1"] = stats.get("test_f1", 0.0)
            
            stats["folds"] = all_fold_metrics
            stats["total_training_time_s"] = time.time() - start_time
            self.ctx.save_results({"stats": stats})
            return last_model, stats

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        skf = StratifiedKFold(
            n_splits=self.config.k_folds, shuffle=True, random_state=self.run_id
        )
        all_fold_metrics = []
        last_fold_info = {}
        last_model = None
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
            X_train, X_test, y_train, y_test = (
                X_scaled[train_idx],
                X_scaled[test_idx],
                y[train_idx],
                y[test_idx],
            )
            last_model = self._get_model_instance()
            last_model.fit(X_train, y_train)
            y_pred, y_train_pred = last_model.predict(X_test), last_model.predict(
                X_train
            )
            if y_pred.dtype.kind in "fc":
                y_pred = np.round(y_pred).clip(min(y), max(y)).astype(int)
                y_train_pred = np.round(y_train_pred).clip(min(y), max(y)).astype(int)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                tr_met = self._calculate_metrics(y_train, y_train_pred)
                val_met = self._calculate_metrics(y_test, y_pred)
            fold_res = {f"train_{k}": v for k, v in tr_met.items()}
            fold_res.update({f"val_{k}": v for k, v in val_met.items()})
            all_fold_metrics.append(fold_res)
            if fold == self.config.k_folds:
                y_probs = (
                    last_model.predict_proba(X_test)
                    if hasattr(last_model, "predict_proba")
                    else None
                )
                last_fold_info = {"labels": y_test, "preds": y_pred, "probs": y_probs}
                if self.ctx.wandb_run:
                    self.ctx.log_prediction_table(
                        X_test,
                        y_pred.astype(int),
                        y_test.astype(int),
                        (
                            y_probs
                            if y_probs is not None
                            else np.eye(self.num_classes)[y_pred.astype(int)]
                        ),
                        self.data_module.get_class_names(),
                    )
        stats = {
            k: float(np.mean([m[k] for m in all_fold_metrics]))
            for k in all_fold_metrics[0].keys()
        }
        stats["folds"] = all_fold_metrics
        if hasattr(last_model, "best_individual"):
            weights = last_model.best_individual
            if hasattr(weights, "tolist"):
                stats["feature_weights"] = weights.tolist()
            else:
                stats["feature_weights"] = list(weights)

        stats["predictions"] = last_fold_info
        stats["total_training_time_s"] = time.time() - start_time

        self.ctx.save_results({"stats": stats})
        return last_model, stats

    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        all_labels = np.arange(self.num_classes) if self.num_classes else None
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0, labels=all_labels
            ),
            "recall": recall_score(
                y_true, y_pred, average="weighted", zero_division=0, labels=all_labels
            ),
            "f1": f1_score(
                y_true, y_pred, average="weighted", zero_division=0, labels=all_labels
            ),
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
        }

    def _get_model_instance(self) -> Any:
        model_path = (
            self.model_entry["path"]
            if isinstance(self.model_entry, dict)
            else self.model_entry
        )
        model_class = get_model_class(model_path)
        if self.method == "evolutionary":
            pop, gens = (self.config.batch_size or 100), (self.config.epochs or 20)
            if self.model_name in ["gp", "ga", "eda"]:
                return model_class(
                    generations=gens, population_size=pop, random_state=self.run_id
                )
            if self.model_name == "es":
                return model_class(
                    generations=gens, mu=pop // 2, lambda_=pop, random_state=self.run_id
                )
            if self.model_name == "pso":
                return model_class(
                    iterations=gens, population_size=pop, random_state=self.run_id
                )
        try:
            return model_class(random_state=self.run_id)
        except TypeError:
            return model_class()


def run_sklearn_experiment(
    config, model_name, dataset_name, run_id=0, file_path=None, wandb_run=None, ctx=None
):
    started_wandb = False
    if wandb_run is None and config.wandb_log:
        started_wandb = True

    # Ensure config properties are set correctly
    if run_id != 0:
        config.run = run_id
    if file_path:
        config.file_path = file_path

    trainer = SklearnTrainer(
        config,
        model_name,
        dataset_name,
        config.run,
        config.file_path,
        wandb_run=wandb_run,
        ctx=ctx,
    )
    try:
        model, stats = trainer.run()
        return stats
    finally:
        if started_wandb and trainer.wandb_run:
            trainer.wandb_run.finish()
