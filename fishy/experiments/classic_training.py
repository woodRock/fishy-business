# -*- coding: utf-8 -*-
"""
Unified orchestrator for classic machine learning experiments.
Uses DataModule for standardized data loading.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

from fishy.data.module import create_data_module
from fishy._core.utils import RunContext
from fishy._core.config import TrainingConfig
from fishy._core.config_loader import load_config
from fishy._core.factory import get_model_class
import wandb
from dataclasses import asdict


class ClassicTrainer:
    """
    Orchestrates training and evaluation for non-deep learning models.
    Standardized to use DataModule for all data loading.
    """

    def __init__(
        self,
        config: TrainingConfig,
        model_name: str,
        dataset_name: str,
        run_id: int = 0,
        file_path: Optional[str] = None,
        wandb_run: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.model_name = model_name.lower()
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.file_path = file_path

        # Load classic models configuration
        self.models_cfg = load_config("models")["classic_models"]

        self.wandb_run = wandb_run
        if self.wandb_run is None and self.config.wandb_log:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=asdict(self.config),
                reinit=True,
                group=f"{self.config.dataset}_{self.model_name}",
                job_type="classic_training",
            )

        self.ctx = RunContext(
            dataset=dataset_name,
            method="classic",
            model_name=self.model_name,
            wandb_run=self.wandb_run,
        )
        self.logger = self.ctx.logger

    def run(self) -> Dict[str, float]:
        """Runs the classic experiment pipeline."""
        self.logger.info(
            f"Starting classic experiment: {self.model_name} on {self.dataset_name}"
        )

        # Standardized Data Loading
        data_module = create_data_module(dataset_name=self.dataset_name, file_path=self.file_path)
        data_module.setup()
        X, y = data_module.get_numpy_data(labels_as_indices=True)

        return self._run_standard_model(X, y, data_module)

    def _run_standard_model(self, X: np.ndarray, y: np.ndarray, data_module: Any) -> Dict[str, float]:
        if self.model_name not in self.models_cfg:
            raise ValueError(f"Model {self.model_name} not found in models.yaml.")

        model_class = get_model_class(self.models_cfg[self.model_name])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        skf = StratifiedKFold(n_splits=self.config.k_folds, shuffle=True, random_state=self.run_id)

        val_results, train_results = [], []
        last_fold_info = {}
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                clf = model_class(random_state=self.run_id)
            except TypeError:
                clf = model_class()
                
            clf.fit(X_train, y_train)

            y_train_pred = clf.predict(X_train)
            train_acc = balanced_accuracy_score(y_train, y_train_pred)
            train_results.append(train_acc)

            y_pred = clf.predict(X_test)
            y_probs = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
            val_acc = balanced_accuracy_score(y_test, y_pred)
            val_results.append(val_acc)

            self.ctx.log_metric(fold, {
                "epoch/train_balanced_accuracy": train_acc,
                "epoch/val_balanced_accuracy": val_acc
            })

            # Capture last fold for figure generation
            if fold == self.config.k_folds:
                last_fold_info = {
                    "labels": y_test,
                    "preds": y_pred,
                    "probs": y_probs
                }

            if fold == self.config.k_folds and self.ctx.wandb_run:
                class_names = data_module.get_class_names()
                if y_probs is not None:
                    self.ctx.log_summary_charts(y_test, y_probs, class_names)
                self.ctx.log_prediction_table(
                    spectra=X_test, preds=y_pred.astype(int), targets=y_test.astype(int),
                    probs=y_probs if y_probs is not None else np.eye(len(class_names))[y_pred.astype(int)],
                    class_names=class_names,
                )

        stats = {
            "train_balanced_accuracy": float(np.mean(train_results)),
            "val_balanced_accuracy": float(np.mean(val_results)),
            "val_balanced_accuracy_std": float(np.std(val_results)),
            "epoch_metrics": {
                "val_balanced_accuracy": val_results,
                "train_balanced_accuracy": train_results
            },
            "best_val_predictions": last_fold_info,
            "predictions": last_fold_info
        }
        self.ctx.save_results({"fold_accuracies": val_results, "stats": stats})
        return stats


def run_classic_experiment(
    config: TrainingConfig,
    model_name: str,
    dataset_name: str,
    run_id: int = 0,
    file_path: Optional[str] = None,
    wandb_run: Optional[Any] = None,
) -> Dict[str, float]:
    started_wandb = False
    if wandb_run is None and config.wandb_log:
        started_wandb = True

    trainer = ClassicTrainer(config, model_name, dataset_name, run_id, file_path, wandb_run=wandb_run)
    try:
        return trainer.run()
    finally:
        if started_wandb and trainer.wandb_run:
            trainer.wandb_run.finish()
