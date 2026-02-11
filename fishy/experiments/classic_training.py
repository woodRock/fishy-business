# -*- coding: utf-8 -*-
"""
Unified orchestrator for non-deep learning experiments (Classic ML & Evolutionary).
Standardized to handle all models following the scikit-learn fit/predict interface.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from dataclasses import asdict

from fishy.data.module import create_data_module
from fishy._core.utils import RunContext
from fishy._core.config import TrainingConfig
from fishy._core.config_loader import load_config
from fishy._core.factory import get_model_class
import wandb

logger = logging.getLogger(__name__)


class SklearnTrainer:
    """
    Orchestrates training and evaluation for models following the sklearn interface.
    This includes classic ML (SVM, RF) and Evolutionary Algorithms (GP, GA).

    Examples:
        >>> from fishy._core.config import TrainingConfig
        >>> cfg = TrainingConfig(model="opls-da", dataset="species")
        >>> trainer = SklearnTrainer(cfg, "opls-da", "species")
        >>> trainer.method
        'classic'
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

        # Load registries
        all_cfg = load_config("models")
        self.classic_cfg = all_cfg.get("classic_models", {})
        self.evolutionary_cfg = all_cfg.get("evolutionary_models", {})
        self.probabilistic_cfg = all_cfg.get("probabilistic_models", {})

        # Determine method for RunContext
        if self.model_name in self.classic_cfg:
            self.method = "classic"
            self.model_entry = self.classic_cfg[self.model_name]
        elif self.model_name in self.evolutionary_cfg:
            self.method = "evolutionary"
            self.model_entry = self.evolutionary_cfg[self.model_name]
        elif self.model_name in self.probabilistic_cfg:
            self.method = "probabilistic"
            self.model_entry = self.probabilistic_cfg[self.model_name]
        else:
            # Fallback/Unknown
            self.method = config.method if hasattr(config, "method") else "classic"
            self.model_entry = None

        self.wandb_run = wandb_run
        if self.wandb_run is None and self.config.wandb_log:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=asdict(self.config),
                reinit=True,
                group=f"{self.config.dataset}_{self.model_name}",
                job_type=f"{self.method}_training",
            )

        self.ctx = RunContext(
            dataset=dataset_name,
            method=self.method,
            model_name=self.model_name,
            wandb_run=self.wandb_run,
        )
        self.logger = self.ctx.logger

    def run(self) -> Dict[str, Any]:
        """Runs the training and evaluation pipeline."""
        self.logger.info(
            f"Starting {self.method} experiment: {self.model_name} on {self.dataset_name}"
        )

        # Standardized Data Loading
        data_module = create_data_module(
            dataset_name=self.dataset_name, file_path=self.file_path
        )
        data_module.setup()
        X, y = data_module.get_numpy_data(labels_as_indices=True)

        return self._run_cv(X, y, data_module)

    def _get_model_instance(self) -> Any:
        """Instantiates the model with appropriate parameters."""
        if self.model_entry is None:
            raise ValueError(f"Model {self.model_name} not found in any registry.")

        # Handle string path vs dict entry
        model_path = (
            self.model_entry["path"]
            if isinstance(self.model_entry, dict)
            else self.model_entry
        )
        model_class = get_model_class(model_path)

        # Evolutionary specific instantiation
        if self.method == "evolutionary":
            # Extract parameters with defaults
            pop = self.config.batch_size if self.config.batch_size else 100
            gens = self.config.epochs if self.config.epochs else 20

            if self.model_name in ["gp", "ga", "eda"]:
                return model_class(
                    generations=gens, population_size=pop, random_state=self.run_id
                )
            elif self.model_name == "es":
                return model_class(
                    generations=gens, mu=pop // 2, lambda_=pop, random_state=self.run_id
                )
            elif self.model_name == "pso":
                return model_class(
                    iterations=gens, population_size=pop, random_state=self.run_id
                )
            else:
                return model_class()

        # Classic ML instantiation
        try:
            return model_class(random_state=self.run_id)
        except TypeError:
            return model_class()

    def _run_cv(self, X: np.ndarray, y: np.ndarray, data_module: Any) -> Dict[str, Any]:
        # Scale data for classic models (evolutionary usually don't strictly require it but it doesn't hurt)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        skf = StratifiedKFold(
            n_splits=self.config.k_folds, shuffle=True, random_state=self.run_id
        )

        val_results, train_results = [], []
        last_fold_info = {}

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = self._get_model_instance()
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)

            if self.config.regression:
                from sklearn.metrics import mean_absolute_error, r2_score

                train_acc = r2_score(y_train, y_train_pred)
                val_acc = r2_score(y_test, y_pred)
                val_mae = mean_absolute_error(y_test, y_pred)
                self.logger.info(f"Fold {fold}: R2={val_acc:.4f}, MAE={val_mae:.4f}")
            else:
                # If model is a regressor but we are in classification mode,
                # round continuous outputs to nearest class index
                if y_pred.dtype.kind in "fc":  # float or complex
                    self.logger.debug(
                        "Rounding continuous predictions for classification metrics."
                    )
                    y_pred_metrics = np.round(y_pred).clip(min(y), max(y)).astype(int)
                    y_train_pred_metrics = (
                        np.round(y_train_pred).clip(min(y), max(y)).astype(int)
                    )
                else:
                    y_pred_metrics = y_pred
                    y_train_pred_metrics = y_train_pred

                train_acc = balanced_accuracy_score(y_train, y_train_pred_metrics)
                val_acc = balanced_accuracy_score(y_test, y_pred_metrics)

                # Update y_pred for visualizations to ensure they are integers
                y_pred = y_pred_metrics

            train_results.append(train_acc)
            val_results.append(val_acc)

            metric_name = (
                "epoch/train_r2"
                if self.config.regression
                else "epoch/train_balanced_accuracy"
            )
            val_metric_name = (
                "epoch/val_r2"
                if self.config.regression
                else "epoch/val_balanced_accuracy"
            )

            self.ctx.log_metric(
                fold, {metric_name: train_acc, val_metric_name: val_acc}
            )

            # Capture last fold info
            if fold == self.config.k_folds:
                y_probs = None
                if not self.config.regression:
                    y_probs = (
                        model.predict_proba(X_test)
                        if hasattr(model, "predict_proba")
                        else None
                    )

                last_fold_info = {"labels": y_test, "preds": y_pred, "probs": y_probs}

                # Log advanced visualizations to W&B
                if self.ctx.wandb_run:
                    if self.config.regression:
                        from fishy.analysis.statistical import (
                            analyze_regression_predictions,
                        )

                        analyze_regression_predictions(
                            last_fold_info,
                            fold,
                            self.ctx,
                            dataset_name=self.dataset_name,
                        )
                    else:
                        class_names = data_module.get_class_names()
                        if y_probs is not None:
                            self.ctx.log_summary_charts(y_test, y_probs, class_names)

                        # Ensure we have probs for the table
                        table_probs = (
                            y_probs
                            if y_probs is not None
                            else np.eye(len(class_names))[y_pred.astype(int)]
                        )
                        self.ctx.log_prediction_table(
                            spectra=X_test,
                            preds=y_pred.astype(int),
                            targets=y_test.astype(int),
                            probs=table_probs,
                            class_names=class_names,
                        )

        # Update stats names
        acc_key = "train_r2" if self.config.regression else "train_balanced_accuracy"
        val_key = "val_r2" if self.config.regression else "val_balanced_accuracy"

        stats = {
            acc_key: float(np.mean(train_results)),
            val_key: float(np.mean(val_results)),
            f"{val_key}_std": float(np.std(val_results)),
            "epoch_metrics": {val_key: val_results, acc_key: train_results},
            "best_val_predictions": last_fold_info,
            "predictions": last_fold_info,
        }
        self.ctx.save_results({"fold_accuracies": val_results, "stats": stats})
        return stats


def run_sklearn_experiment(
    config: TrainingConfig,
    model_name: str,
    dataset_name: str,
    run_id: int = 0,
    file_path: Optional[str] = None,
    wandb_run: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Unified entry point for Sklearn-compatible model experiments.

    Examples:
        >>> from fishy._core.config import TrainingConfig
        >>> cfg = TrainingConfig(model="opls-da", dataset="species")
        >>> # results = run_sklearn_experiment(cfg, "opls-da", "species")
        >>> True
        True
    """
    started_wandb = False
    if wandb_run is None and config.wandb_log:
        started_wandb = True

    trainer = SklearnTrainer(
        config, model_name, dataset_name, run_id, file_path, wandb_run=wandb_run
    )
    try:
        return trainer.run()
    finally:
        if started_wandb and trainer.wandb_run:
            trainer.wandb_run.finish()


# Aliases for backward compatibility if needed
run_classic_experiment = run_sklearn_experiment
run_evolutionary_experiment = run_sklearn_experiment
