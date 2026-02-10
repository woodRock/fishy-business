# -*- coding: utf-8 -*-
"""
Unified orchestrator for classic machine learning experiments.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

try:
    from pyopls import OPLS
except ImportError:
    OPLS = None

from fishy.data.classic_loader import load_dataset
from fishy._core.utils import RunContext
import wandb  # New line

from fishy._core.config import TrainingConfig  # Added import
from dataclasses import asdict  # Added import for asdict


class ClassicTrainer:
    """
    Orchestrates training and evaluation for non-deep learning models.
    """

    MODELS = {
        "knn": KNeighborsClassifier,
        "dt": DecisionTreeClassifier,
        "lr": LogisticRegression,
        "lda": LinearDiscriminantAnalysis,
        "nb": GaussianNB,
        "rf": RandomForestClassifier,
        "svm": SVC,
    }

    def __init__(
        self,
        config: TrainingConfig,
        model_name: str,
        dataset_name: str,
        run_id: int = 0,
        file_path: str = None,
    ):
        self.config = config  # Store config
        self.model_name = model_name.lower()
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.file_path = file_path

        self.wandb_run = None
        if self.config.wandb_log:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=asdict(self.config),  # Pass TrainingConfig as W&B config
                reinit=True,  # Important for multiple runs in one script
                group=f"{self.config.dataset}_{self.model_name}",  # Group runs by dataset and model
                job_type="classic_training",
            )

        self.ctx = RunContext(
            dataset=dataset_name,
            method="classic",
            model_name=self.model_name,
            wandb_run=self.wandb_run,
        )
        self.logger = self.ctx.logger

    def run(self):
        self.logger.info(
            f"Starting classic experiment: {self.model_name} on {self.dataset_name}"
        )

        # Load data using the classic loader
        X, y, groups = load_dataset(dataset=self.dataset_name, file_path=self.file_path)

        # OPLS-DA requires a specific pipeline
        if self.model_name == "opls-da":
            return self._run_opls_da(X, y, groups)
        else:
            return self._run_standard_model(X, y, groups)

    def _run_standard_model(self, X, y, groups):
        if self.model_name not in self.MODELS:
            raise ValueError(
                f"Model {self.model_name} not supported. Options: {list(self.MODELS.keys()) + ['opls-da']}"
            )

        model_class = self.MODELS[self.model_name]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 5-fold cross-validation using run_id for unique reproducible splits
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.run_id)

        val_results = []
        train_results = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Pass random_state if supported by the model (e.g. Random Forest, SVM)
            try:
                clf = model_class(random_state=self.run_id)
            except TypeError:
                clf = model_class()
                
            clf.fit(X_train, y_train)

            # Calculate Train Balanced Accuracy
            y_train_pred = clf.predict(X_train)
            train_acc = balanced_accuracy_score(y_train, y_train_pred)
            train_results.append(train_acc)

            # Calculate Validation Balanced Accuracy
            y_pred = clf.predict(X_test)
            val_acc = balanced_accuracy_score(y_test, y_pred)
            val_results.append(val_acc)

            self.ctx.log_metric(fold, {
                "epoch/train_balanced_accuracy": train_acc,
                "epoch/val_balanced_accuracy": val_acc
            })
            self.logger.info(f"Fold {fold}: Train BA = {train_acc:.4f}, Val BA = {val_acc:.4f}")

            # Advanced Visualizations for W&B (last fold)
            if fold == 5 and self.ctx.wandb_run:
                # Attempt to get class names from the dataset if available, else use unique labels
                unique_labels = np.unique(y)
                class_names = [str(int(c)) for c in unique_labels]

                # Get probabilities if the classifier supports it
                y_probs = None
                if hasattr(clf, "predict_proba"):
                    y_probs = clf.predict_proba(X_test)

                # 1. Log Summary Charts
                if y_probs is not None:
                    self.ctx.log_summary_charts(y_test, y_probs, class_names)

                # 2. Log Prediction Table with Spectral Plots
                # Use X_test which are the spectra for this fold
                self.ctx.log_prediction_table(
                    spectra=X_test,
                    preds=y_pred.astype(int),
                    targets=y_test.astype(int),
                    probs=(
                        y_probs
                        if y_probs is not None
                        else np.eye(len(class_names))[y_pred.astype(int)]
                    ),
                    class_names=class_names,
                    table_name="val_predictions_samples_last_fold",
                )

        avg_val_acc = np.mean(val_results)
        std_val_acc = np.std(val_results)
        avg_train_acc = np.mean(train_results)
        
        stats = {
            "train_balanced_accuracy": avg_train_acc,
            "val_balanced_accuracy": avg_val_acc,
            "val_balanced_accuracy_std": std_val_acc
        }
        
        self.ctx.save_results(
            {
                "fold_accuracies": val_results,
                "stats": stats
            }
        )
        self.logger.info(
            f"Finished {self.model_name}. Average Val Balanced Accuracy: {avg_val_acc:.4f} ± {std_val_acc:.4f}"
        )
        return stats

    def _run_opls_da(self, X, y, groups):
        if OPLS is None:
            self.logger.error("pyopls not installed. Skipping OPLS-DA.")
            return {}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Use Group K-Fold if groups are meaningful, otherwise standard Stratified
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.run_id)

        val_results = []
        train_results = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # OPLS for feature extraction
            opls = OPLS(n_components=1)
            X_train_opls = opls.fit_transform(X_train, y_train)
            X_test_opls = opls.transform(X_test)

            # LDA on top of OPLS
            clf = LinearDiscriminantAnalysis()
            # Most LDA implementations are deterministic, but we follow standard practice
            clf.fit(X_train_opls, y_train)

            # Calculate Train Balanced Accuracy
            y_train_pred = clf.predict(X_train_opls)
            train_acc = balanced_accuracy_score(y_train, y_train_pred)
            train_results.append(train_acc)

            # Calculate Validation Balanced Accuracy
            y_pred = clf.predict(X_test_opls)
            val_acc = balanced_accuracy_score(y_test, y_pred)
            val_results.append(val_acc)

            self.ctx.log_metric(fold, {
                "epoch/train_balanced_accuracy": train_acc,
                "epoch/val_balanced_accuracy": val_acc
            })
            self.logger.info(f"Fold {fold}: Train BA = {train_acc:.4f}, Val BA = {val_acc:.4f}")

            # Advanced Visualizations for W&B (last fold)
            if fold == 5 and self.ctx.wandb_run:
                unique_labels = np.unique(y)
                class_names = [str(int(c)) for c in unique_labels]
                y_probs = None
                if hasattr(clf, "predict_proba"):
                    y_probs = clf.predict_proba(X_test_opls)

                if y_probs is not None:
                    self.ctx.log_summary_charts(y_test, y_probs, class_names)

                self.ctx.log_prediction_table(
                    spectra=X_test,  # Use original spectral features for plotting
                    preds=y_pred.astype(int),
                    targets=y_test.astype(int),
                    probs=(
                        y_probs
                        if y_probs is not None
                        else np.eye(len(class_names))[y_pred.astype(int)]
                    ),
                    class_names=class_names,
                    table_name="val_predictions_samples_last_fold",
                )

        avg_val_acc = np.mean(val_results)
        std_val_acc = np.std(val_results)
        avg_train_acc = np.mean(train_results)
        
        stats = {
            "train_balanced_accuracy": avg_train_acc,
            "val_balanced_accuracy": avg_val_acc,
            "val_balanced_accuracy_std": std_val_acc
        }
        
        self.ctx.save_results(
            {
                "fold_accuracies": val_results,
                "stats": stats
            }
        )
        self.logger.info(
            f"Finished OPLS-DA. Average Val Balanced Accuracy: {avg_val_acc:.4f} ± {std_val_acc:.4f}"
        )
        return stats


def run_classic_experiment(
    config: TrainingConfig,
    model_name: str,
    dataset_name: str,
    run_id: int = 0,
    file_path: str = None,
):
    trainer = ClassicTrainer(config, model_name, dataset_name, run_id, file_path)
    try:
        return trainer.run()
    finally:
        if trainer.wandb_run:
            trainer.wandb_run.finish()
