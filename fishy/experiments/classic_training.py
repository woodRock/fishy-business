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

    def __init__(self, model_name: str, dataset_name: str, run_id: int = 0, file_path: str = None):
        self.model_name = model_name.lower()
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.file_path = file_path
        self.ctx = RunContext(experiment_name=f"classic_{self.model_name}_{dataset_name}", run_id=run_id)
        self.logger = self.ctx.logger

    def run(self):
        self.logger.info(f"Starting classic experiment: {self.model_name} on {self.dataset_name}")
        
        # Load data using the classic loader
        X, y, groups = load_dataset(dataset=self.dataset_name, file_path=self.file_path)
        
        # OPLS-DA requires a specific pipeline
        if self.model_name == "opls-da":
            self._run_opls_da(X, y, groups)
        else:
            self._run_standard_model(X, y, groups)

    def _run_standard_model(self, X, y, groups):
        if self.model_name not in self.MODELS:
            raise ValueError(f"Model {self.model_name} not supported. Options: {list(self.MODELS.keys()) + ['opls-da']}")
        
        model_class = self.MODELS[self.model_name]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 5-fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        results = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            clf = model_class()
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            acc = balanced_accuracy_score(y_test, y_pred)
            
            results.append(acc)
            self.ctx.log_metric(fold, {"balanced_accuracy": acc})
            self.logger.info(f"Fold {fold}: Balanced Accuracy = {acc:.4f}")

        avg_acc = np.mean(results)
        std_acc = np.std(results)
        self.ctx.save_results({"fold_accuracies": results, "mean_accuracy": avg_acc, "std_accuracy": std_acc})
        self.logger.info(f"Finished {self.model_name}. Average Balanced Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")

    def _run_opls_da(self, X, y, groups):
        if OPLS is None:
            self.logger.error("pyopls not installed. Skipping OPLS-DA.")
            return

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Use Group K-Fold if groups are meaningful, otherwise standard Stratified
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        results = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # OPLS for feature extraction
            opls = OPLS(n_components=1)
            X_train_opls = opls.fit_transform(X_train, y_train)
            X_test_opls = opls.transform(X_test)
            
            # LDA on top of OPLS
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train_opls, y_train)
            
            y_pred = clf.predict(X_test_opls)
            acc = balanced_accuracy_score(y_test, y_pred)
            
            results.append(acc)
            self.ctx.log_metric(fold, {"balanced_accuracy": acc})
            self.logger.info(f"Fold {fold}: Balanced Accuracy = {acc:.4f}")

        avg_acc = np.mean(results)
        std_acc = np.std(results)
        self.ctx.save_results({"fold_accuracies": results, "mean_accuracy": avg_acc, "std_accuracy": std_acc})
        self.logger.info(f"Finished OPLS-DA. Average Balanced Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")

def run_classic_experiment(model_name: str, dataset_name: str, run_id: int = 0, file_path: str = None):
    trainer = ClassicTrainer(model_name, dataset_name, run_id, file_path)
    trainer.run()
