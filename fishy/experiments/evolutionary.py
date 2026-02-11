# -*- coding: utf-8 -*-
"""
Evolutionary experiments module using the consolidated GP engine.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from fishy.models.evolutionary.gp import GP
from fishy.models.evolutionary.ga import GA
from fishy.models.evolutionary.es import ES
from fishy.models.evolutionary.pso import PSO
from fishy.models.evolutionary.eda import EDA
from fishy.data.module import create_data_module
from fishy._core.config_loader import load_config
from fishy._core.factory import get_model_class
from fishy._core.utils import RunContext
import wandb
from dataclasses import asdict

logger = logging.getLogger(__name__)

def run_evolutionary_experiment(
    model_name: str = "gp",
    dataset: str = "species",
    generations: int = 10,
    population: int = 1023,
    run: int = 0,
    data_file_path: Optional[str] = None,
    wandb_project: str = "fishy-business",
    wandb_entity: str = "victoria-university-of-wellington",
    wandb_log: bool = False,
) -> Dict[str, float]:
    """
    Runs an evolutionary experiment using the specified model.
    """
    wandb_run = None
    if wandb_log:
        wandb_run = wandb.init(
            project=wandb_project, entity=wandb_entity,
            config={"dataset": dataset, "model": model_name, "gens": generations, "pop": population, "run": run},
            reinit=True, group=f"{dataset}_evolutionary", job_type="evolutionary_training",
        )

    ctx = RunContext(dataset=dataset, method="evolutionary", model_name=model_name, wandb_run=wandb_run)
    
    # Standardized Data Loading
    data_module = create_data_module(dataset_name=dataset, file_path=data_file_path)
    data_module.setup()
    X, y = data_module.get_numpy_data(labels_as_indices=True)

    # Get model class and default config
    evo_cfg = load_config("models")["evolutionary_models"]
    if model_name not in evo_cfg:
        raise ValueError(f"Evolutionary model {model_name} not found in models.yaml")
    
    entry = evo_cfg[model_name]
    model_path = entry["path"] if isinstance(entry, dict) else entry
    model_class = get_model_class(model_path)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=run)
    fold_metrics = []
    last_fold_info = {}

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Instantiate model with parameters, prioritizing CLI overrides
        if model_name == "gp":
            model = model_class(generations=generations, population_size=population, random_state=run)
        elif model_name == "ga":
            model = model_class(generations=generations, population_size=population, random_state=run)
        elif model_name == "es":
            model = model_class(generations=generations, mu=population // 2, lambda_=population, random_state=run)
        elif model_name == "pso":
            model = model_class(iterations=generations, population_size=population, random_state=run)
        elif model_name == "eda":
            model = model_class(generations=generations, population_size=population, random_state=run)
        else:
            model = model_class()

        model.fit(X_train, y_train)

        # For these weighting models, we might need a classifier to get accurate BCA
        # For now, keep the simple bca check if predict returns labels
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        val_acc = balanced_accuracy_score(y_test, y_pred)
        
        train_acc = balanced_accuracy_score(y_train, model.predict(X_train))
        fold_metrics.append({"train": train_acc, "val": val_acc})
        ctx.log_metric(fold_idx + 1, {"epoch/train_balanced_accuracy": train_acc, "epoch/val_balanced_accuracy": val_acc})

        if fold_idx == 4: # Last fold
            last_fold_info = {
                "labels": y_test,
                "preds": y_pred,
                "probs": y_probs
            }

    stats = {
        "train_balanced_accuracy": np.mean([m["train"] for m in fold_metrics]),
        "val_balanced_accuracy": np.mean([m["val"] for m in fold_metrics]),
        "epoch_metrics": {
            "val_balanced_accuracy": [m["val"] for m in fold_metrics],
            "train_balanced_accuracy": [m["train"] for m in fold_metrics]
        },
        "best_val_predictions": last_fold_info,
        "predictions": last_fold_info
    }
    ctx.save_results({"folds": fold_metrics, "stats": stats})
    if wandb_run: wandb_run.finish()
    return stats
