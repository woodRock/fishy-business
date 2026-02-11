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
from fishy.data.module import create_data_module
from fishy._core.utils import RunContext
import wandb
from dataclasses import asdict

logger = logging.getLogger(__name__)

def run_gp_experiment(
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
    Runs a Genetic Programming experiment using the standardized GP class.
    """
    wandb_run = None
    if wandb_log:
        wandb_run = wandb.init(
            project=wandb_project, entity=wandb_entity,
            config={"dataset": dataset, "gens": generations, "pop": population, "run": run},
            reinit=True, group=f"{dataset}_evolutionary", job_type="evolutionary_training",
        )

    ctx = RunContext(dataset=dataset, method="evolutionary", model_name="gp_algorithm", wandb_run=wandb_run)
    
    # Standardized Data Loading
    data_module = create_data_module(dataset_name=dataset, file_path=data_file_path)
    data_module.setup()
    X, y = data_module.get_numpy_data(labels_as_indices=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=run)
    fold_metrics = []

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = GP(generations=generations, population_size=population, random_state=run)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        val_acc = balanced_accuracy_score(y_test, y_pred)
        
        train_acc = balanced_accuracy_score(y_train, model.predict(X_train))
        fold_metrics.append({"train": train_acc, "val": val_acc})
        ctx.log_metric(fold_idx + 1, {"epoch/train_balanced_accuracy": train_acc, "epoch/val_balanced_accuracy": val_acc})

    stats = {
        "train_balanced_accuracy": np.mean([m["train"] for m in fold_metrics]),
        "val_balanced_accuracy": np.mean([m["val"] for m in fold_metrics])
    }
    ctx.save_results({"folds": fold_metrics, "stats": stats})
    if wandb_run: wandb_run.finish()
    return stats
