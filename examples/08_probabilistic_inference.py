# -*- coding: utf-8 -*-
"""
Tutorial 08: Probabilistic Inference and Uncertainty
----------------------------------------------------
This tutorial demonstrates how to use Bayesian models like Gaussian Processes
to get not just predictions, but also uncertainty estimates.
"""

from fishy._core.config import TrainingConfig
from fishy.experiments.classic_training import run_sklearn_experiment
from fishy.data.module import create_data_module
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    print("--- Tutorial 08: Probabilistic Inference ---")

    # 1. Configure a Probabilistic Model
    # We'll use Gaussian Process (GP) which is registered in `models.yaml`.
    config = TrainingConfig(
        model="gp",
        dataset="species",
        k_folds=2,  # Fast for example
    )

    # 2. Run the experiment
    print(f"Training {config.model} with uncertainty estimation...")
    stats = run_sklearn_experiment(config, "gp", "species")

    print(f"\nGP Mean Accuracy: {stats['val_balanced_accuracy']:.4f}")

    # 3. Manual Uncertainty Inspection
    # Let's see how we can get uncertainty from the model directly.
    dm = create_data_module("species")
    dm.setup()
    X, y = dm.get_numpy_data(labels_as_indices=True)

    # SHUFFLE to ensure we get both classes in a small subset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=50, stratify=y, random_state=42
    )

    # Instantiate the GP class (Scikit-learn wrapper)
    from fishy.models.probabilistic.gp import GP

    model = GP()

    # Fit on the shuffled subset
    model.fit(X_train, y_train)

    # Get uncertainty (1.0 - max_prob)
    uncertainty = model.get_uncertainty(X_test[:5])
    preds = model.predict(X_test[:5])

    print("\nPredictions with Uncertainty (Test Subset):")
    for i, (p, u) in enumerate(zip(preds, uncertainty)):
        print(f"  Sample {i}: Pred={p}, Uncertainty={u:.4f}")


if __name__ == "__main__":
    main()
