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
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(PROJECT_ROOT / "data" / "REIMS.xlsx")

def main():
    print("--- Tutorial 08: Probabilistic Inference ---")

    # 1. Configure a Probabilistic Model
    # We'll use Gaussian Process (GP) which is registered in `models.yaml`.
    config = TrainingConfig(
        model="gp", 
        dataset="species", 
        file_path=DATA_PATH,
        k_folds=2  # Fast for example
    )

    # 2. Run the experiment
    print(f"Training {config.model} with uncertainty estimation...")
    stats = run_sklearn_experiment(config, "gp", "species", file_path=DATA_PATH)
    
    print(f"\nGP Mean Accuracy: {stats['val_balanced_accuracy']:.4f}")

    # 3. Manual Uncertainty Inspection
    # Let's see how we can get uncertainty from the model directly.
    dm = create_data_module("species", DATA_PATH)
    dm.setup()
    X, y = dm.get_numpy_data(labels_as_indices=True)

    # Instantiate the GP class
    from fishy.models.probabilistic.gp import GaussianProcess
    model = GaussianProcess(kernel_type="matern")
    
    # Fit on a subset
    model.fit(X[:20], y[:20])
    
    # Get uncertainty (1.0 - max_prob)
    uncertainty = model.get_uncertainty(X[20:25])
    preds = model.predict(X[20:25])

    print("\nPredictions with Uncertainty:")
    for i, (p, u) in enumerate(zip(preds, uncertainty)):
        print(f"  Sample {i}: Pred={p}, Uncertainty={u:.4f}")

if __name__ == "__main__":
    main()
