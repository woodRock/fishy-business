# -*- coding: utf-8 -*-
"""
Tutorial 01: Getting Started
----------------------------
This tutorial demonstrates the simplest way to run a training experiment
using the high-level `run_unified_training` interface.
"""

from pathlib import Path
from fishy._core.config import TrainingConfig
from fishy.experiments.unified_trainer import run_unified_training

# Set up the path to your data
# By default, the project expects data in the 'data/' folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(PROJECT_ROOT / "data" / "REIMS.xlsx")

def main():
    print("--- Tutorial 01: Getting Started ---")

    # 1. Define a minimal configuration
    # We specify the model, dataset, and path to the data file.
    # Other parameters will use defaults from TrainingConfig or model registries.
    config = TrainingConfig(
        model="transformer",
        dataset="species",
        file_path=DATA_PATH,
        epochs=5,           # Short run for demonstration
        batch_size=32,
        wandb_log=False     # We'll disable Weights & Biases for now
    )

    print(f"Launching a {config.model} training on the {config.dataset} dataset...")

    # 2. Run the experiment
    # run_unified_training is the universal entry point that handles
    # data loading, model creation, and the training loop.
    results = run_unified_training(config)

    # 3. Inspect the results
    print("\nTraining complete! Summary of results:")
    if isinstance(results, dict):
        # results contains metrics like balanced_accuracy, train_loss, etc.
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
