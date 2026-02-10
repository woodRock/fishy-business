# -*- coding: utf-8 -*-
"""
Example 01: Programmatic Training
---------------------------------
This script demonstrates how to use the high-level ModelTrainer to run an experiment 
programmatically instead of using the CLI.
"""

import os
from pathlib import Path
from fishy._core.config import TrainingConfig
from fishy.experiments.deep_training import ModelTrainer

# Set up the path to your data
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(PROJECT_ROOT / "data" / "REIMS.xlsx")

def main():
    # 1. Define your configuration
    # Note: Model-specific defaults (like hidden_dimension for transformer) 
    # will be automatically loaded if not provided here.
    config = TrainingConfig(
        file_path=DATA_PATH,
        dataset="species",
        model="transformer",
        epochs=5,           # Short run for example
        batch_size=8,
        k_folds=3,          # Cross-validation
        wandb_log=False     # Disable logging for this example
    )

    print(f"Starting experiment: {config.model} on {config.dataset}")

    # 2. Initialize the Trainer
    trainer = ModelTrainer(config)

    # 3. Optional: Pre-training
    # If any pre-training tasks were enabled in config, we would run:
    # pre_trained_model = trainer.pre_train()
    pre_trained_model = None

    # 4. Run Training (Fine-tuning)
    results = trainer.train(pre_trained_model)

    print("
Training Results:")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
