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
from fishy.cli.main import display_final_summary

# Set up the path to your data
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    print("--- Tutorial 01: Getting Started ---")

    # 1. Define a minimal configuration
    config = TrainingConfig(
        model="transformer",
        dataset="species",
        epochs=5,
        batch_size=32,
        wandb_log=False,
    )

    print(f"Launching a {config.model} training on the {config.dataset} dataset...")

    # 2. Run the experiment
    results = run_unified_training(config)

    # 3. Inspect the results using the beautiful summary table
    display_final_summary(results)


if __name__ == "__main__":
    main()
