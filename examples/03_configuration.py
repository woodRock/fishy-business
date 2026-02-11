# -*- coding: utf-8 -*-
"""
Tutorial 03: Configuration Management
-------------------------------------
This tutorial covers the `TrainingConfig` and `ExperimentConfig` classes,
which centralize all hyperparameters and experimental settings.
"""

from fishy._core.config import TrainingConfig, ExperimentConfig
from pathlib import Path


def main():
    print("--- Tutorial 03: Configuration Management ---")

    # 1. Single Run Configuration (TrainingConfig)
    # This class holds everything needed for one training session.
    train_cfg = TrainingConfig(
        model="cnn",
        dataset="part",
        epochs=10,
        learning_rate=5e-4,
        batch_size=16,
        data_augmentation=True,  # Enable built-in augmentation
    )

    print("\nTrainingConfig created:")
    print(f"  Model: {train_cfg.model}")
    print(f"  Augmentation: {train_cfg.data_augmentation}")

    # 2. Saving and Loading YAML
    # Configs can be serialized to disk for reproducibility or CLI use.
    yaml_path = "example_config.yaml"
    train_cfg.to_yaml(yaml_path)
    print(f"  Config saved to {yaml_path}")

    # Loading it back
    loaded_cfg = TrainingConfig.from_yaml(yaml_path)
    print(f"  Loaded Model: {loaded_cfg.model}")

    # 3. Batch Configuration (ExperimentConfig)
    # Used to orchestrate multiple models across multiple datasets.
    exp_cfg = ExperimentConfig(
        name="my_first_batch",
        num_runs=5,  # Run each combination 5 times for statistics
        datasets=["species", "oil"],
        models=["cnn", "transformer", "opls-da"],
        benchmark=True,  # Enable performance measuring for all
        overrides={"epochs": 2},  # Force these settings on all runs
    )

    print("\nExperimentConfig created:")
    print(f"  Batch Name: {exp_cfg.name}")
    print(f"  Total combinations: {len(exp_cfg.datasets) * len(exp_cfg.models)}")

    # Clean up
    if Path(yaml_path).exists():
        Path(yaml_path).unlink()


if __name__ == "__main__":
    main()
