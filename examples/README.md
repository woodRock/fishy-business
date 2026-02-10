# Fishy Business Library Examples

This directory contains examples of how to use the `fishy` library programmatically.

## Overview

1.  **`01_programmatic_training.py`**: Shows how to use `ModelTrainer` and `TrainingConfig` to run experiments in Python instead of the CLI.
2.  **`02_data_module_exploration.py`**: Demonstrates the `DataModule` for loading, filtering, and inspecting metadata (classes, dimensions) from the `datasets.yaml` configuration.
3.  **`03_low_level_trainer.py`**: Explains how to use the core `Trainer` class with any standard PyTorch model.
4.  **`04_pretraining_and_transfer.py`**: Covers advanced research workflows including self-supervised pre-training tasks and sequential transfer learning.

## Configuration-Driven Design

All examples leverage the new configuration-driven architecture. This means:
- Model defaults (hidden dimensions, etc.) are automatically loaded from `fishy/configs/models.yaml`.
- Data filtering rules are defined in `fishy/configs/datasets.yaml`.
- You only need to provide parameters you wish to override.

## Running Examples

From the project root:
```bash
python3 examples/01_programmatic_training.py
```
