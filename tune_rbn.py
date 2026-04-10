#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter optimization for RBN using Optuna.
Gracefully handles OOM (Out Of Memory) errors for CUDA and MPS.
"""

import os
import torch
import optuna
import logging
import gc
from typing import Dict, Any

from fishy._core.config import TrainingConfig
from fishy.experiments.unified_trainer import run_unified_training
from fishy._core.utils import console, set_seed

# Suppress noisy logs
logging.getLogger("fishy").setLevel(logging.ERROR)
os.environ["WANDB_SILENT"] = "true"


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for RBN hyperparameter tuning."""

    # 1. Suggest Hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    num_heads = trial.suggest_categorical("num_heads", [1, 2, 4, 8])

    # Ensure hidden_dim is divisible by num_heads
    if hidden_dim % num_heads != 0:
        # Round num_heads down to the nearest divisor or just skip this combination
        # For simplicity in this script, we'll force a valid num_heads
        divisors = [d for d in [1, 2, 4, 8] if hidden_dim % d == 0]
        num_heads = max(divisors)

    top_k = trial.suggest_categorical("top_k", [100, 200, 300, 400, 500])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    binding_type = trial.suggest_categorical(
        "binding_type", ["hadamard", "outer_product"]
    )

    # 2. Configure Training
    config = TrainingConfig(
        model="rbn",
        dataset="species",
        epochs=30,  # Shorter runs for tuning
        batch_size=batch_size,
        learning_rate=lr,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        top_k=top_k,
        binding_type=binding_type,
        use_checkpointing=True,  # Enable checkpointing for tuning
        rbn_chunk_size=64,
        run=42,  # Fixed seed for reproducibility across trials
        wandb_log=False,
    )

    set_seed(42)

    # 3. Execute Training with OOM Handling
    try:
        results = run_unified_training(config)

        # We want to maximize Balanced Accuracy
        # run_unified_training returns val_balanced_accuracy in the final results
        acc = results.get("val_balanced_accuracy", 0.0)

        # If accuracy is 0, it might have failed silently or been a poor model
        return acc

    except RuntimeError as e:
        # Check for OOM strings in the error message
        error_msg = str(e).lower()
        is_oom = any(
            s in error_msg
            for s in ["out of memory", "allocat", "cuda error: out of memory", "mps"]
        )

        if is_oom:
            console.print(
                f"[yellow]Trial {trial.number} failed due to OOM: {hidden_dim=}, {num_layers=}, {top_k=}. Skipping...[/]"
            )
            # Return a very low score so Optuna avoids this region
            return 0.0
        else:
            # Re-raise other runtime errors
            console.print(f"[red]Trial {trial.number} failed with RuntimeError: {e}[/]")
            return 0.0
    except Exception as e:
        console.print(f"[red]Trial {trial.number} failed with unexpected error: {e}[/]")
        return 0.0
    finally:
        # Forced cleanup after every trial
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()


def run_study(n_trials: int = 20):
    """Creates and runs the Optuna study."""
    console.print(
        f"[bold green]Starting Optuna study for RBN on species dataset ({n_trials} trials)...[/]"
    )

    study = optuna.create_study(
        direction="maximize",
        study_name="rbn_species_tuning",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    console.print("\n[bold green]Study Complete![/]")
    console.print(f"Best trial value: {study.best_value:.4f}")
    console.print("Best hyperparameters:")
    for key, value in study.best_params.items():
        console.print(f"  {key}: {value}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-trials", type=int, default=20)
    args = parser.parse_args()

    run_study(n_trials=args.n_trials)
