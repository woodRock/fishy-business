import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
import os
import json
import numpy as np
import argparse

# Add the parent directory to the Python path to import contrastive
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contrastive.main import ContrastiveConfig, main as contrastive_main

# Set up logging for Optuna
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

# Configure logging for the training pipeline
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, encoder_type: str) -> float:
    """Optuna objective function to optimize hyperparameters for a given encoder type."""
    # 1. Suggest hyperparameters for ContrastiveConfig
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    num_epochs = trial.suggest_int("num_epochs", 100, 1000)
    temperature = trial.suggest_float("temperature", 0.05, 0.5, log=True)
    embedding_dim = trial.suggest_categorical("embedding_dim", [128, 256, 512])
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # Encoder-specific hyperparameters
    if encoder_type == "transformer":
        num_layers = trial.suggest_int("num_layers", 1, 6)
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        num_inner_functions = 10  # Default
    elif encoder_type == "cnn":
        num_layers = trial.suggest_int("num_layers", 2, 6)
        num_heads = 8  # Default
        num_inner_functions = 10  # Default
    elif encoder_type == "kan":
        num_layers = trial.suggest_int("num_layers", 1, 5)
        num_inner_functions = trial.suggest_categorical(
            "num_inner_functions", [5, 10, 15]
        )
        num_heads = 8  # Default
    elif encoder_type == "lstm":
        num_layers = trial.suggest_int("num_layers", 1, 3)
        num_heads = 8  # Default
        num_inner_functions = 10  # Default
    elif encoder_type == "rcnn":
        num_layers = 6  # Default
        num_heads = 8  # Default
        num_inner_functions = 10  # Default
    else:
        raise ValueError(f"Unsupported encoder_type: {encoder_type}")

    # Augmentation parameters
    noise_enabled = trial.suggest_categorical("noise_enabled", [True, False])
    shift_enabled = trial.suggest_categorical("shift_enabled", [True, False])
    scale_enabled = trial.suggest_categorical("scale_enabled", [True, False])
    crop_enabled = trial.suggest_categorical("crop_enabled", [True, False])
    flip_enabled = trial.suggest_categorical("flip_enabled", [True, False])
    permutation_enabled = trial.suggest_categorical(
        "permutation_enabled", [True, False]
    )

    noise_level = (
        trial.suggest_float("noise_level", 0.01, 0.2) if noise_enabled else 0.0
    )
    crop_size = trial.suggest_float("crop_size", 0.5, 0.9) if crop_enabled else 1.0

    # 2. Create ContrastiveConfig instance
    config = ContrastiveConfig(
        encoder_type=encoder_type,
        contrastive_method="simclr",
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        temperature=temperature,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_layers=num_layers,
        num_heads=num_heads,
        num_inner_functions=num_inner_functions,
        noise_enabled=noise_enabled,
        shift_enabled=shift_enabled,
        scale_enabled=scale_enabled,
        crop_enabled=crop_enabled,
        flip_enabled=flip_enabled,
        permutation_enabled=permutation_enabled,
        noise_level=noise_level,
        crop_size=crop_size,
        input_dim=2080,
        num_runs=3,
        patience=100,
        weight_decay=1e-6,
    )

    # 3. Run the main training function
    stats = contrastive_main(config)

    # 4. Return the metric to optimize
    if "val_accuracy" in stats and "mean" in stats["val_accuracy"]:
        return -stats["val_accuracy"]["mean"]
    else:
        return float("inf")


def main():
    """Main function to run the Optuna study."""
    parser = argparse.ArgumentParser(
        description="Run Optuna optimization for SimCLR with a specified encoder."
    )
    parser.add_argument(
        "encoder_type",
        type=str,
        choices=["cnn", "kan", "lstm", "rcnn", "transformer"],
        help="The encoder type to optimize.",
    )
    parser.add_argument(
        "--n_trials", type=int, default=10, help="Number of Optuna trials."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout for the Optuna study in seconds.",
    )
    args = parser.parse_args()

    encoder_type = args.encoder_type

    # Create directories if they don't exist
    os.makedirs("optuna_logs_contrastive", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Create and run the Optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name=f"contrastive_simclr_{encoder_type}_optimization",
    )
    study.optimize(
        lambda trial: objective(trial, encoder_type),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    print(f"\nOptimization for {encoder_type} finished.")
    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Value (negative val_accuracy): {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Re-run with best parameters to get final stats
    best_params = best_trial.params
    noise_enabled = best_params.get("noise_enabled", False)
    crop_enabled = best_params.get("crop_enabled", False)

    config = ContrastiveConfig(
        encoder_type=encoder_type,
        contrastive_method="simclr",
        learning_rate=best_params["learning_rate"],
        batch_size=best_params["batch_size"],
        num_epochs=best_params["num_epochs"],
        temperature=best_params["temperature"],
        embedding_dim=best_params["embedding_dim"],
        hidden_dim=best_params["hidden_dim"],
        dropout=best_params["dropout"],
        num_layers=best_params.get("num_layers", 6),
        num_heads=best_params.get("num_heads", 8),
        num_inner_functions=best_params.get("num_inner_functions", 10),
        noise_enabled=noise_enabled,
        shift_enabled=best_params.get("shift_enabled", False),
        scale_enabled=best_params.get("scale_enabled", False),
        crop_enabled=crop_enabled,
        flip_enabled=best_params.get("flip_enabled", False),
        permutation_enabled=best_params.get("permutation_enabled", False),
        noise_level=best_params.get("noise_level", 0.0) if noise_enabled else 0.0,
        crop_size=best_params.get("crop_size", 1.0) if crop_enabled else 1.0,
        input_dim=2080,
        num_runs=3,
        patience=100,
        weight_decay=1e-6,
    )

    final_stats = contrastive_main(config)

    # Construct the results dictionary
    results_to_save = {
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
        "stats": {
            "train_loss": final_stats.get("train_loss", {}),
            "train_accuracy": final_stats.get("train_accuracy", {}),
            "val_loss": final_stats.get("val_loss", {}),
            "val_accuracy": final_stats.get("val_accuracy", {}),
            "epoch": final_stats.get("epoch", {}),
            "test_loss": final_stats.get("test_loss", 0.0),
            "test_accuracy": final_stats.get("test_accuracy", 0.0),
        },
        "folds": final_stats.get("folds", []),
    }

    # Save the best hyperparameters
    config_path = os.path.join("optuna_logs_contrastive", f"{encoder_type}.config")
    with open(config_path, "w") as f:
        config_to_save = {
            "hyperparameters": best_trial.params,
            "validation_score": -best_trial.value,
        }
        json.dump(config_to_save, f, indent=4)
    logger.info(f"Best hyperparameters for {encoder_type} saved to {config_path}")

    # Save detailed statistics
    results_path = os.path.join("results", f"stats_simclr_{encoder_type}.json")
    with open(results_path, "w") as f:
        json.dump(results_to_save, f, indent=4)
    logger.info(f"Final statistics for {encoder_type} saved to {results_path}")


if __name__ == "__main__":
    main()
