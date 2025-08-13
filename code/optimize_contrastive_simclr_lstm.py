import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
import os

# Add the parent directory to the Python path to import contrastive
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contrastive.main import (
    ContrastiveConfig,
    main as contrastive_main,
)  # Import main as contrastive_main to avoid name collision
from contrastive.main import ENCODER_REGISTRY, CONTRASTIVE_MODEL_REGISTRY  # For choices

# Set up logging for Optuna
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

# Configure logging for the training pipeline
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Define the objective function for Optuna
def objective(trial: optuna.Trial) -> float:
    # 1. Suggest hyperparameters for ContrastiveConfig
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    num_epochs = trial.suggest_int("num_epochs", 100, 1000)
    temperature = trial.suggest_float("temperature", 0.05, 0.5, log=True)
    embedding_dim = trial.suggest_categorical("embedding_dim", [128, 256, 512])
    hidden_dim = trial.suggest_categorical(
        "hidden_dim", [128, 256, 512]
    )  # For encoder internal layers
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    num_layers = trial.suggest_int("num_layers", 1, 3)  # LSTM specific

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
        encoder_type="lstm",  # Optimizing for LSTM
        contrastive_method="simclr",  # Optimizing for SimCLR
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        temperature=temperature,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_layers=num_layers,
        # Augmentation parameters
        noise_enabled=noise_enabled,
        shift_enabled=shift_enabled,
        scale_enabled=scale_enabled,
        crop_enabled=crop_enabled,
        flip_enabled=flip_enabled,
        permutation_enabled=permutation_enabled,
        noise_level=noise_level,
        crop_size=crop_size,
        # Fixed parameters for now
        input_dim=2080,  # Based on REIMS.xlsx
        num_runs=3,  # Number of CV folds
        patience=100,
        weight_decay=1e-6,
        num_heads=8,  # Only relevant for Transformer, but required by ContrastiveConfig
    )

    # 3. Call contrastive.main.main(config)
    # This function now returns the aggregated stats dictionary
    stats = contrastive_main(config)

    # 4. Return the metric to optimize (e.g., negative validation accuracy)
    # We want to maximize val_accuracy, so we return its negative
    if "val_accuracy" in stats and "mean" in stats["val_accuracy"]:
        return -stats["val_accuracy"]["mean"]
    else:
        # If no valid metrics are returned (e.g., due to errors or no folds completed)
        # return a very large value to indicate a bad trial.
        return float("inf")


if __name__ == "__main__":
    # Create a directory for Optuna logs if it doesn't exist
    os.makedirs("optuna_logs_contrastive", exist_ok=True)

    # Create an Optuna study
    # We want to minimize the negative val_accuracy, which is equivalent to maximizing val_accuracy
    study = optuna.create_study(
        direction="minimize", study_name="contrastive_simclr_lstm_optimization"
    )

    # Run the optimization
    study.optimize(
        objective, n_trials=10, timeout=3600
    )  # Example: 10 trials, max 1 hour

    print("\nOptimization finished.")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (negative val_accuracy): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # You can also save the study results
    # import joblib
    # joblib.dump(study, "contrastive_simclr_lstm_optuna_study.pkl")
