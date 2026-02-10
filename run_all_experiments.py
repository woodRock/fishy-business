import argparse
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fishy.experiments.classic_training import run_classic_experiment
from fishy.experiments.deep_training import run_training_pipeline
from fishy._core.config import TrainingConfig
from fishy._core.utils import RunContext  # For consistent logging

# Define lists of models and datasets
CLASSIC_MODELS = ["opls-da", "knn", "dt", "lr", "lda", "nb", "rf", "svm"]
DEEP_MODELS = [
    "lstm",
    "vae",
    "kan",
    "cnn",
    "mamba",
    "transformer",
    "transformer_msm",  # Special handling for "Pretrained Transformer (w/ MSM)"
    "moe",
    "ensemble",
]
DATASETS = ["species", "part", "oil", "cross-species"]


def main():
    parser = argparse.ArgumentParser(description="Run all fishy business experiments.")
    parser.add_argument(
        "--wandb-log",
        action="store_true",
        help="Enable Weights & Biases logging for all runs.",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="fishy-business", help="W&B project name."
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="victoria-university-of-wellington",
        help="W&B entity name.",
    )
    args = parser.parse_args()

    # Initialize a logger for the main script
    main_ctx = RunContext(
        dataset="summary", method="orchestrator", model_name="all_experiments"
    )
    logger = main_ctx.logger

    logger.info("Starting all classic experiments...")
    for dataset in DATASETS:
        for model_name in CLASSIC_MODELS:
            logger.info(
                f"Running classic experiment: Model={model_name}, Dataset={dataset}"
            )

            config = TrainingConfig(
                dataset=dataset,
                model=model_name,
                wandb_log=args.wandb_log,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                file_path="data/REIMS.xlsx",
            )

            try:
                run_classic_experiment(
                    config=config, model_name=model_name, dataset_name=dataset
                )
                logger.info(
                    f"Successfully completed classic experiment: Model={model_name}, Dataset={dataset}"
                )
            except Exception as e:
                logger.error(
                    f"Error running classic experiment Model={model_name}, Dataset={dataset}: {e}"
                )

    logger.info("Starting all deep learning experiments...")
    for dataset in DATASETS:
        for model_name in DEEP_MODELS:
            logger.info(
                f"Running deep learning experiment: Model={model_name}, Dataset={dataset}"
            )

            # Initialize TrainingConfig with common defaults
            config = TrainingConfig(
                dataset=dataset,
                model=model_name,
                epochs=10,  # Reduced for quicker demonstration, adjust as needed
                batch_size=32,  # Default to a reasonable batch size
                learning_rate=1e-3,
                run=0,  # Reset run ID for each experiment
                file_path="data/REIMS.xlsx",
                wandb_log=args.wandb_log,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
            )

            # Special handling for "Pretrained Transformer (w/ MSM)"
            if model_name == "transformer_msm":
                config.model = "transformer"
                config.masked_spectra_modelling = True
                logger.info(
                    f"Configuring Transformer with Masked Spectra Modelling pre-training for Dataset={dataset}"
                )

            try:
                run_training_pipeline(config=config)
                logger.info(
                    f"Successfully completed deep learning experiment: Model={model_name}, Dataset={dataset}"
                )
            except Exception as e:
                logger.error(
                    f"Error running deep learning experiment Model={model_name}, Dataset={dataset}: {e}"
                )

    logger.info("All experiments finished.")


if __name__ == "__main__":
    main()
