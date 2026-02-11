# -*- coding: utf-8 -*-
"""
Tutorial 06: Self-Supervised Pre-training
-----------------------------------------
This tutorial demonstrates how to use unlabeled or semi-labeled data
to pre-train a model using various self-supervised tasks.
"""

from pathlib import Path
from fishy._core.config import TrainingConfig
from fishy.experiments.deep_training import ModelTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(PROJECT_ROOT / "data" / "REIMS.xlsx")


def main():
    print("--- Tutorial 06: Self-Supervised Pre-training ---")

    # 1. Define pre-training tasks in the config
    # Here we enable Masked Spectra Modelling (MSM) and Denoising (SDA).
    config = TrainingConfig(
        model="transformer",
        dataset="species",
        file_path=DATA_PATH,
        epochs=5,
        masked_spectra_modelling=True,
        spectrum_denoising_autoencoding=True,
        wandb_log=False,
    )

    # 2. Initialize the high-level ModelTrainer
    trainer = ModelTrainer(config)

    # 3. Run the pre-training phase
    # This will sequentially execute each enabled task, chaining the weights.
    print(f"Starting pre-training tasks for {config.model}...")
    pre_trained_model = trainer.pre_train()

    if pre_trained_model:
        print("\nPre-training successful! Model is now ready for fine-tuning.")

        # 4. Optional: Proceed to fine-tuning with the learned weights
        # results = trainer.train(pre_trained_model)
        # print(f"Fine-tuned Accuracy: {results.get('balanced_accuracy', 0):.4f}")


if __name__ == "__main__":
    main()
