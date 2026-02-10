# -*- coding: utf-8 -*-
"""
Example 04: Pre-training and Transfer Learning
----------------------------------------------
This script demonstrates how to configure self-supervised pre-training 
and transfer learning.
"""

from fishy._core.config import TrainingConfig
from fishy.experiments.deep_training import ModelTrainer
from fishy.experiments.transfer import run_sequential_transfer_learning
from pathlib import Path

# Set up the path to your data
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(PROJECT_ROOT / "data" / "REIMS.xlsx")

def run_pretrain_example():
    print("--- Running Pre-training Example ---")
    config = TrainingConfig(
        file_path=DATA_PATH,
        model="transformer",
        epochs=2,
        # Enable specific self-supervised tasks
        masked_spectra_modelling=True,
        spectrum_denoising_autoencoding=True
    )
    
    trainer = ModelTrainer(config)
    pre_trained_model = trainer.pre_train()
    print("Pre-training complete.")
    return pre_trained_model

def run_transfer_example():
    print("
--- Running Sequential Transfer Example ---")
    # Transfer from 'part' to 'species'
    run_sequential_transfer_learning(
        model_name="transformer",
        transfer_datasets=["part"],
        target_dataset="species",
        num_epochs_transfer=2,
        num_epochs_finetune=2,
        file_path=DATA_PATH
    )

if __name__ == "__main__":
    run_pretrain_example()
    run_transfer_example()
