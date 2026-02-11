# -*- coding: utf-8 -*-
"""
Tutorial 07: Sequential Transfer Learning
-----------------------------------------
This tutorial shows how to transfer knowledge from one dataset to another
sequentially, using different classes/tasks at each stage.
"""

from pathlib import Path
from fishy.experiments.transfer import run_sequential_transfer_learning

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(PROJECT_ROOT / "data" / "REIMS.xlsx")

def main():
    print("--- Tutorial 07: Sequential Transfer Learning ---")

    # We want to train on 'part' (e.g. skin vs fillet) 
    # and then transfer that knowledge to 'species' (e.g. hoki vs mackerel).
    
    # run_sequential_transfer_learning automates:
    # 1. Loading the source dataset
    # 2. Training the model
    # 3. Swapping the classification head
    # 4. Loading the target dataset
    # 5. Fine-tuning the final model
    
    print("Starting Transfer Learning: [part] -> [species]")
    
    model, history = run_sequential_transfer_learning(
        model_name="transformer",
        transfer_datasets=["part"],
        target_dataset="species",
        num_epochs_transfer=5,   # Short run for example
        num_epochs_finetune=5,
        batch_size=32,
        file_path=DATA_PATH,
        wandb_log=False
    )

    print("\nTransfer Learning complete.")
    print(f"Final accuracy on target dataset: {history['finetune']['species']['val_balanced_acc'][-1]:.2f}%")

if __name__ == "__main__":
    main()
