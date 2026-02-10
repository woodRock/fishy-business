# -*- coding: utf-8 -*-
"""
Example 03: Low-Level Trainer Usage
-----------------------------------
This script shows how to use the core Trainer class directly with a custom 
PyTorch model and standard DataLoaders.
"""

import torch
import torch.nn as nn
from fishy.engine.trainer import Trainer
from fishy.data.module import create_data_module
from pathlib import Path

# Set up the path to your data
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(PROJECT_ROOT / "data" / "REIMS.xlsx")

def main():
    # 1. Prepare data
    dm = create_data_module("species", DATA_PATH, batch_size=8)
    dm.setup()
    train_loader = dm.get_train_dataloader()
    input_dim = dm.get_input_dim()
    num_classes = dm.get_num_classes()

    # 2. Define a simple custom model
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes)
    )

    # 3. Setup standard PyTorch components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cpu") # Use "cuda" if available

    # 4. Use the fishy Trainer
    # This automatically handles early stopping, metric tracking, and logging
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=10,
        patience=5
    )

    print("Starting custom training...")
    results = trainer.train(train_loader)

    print(f"
Best accuracy achieved: {results['best_accuracy']:.4f}")

if __name__ == "__main__":
    main()
