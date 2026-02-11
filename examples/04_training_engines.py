# -*- coding: utf-8 -*-
"""
Tutorial 04: Training Engines (High vs. Low Level)
--------------------------------------------------
This tutorial explores different ways to train models, from automated
orchestration to direct control over the training loop.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from fishy.engine.trainer import Trainer
from fishy._core.config import TrainingConfig
from fishy.experiments.unified_trainer import run_unified_training
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(PROJECT_ROOT / "data" / "REIMS.xlsx")


def main():
    print("--- Tutorial 04: Training Engines ---")

    # --- LEVEL 1: Unified Orchestration ---
    # Good for standard experiments and benchmarking.
    print("\n--- Level 1: run_unified_training ---")
    config = TrainingConfig(model="cnn", dataset="oil", file_path=DATA_PATH, epochs=1)
    results = run_unified_training(config)
    print(f"Orchestrated Accuracy: {results.get('val_balanced_accuracy', 0):.4f}")

    # --- LEVEL 2: Direct Trainer usage ---
    # Good for when you have your own model/data but want our optimized loop.
    print("\n--- Level 2: Custom Trainer Control ---")

    # 1. Create a simple PyTorch model
    model = nn.Sequential(nn.Linear(100, 64), nn.ReLU(), nn.Linear(64, 2))

    # 2. Setup your own data
    x = torch.randn(32, 100)
    y = torch.randint(0, 2, (32,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    # 3. Use the Trainer class directly
    trainer = Trainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        device=torch.device("cpu"),
        num_epochs=2,
    )

    # 4. Execute training
    # This gives you raw access to epoch logs and best model states.
    train_res = trainer.train(loader, val_loader=loader)
    print(f"Manual Trainer Accuracy: {train_res['best_accuracy']:.4f}")


if __name__ == "__main__":
    main()
