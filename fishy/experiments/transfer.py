# -*- coding: utf-8 -*-
"""
Transfer learning module for deep learning models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
)
import seaborn as sns
import copy
import os
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import asdict
import wandb

from fishy.engine.trainer import Trainer  # Use Trainer
from fishy.data.module import create_data_module
from fishy._core.factory import create_model
from fishy._core.config import TrainingConfig
from fishy._core.utils import RunContext


def _adapt_trainer_output(trainer_output: Dict) -> Dict:
    """Adapts Trainer output to the legacy history format used in transfer learning."""
    epoch_metrics = trainer_output["epoch_metrics"]
    history = {
        "train_loss": epoch_metrics["train_losses"],
        "val_loss": epoch_metrics["val_losses"],
        "train_acc": [m.get("accuracy", 0) * 100 for m in epoch_metrics["train_metrics"]],
        "val_acc": [m.get("accuracy", 0) * 100 for m in epoch_metrics["val_metrics"]],
        "train_balanced_acc": [
            m.get("balanced_accuracy", 0) * 100 for m in epoch_metrics["train_metrics"]
        ],
        "val_balanced_acc": [
            m.get("balanced_accuracy", 0) * 100 for m in epoch_metrics["val_metrics"]
        ],
        "learning_rates": epoch_metrics.get("learning_rates", []),
    }
    return history


def run_sequential_transfer_learning(
    model_name: str,
    transfer_datasets: List[str],
    target_dataset: str,
    num_epochs_transfer: int = 10,
    num_epochs_finetune: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    finetune_lr: float = 5e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_intermediate: bool = False,
    val_split: float = 0.2,
    file_path: str = None,
    # New W&B parameters
    wandb_project: Optional[str] = "fishy-business",
    wandb_entity: Optional[str] = "victoria-university-of-wellington",
    wandb_log: bool = False,
    run: int = 0,
):
    """
    Performs sequential transfer learning across multiple datasets.
    """
    wandb_run = None
    if wandb_log:
        # Create a dict for W&B config from function arguments
        wandb_config_dict = {
            "model_name": model_name,
            "transfer_datasets": transfer_datasets,
            "target_dataset": target_dataset,
            "num_epochs_transfer": num_epochs_transfer,
            "num_epochs_finetune": num_epochs_finetune,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "finetune_lr": finetune_lr,
            "save_intermediate": save_intermediate,
            "val_split": val_split,
            "device": device,
            "file_path": file_path,
            "run": run,
        }
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config=wandb_config_dict,
            reinit=True,
            group=f"{target_dataset}_{model_name}_transfer",
            job_type="transfer_learning",
        )

    ctx = RunContext(
        dataset=target_dataset,
        method="transfer",
        model_name=model_name,
        wandb_run=wandb_run,
    )
    logger = ctx.logger

    try:  # Start try block for wandb.finish
        history = {"transfer": {}, "finetune": {}}
        device_obj = torch.device(device)
        data_path = (
            file_path
            if file_path
            else str(
                Path(__file__).resolve().parent.parent.parent / "data" / "REIMS.xlsx"
            )
        )

        logger.info(f"Starting sequential transfer learning for model: {model_name}")
        logger.info(f"Target dataset: {target_dataset}")

        # Initial data module to get dimensions
        data_module = create_data_module(
            file_path=data_path,
            dataset_name=transfer_datasets[0],
            batch_size=batch_size,
        )
        data_module.setup()
        input_dim = data_module.get_input_dim()

        # We need to handle the output_dim changing.
        # Initial model creation
        config = TrainingConfig(
            file_path=data_path,
            model=model_name,
            dataset=transfer_datasets[0],
            run=run,
            output="",
            data_augmentation=False,
            masked_spectra_modelling=False,
            next_spectra_prediction=False,
            next_peak_prediction=False,
            spectrum_denoising_autoencoding=False,
            peak_parameter_regression=False,
            spectrum_segment_reordering=False,
            contrastive_transformation_invariance_learning=False,
            early_stopping=0,
            dropout=0.2,
            label_smoothing=0.1,
            epochs=num_epochs_transfer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            hidden_dimension=128,
            num_layers=4,
            num_heads=4,
            num_augmentations=0,
            noise_level=0.0,
            shift_enabled=False,
            scale_enabled=False,
            k_folds=1,
        )
        ctx.save_config(config)

        # Determine initial num_classes
        from fishy.experiments.deep_training import ModelTrainer

        num_classes = ModelTrainer.N_CLASSES_PER_DATASET.get(transfer_datasets[0], 2)
        model = create_model(config, input_dim, num_classes).to(device_obj)
        logger.info(f"Model {model_name} initialized on {device}")

        # Generator for seeded splits
        gen = torch.Generator().manual_seed(run)

        # Sequential transfer learning
        for i, dataset_name in enumerate(transfer_datasets):
            logger.info(f"Phase {i+1}: Transfer Learning on '{dataset_name}'")

            data_module = create_data_module(
                file_path=data_path,
                dataset_name=dataset_name,
                batch_size=batch_size,
            )
            data_module.setup()
            dataset = data_module.get_dataset()

            val_size = int(val_split * len(dataset))
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size], generator=gen
            )

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            # Adapt output layer if necessary
            current_num_classes = ModelTrainer.N_CLASSES_PER_DATASET.get(
                dataset_name, 2
            )

            output_layer = None
            for attr in ["fc_out", "classifier", "fc"]:
                if hasattr(model, attr):
                    output_layer = getattr(model, attr)
                    layer_name = attr
                    break

            if output_layer and isinstance(output_layer, nn.Linear):
                if output_layer.out_features != current_num_classes:
                    in_features = output_layer.in_features
                    new_layer = nn.Linear(in_features, current_num_classes).to(
                        device_obj
                    )
                    setattr(model, layer_name, new_layer)
                    logger.info(
                        f"Adapted {layer_name} to {current_num_classes} classes"
                    )

            optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

            trainer = Trainer(
                model=model,
                criterion=nn.CrossEntropyLoss(),
                optimizer=optimizer,
                device=device_obj,
                num_epochs=num_epochs_transfer,
                scheduler=scheduler,
                patience=100,  # No early stopping in transfer phase usually, or high
                num_classes=current_num_classes,
                logger=logger,
                ctx=ctx,
            )
            trainer_output = trainer.train(train_loader, val_loader)
            dataset_history = _adapt_trainer_output(trainer_output)
            history["transfer"][dataset_name] = dataset_history

            if save_intermediate:
                torch.save(
                    model.state_dict(),
                    ctx.get_checkpoint_path(f"model_transfer_{dataset_name}.pth"),
                )

        # Fine-tuning
        logger.info(f"Final Phase: Fine-tuning on '{target_dataset}'")
        data_module = create_data_module(
            file_path=data_path,
            dataset_name=target_dataset,
            batch_size=batch_size,
        )
        data_module.setup()
        target_data = data_module.get_dataset()

        val_size = int(val_split * len(target_data))
        train_size = len(target_data) - val_size
        train_dataset, val_dataset = random_split(
            target_data, [train_size, val_size], generator=gen
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        current_num_classes = ModelTrainer.N_CLASSES_PER_DATASET.get(target_dataset, 2)
        # Adapt output layer again
        output_layer = None
        for attr in ["fc_out", "classifier", "fc"]:
            if hasattr(model, attr):
                output_layer = getattr(model, attr)
                layer_name = attr
                break

        if output_layer and isinstance(output_layer, nn.Linear):
            if output_layer.out_features != current_num_classes:
                in_features = output_layer.in_features
                new_layer = nn.Linear(in_features, current_num_classes).to(device_obj)
                setattr(model, layer_name, new_layer)
                logger.info(f"Adapted {layer_name} to {current_num_classes} classes")

        optimizer = AdamW(model.parameters(), lr=finetune_lr, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

        trainer = Trainer(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            device=device_obj,
            num_epochs=num_epochs_finetune,
            scheduler=scheduler,
            patience=20,
            num_classes=current_num_classes,
            logger=logger,
            ctx=ctx,
        )
        trainer_output = trainer.train(train_loader, val_loader)
        finetune_history = _adapt_trainer_output(trainer_output)
        history["finetune"][target_dataset] = finetune_history

        # Final Evaluation
        # Reuse trainer.evaluate()
        eval_trainer = Trainer(
             model=model,
             criterion=nn.CrossEntropyLoss(),
             optimizer=optimizer, # Dummy
             device=device_obj,
             num_epochs=1,
             num_classes=current_num_classes,
        )
        eval_results = eval_trainer.evaluate(val_loader)
        final_acc = eval_results["metrics"]["balanced_accuracy"]
        logger.info(f"Final Balanced Accuracy: {final_acc*100:.2f}%")

        ctx.save_results(
            {"history": history, "final_balanced_accuracy": final_acc},
            filename="transfer_results.json",
        )
        torch.save(
            model.state_dict(), ctx.get_checkpoint_path("final_transfer_model.pth")
        )

        return model, history

    finally:
        if wandb_run:
            wandb_run.finish()


def visualize_transfer_results(history: Dict):
    # Reuse the logic from original transfer_learning.py for plotting if needed
    pass
