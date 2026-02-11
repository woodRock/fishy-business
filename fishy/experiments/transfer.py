# -*- coding: utf-8 -*-
"""
Transfer learning module for deep learning models.
Standardized to use DataModule and Trainer patterns.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import wandb

from fishy.engine.trainer import Trainer, DeepEngine
from fishy.data.module import create_data_module
from fishy._core.factory import create_model
from fishy._core.config import TrainingConfig
from fishy._core.utils import RunContext, get_device


def _adapt_trainer_output(trainer_output: Dict[str, Any]) -> Dict[str, List]:
    """Adapts Trainer output to the legacy history format."""
    epoch_metrics = trainer_output["epoch_metrics"]
    return {
        "train_loss": epoch_metrics["train_losses"],
        "val_loss": epoch_metrics["val_losses"],
        "train_acc": [m.get("accuracy", 0) * 100 for m in epoch_metrics["train_metrics"]],
        "val_acc": [m.get("accuracy", 0) * 100 for m in epoch_metrics["val_metrics"]],
        "train_balanced_acc": [m.get("balanced_accuracy", 0) * 100 for m in epoch_metrics["train_metrics"]],
        "val_balanced_acc": [m.get("balanced_accuracy", 0) * 100 for m in epoch_metrics["val_metrics"]],
        "learning_rates": epoch_metrics.get("learning_rates", []),
    }


def run_sequential_transfer_learning(
    model_name: str,
    transfer_datasets: List[str],
    target_dataset: str,
    num_epochs_transfer: int = 10,
    num_epochs_finetune: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    finetune_lr: float = 5e-4,
    device: str = str(get_device()),
    save_intermediate: bool = False,
    val_split: float = 0.2,
    file_path: Optional[str] = None,
    wandb_project: str = "fishy-business",
    wandb_entity: str = "victoria-university-of-wellington",
    wandb_log: bool = False,
    run: int = 0,
    wandb_run: Optional[Any] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Performs sequential transfer learning using standardized DataModules.

    Examples:
        >>> m_name = "transformer"
        >>> isinstance(m_name, str)
        True

    Args:
        model_name (str): Name of the model architecture.
        transfer_datasets (List[str]): List of datasets to pre-train on.
        target_dataset (str): Final dataset to fine-tune on.
        num_epochs_transfer (int): Epochs per transfer phase.
        num_epochs_finetune (int): Epochs for final phase.
        batch_size (int): Batch size.
        learning_rate (float): Initial learning rate.
        finetune_lr (float): Learning rate for fine-tuning.
        device (str): Computation device.
        save_intermediate (bool): Save checkpoints after each phase.
        val_split (float): Fraction of data for validation.
        file_path (str): Path to data file.
        wandb_project (str): W&B project name.
        wandb_entity (str): W&B entity.
        wandb_log (bool): Enable W&B logging.
        run (int): Run identifier/seed.
        wandb_run (Any): Existing W&B run.

    Returns:
        Tuple[nn.Module, Dict[str, Any]]: Trained model and history.
    """
    started_wandb = False
    if wandb_run is None and wandb_log:
        started_wandb = True
        wandb_run = wandb.init(
            project=wandb_project, entity=wandb_entity,
            config={"model": model_name, "transfer": transfer_datasets, "target": target_dataset},
            reinit=True, group=f"{target_dataset}_transfer", job_type="transfer_learning",
        )

    ctx = RunContext(dataset=target_dataset, method="transfer", model_name=model_name, wandb_run=wandb_run)
    logger = ctx.logger

    try:
        # ... (rest of the try block)
        history = {"transfer": {}, "finetune": {}}
        device_obj = get_device()
        
        # 1. Initial Setup to get input dimension
        data_module = create_data_module(file_path=file_path, dataset_name=transfer_datasets[0], batch_size=batch_size)
        data_module.setup()
        input_dim = data_module.get_input_dim()

        config = TrainingConfig(model=model_name, dataset=transfer_datasets[0], run=run, file_path=file_path)
        
        num_classes = data_module.get_num_classes()
        model = create_model(config, input_dim, num_classes).to(device_obj)

        gen = torch.Generator().manual_seed(run)

        # 2. Sequential Transfer
        for dataset_name in transfer_datasets:
            logger.info(f"Transfer phase: {dataset_name}")
            data_module = create_data_module(file_path=file_path, dataset_name=dataset_name, batch_size=batch_size)
            data_module.setup()
            dataset = data_module.get_dataset()

            val_size = int(val_split * len(dataset))
            train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size], generator=gen)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            current_num_classes = data_module.get_num_classes()
            
            # Layer adaptation
            for attr in ["fc_out", "classifier", "fc"]:
                if hasattr(model, attr):
                    layer = getattr(model, attr)
                    if layer.out_features != current_num_classes:
                        setattr(model, attr, nn.Linear(layer.in_features, current_num_classes).to(device_obj))
                    break

            optimizer = AdamW(model.parameters(), lr=learning_rate)
            trainer = Trainer(model=model, criterion=nn.CrossEntropyLoss(), optimizer=optimizer, device=device_obj, num_epochs=num_epochs_transfer, num_classes=current_num_classes, logger=logger, ctx=ctx)
            history["transfer"][dataset_name] = _adapt_trainer_output(trainer.train(train_loader, val_loader))

        # 3. Final Fine-tuning
        logger.info(f"Final Fine-tuning: {target_dataset}")
        data_module = create_data_module(file_path=file_path, dataset_name=target_dataset, batch_size=batch_size)
        data_module.setup()
        target_dataset_obj = data_module.get_dataset()

        val_size = int(val_split * len(target_dataset_obj))
        train_dataset, val_dataset = random_split(target_dataset_obj, [len(target_dataset_obj) - val_size, val_size], generator=gen)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        current_num_classes = data_module.get_num_classes()
        for attr in ["fc_out", "classifier", "fc"]:
            if hasattr(model, attr):
                layer = getattr(model, attr)
                if layer.out_features != current_num_classes:
                    setattr(model, attr, nn.Linear(layer.in_features, current_num_classes).to(device_obj))
                break

        optimizer = AdamW(model.parameters(), lr=finetune_lr)
        trainer = Trainer(model=model, criterion=nn.CrossEntropyLoss(), optimizer=optimizer, device=device_obj, num_epochs=num_epochs_finetune, num_classes=current_num_classes, logger=logger, ctx=ctx)
        history["finetune"][target_dataset] = _adapt_trainer_output(trainer.train(train_loader, val_loader))

        ctx.save_results({"history": history}, filename="transfer_results.json")
        return model, history

    finally:
        if started_wandb and wandb_run: wandb_run.finish()
