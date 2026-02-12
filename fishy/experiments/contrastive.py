# -*- coding: utf-8 -*-
"""
Contrastive learning experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import numpy as np

from fishy._core.config import TrainingConfig
from fishy._core.utils import RunContext, get_device
from fishy.data.module import create_data_module
from fishy.data.datasets import SiameseDataset
from fishy._core.config_loader import load_config
from fishy._core.factory import create_model, get_model_class

from fishy.models.contrastive.simclr import SimCLRModel, SimCLRLoss
from fishy.models.contrastive.simsiam import SimSiamModel, SimSiamLoss
from fishy.models.contrastive.moco import MoCoModel, MoCoLoss
from fishy.models.contrastive.byol import BYOLModel, BYOLLoss
from fishy.models.contrastive.barlow_twins import BarlowTwinsModel, BarlowTwinsLoss

logger = logging.getLogger(__name__)


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive learning."""

    contrastive_method: str = "simclr"
    dataset: str = "species"
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    file_path: str = "data/REIMS.xlsx"
    encoder_type: str = "dense"
    embedding_dim: int = 128
    projection_dim: int = 128
    temperature: float = 0.55
    moco_k: int = 4096
    moco_m: float = 0.999
    moco_t: float = 0.07
    byol_m: float = 0.996
    barlow_lambda: float = 5e-3
    wandb_project: Optional[str] = "fishy-business"
    wandb_entity: Optional[str] = "victoria-university-of-wellington"
    wandb_log: bool = False


class ContrastiveTrainer:
    """Handles contrastive learning training."""

    def __init__(
        self,
        config: ContrastiveConfig,
        wandb_run: Optional[Any] = None,
        ctx: Optional[RunContext] = None,
    ) -> None:
        self.config = config
        self.wandb_run = wandb_run
        if self.wandb_run is None and self.config.wandb_log:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=asdict(self.config),
                reinit=True,
                group=f"{self.config.dataset}_{self.config.contrastive_method}",
                job_type="contrastive_training",
            )
        self.ctx = (
            ctx
            if ctx
            else RunContext(
                dataset=config.dataset,
                method="contrastive",
                model_name=config.contrastive_method,
                wandb_run=self.wandb_run,
            )
        )
        self.logger = self.ctx.logger
        self.device = get_device()
        self.metrics = {}

    def setup(self) -> None:
        self.data_module = create_data_module(
            file_path=self.config.file_path,
            dataset_name=self.config.dataset,
            batch_size=self.config.batch_size,
        )
        self.data_module.setup()
        self.input_dim = self.data_module.get_input_dim()
        t_cfg = TrainingConfig(
            model=self.config.encoder_type,
            hidden_dimension=self.config.embedding_dim,
            num_layers=4,
            num_heads=4,
        )
        encoder = create_model(t_cfg, self.input_dim, self.config.embedding_dim).to(
            self.device
        )
        method = self.config.contrastive_method.lower()
        contrastive_cfg = load_config("models")["contrastive_models"]
        if method not in contrastive_cfg:
            raise ValueError(f"Unsupported method: {method}")
        info = contrastive_cfg[method]
        model_class = get_model_class(info["model"])
        loss_class = get_model_class(info["loss"])

        # Unified instantiation for all contrastive models
        self.model = model_class(
            backbone=encoder,
            embedding_dim=self.config.embedding_dim,
            projection_dim=self.config.projection_dim,
            dropout=0.1,  # Standardized
        ).to(self.device)

        # Standardized loss instantiation
        if method == "simclr":
            self.criterion = loss_class(temperature=self.config.temperature)
        elif method == "barlow_twins":
            self.criterion = loss_class(lambda_param=self.config.barlow_lambda)
        else:
            self.criterion = loss_class()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        full_samples, full_labels = self.data_module.get_numpy_data()
        self.siamese_dataset = SiameseDataset(full_samples, full_labels)
        self.train_loader = DataLoader(
            self.siamese_dataset, batch_size=self.config.batch_size, shuffle=True
        )

    def train(self) -> None:
        self.model.train()
        history = {"loss": []}
        from rich.progress import track

        for epoch in track(
            range(self.config.num_epochs), description="Contrastive Training..."
        ):
            total_loss = 0
            for batch in self.train_loader:
                # Handle flexible batch formats
                if len(batch) >= 2:
                    x1, x2 = batch[0], batch[1]
                else:
                    x1 = batch[0]
                    x2 = x1.clone()

                x1, x2 = x1.to(self.device), x2.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x1, x2)
                loss = (
                    self.criterion(*outputs)
                    if isinstance(outputs, tuple)
                    else self.criterion(outputs)
                )
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            history["loss"].append(total_loss / len(self.train_loader))

        self.metrics = {
            "history": history,
            "val_loss": history["loss"][-1] if history["loss"] else 0,
        }
        self.ctx.save_results(self.metrics)

        if self.ctx.wandb_run:
            self.log_contrastive_visualizations()

    def log_contrastive_visualizations(self) -> None:
        table = wandb.Table(columns=["id", "pair_1", "pair_2", "relationship"])
        import matplotlib.pyplot as plt

        for i in range(min(len(self.siamese_dataset), 20)):
            x1, x2, label = self.siamese_dataset[i]
            plt.figure(figsize=(4, 3))
            plt.plot(x1.numpy())
            plt.axis("off")
            img1 = wandb.Image(plt)
            plt.close()
            plt.figure(figsize=(4, 3))
            plt.plot(x2.numpy())
            plt.axis("off")
            img2 = wandb.Image(plt)
            plt.close()
            table.add_data(
                i, img1, img2, "Same Class" if label == 1 else "Different Class"
            )
        self.ctx.wandb_run.log({"contrastive_pairs_sample": table}, commit=False)


def run_contrastive_experiment(config, wandb_run=None, ctx=None):
    started_wandb = False
    if wandb_run is None and config.wandb_log:
        started_wandb = True
    trainer = ContrastiveTrainer(config, wandb_run=wandb_run, ctx=ctx)
    try:
        trainer.setup()
        trainer.train()
        return trainer.metrics if trainer.metrics is not None else {}
    finally:
        if started_wandb and trainer.wandb_run:
            trainer.wandb_run.finish()
