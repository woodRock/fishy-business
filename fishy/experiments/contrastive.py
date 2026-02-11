# -*- coding: utf-8 -*-
"""
Contrastive learning experiments module.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Any

from fishy.data.module import create_data_module
from fishy.data import SiameseDataset
from fishy._core.factory import create_model, get_model_class
from fishy._core.config import TrainingConfig
from fishy._core.utils import RunContext, get_device, console
from fishy._core.config_loader import load_config
import wandb


@dataclass
class ContrastiveConfig:
    num_runs: int = 1
    temperature: float = 0.55
    projection_dim: int = 256
    embedding_dim: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    batch_size: int = 16
    num_epochs: int = 100
    input_dim: int = 2080
    encoder_type: str = "transformer"
    contrastive_method: str = "simclr"
    file_path: str = ""
    dropout: float = 0.1
    dataset: str = "species"
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
        if method == "simclr":
            self.model = model_class(encoder, self.config).to(self.device)
            self.criterion = loss_class(temperature=self.config.temperature)
        elif method == "moco":
            self.model = model_class(
                encoder,
                encoder_output_dim=self.config.embedding_dim,
                dim=self.config.projection_dim,
                K=self.config.moco_k,
                m=self.config.moco_m,
                T=self.config.moco_t,
            ).to(self.device)
            self.criterion = loss_class(T=self.config.moco_t)
        elif method == "byol":
            self.model = model_class(
                encoder,
                encoder_output_dim=self.config.embedding_dim,
                projection_dim=self.config.projection_dim,
                m=self.config.byol_m,
            ).to(self.device)
            self.criterion = loss_class()
        elif method == "simsiam":
            self.model = model_class(
                encoder,
                encoder_output_dim=self.config.embedding_dim,
                projection_dim=self.config.projection_dim,
            ).to(self.device)
            self.criterion = loss_class()
        elif method == "barlow_twins":
            self.model = model_class(
                encoder,
                encoder_output_dim=self.config.embedding_dim,
                projection_dim=self.config.projection_dim,
            ).to(self.device)
            self.criterion = loss_class(lambda_param=self.config.barlow_lambda)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        samples, labels = self.data_module.get_numpy_data()
        self.siamese_dataset = SiameseDataset(samples, labels)
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
            for x1, x2, _ in self.train_loader:
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
        self.ctx.save_results(
            {"history": history, "stats": {"final_loss": history["loss"][-1]}}
        )
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
    finally:
        if started_wandb and trainer.wandb_run:
            trainer.wandb_run.finish()
