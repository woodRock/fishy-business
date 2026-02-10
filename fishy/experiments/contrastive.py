# -*- coding: utf-8 -*-
"""
Contrastive learning experiments module.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
from dataclasses import asdict # New line

from fishy.data.module import create_data_module
from fishy.data.augmentation import AugmentationConfig, DataAugmenter
from fishy.data.contrastive_util import DataConfig, DataPreprocessor, SiameseDataset, BalancedBatchSampler
from fishy.models.contrastive.simclr import SimCLRModel, SimCLRLoss
from fishy.models.contrastive.moco import MoCoModel, MoCoLoss
from fishy.models.contrastive.byol import BYOLModel, BYOLLoss
from fishy.models.contrastive.simsiam import SimSiamModel, SimSiamLoss
from fishy.models.contrastive.barlow_twins import BarlowTwinsModel, BarlowTwinsLoss
from fishy._core.factory import create_model, MODEL_REGISTRY
from fishy._core.config import TrainingConfig
from fishy._core.utils import RunContext
import wandb # Added import

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
    dataset: str = "species" # Added dataset attribute
    # Weights & Biases parameters
    wandb_project: Optional[str] = "fishy-business"
    wandb_entity: Optional[str] = "victoria-university-of-wellington"
    wandb_log: bool = False

class ContrastiveTrainer:
    def __init__(self, config: ContrastiveConfig):
        self.config = config
        self.wandb_run = None
        if self.config.wandb_log:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=asdict(self.config), # Pass ContrastiveConfig as W&B config
                reinit=True, # Important for multiple runs in one script
                group=f"{self.config.dataset}_{self.config.contrastive_method}", # Group runs by dataset and contrastive method
                job_type="contrastive_training"
            )
        self.ctx = RunContext(dataset=config.dataset, method="contrastive", model_name=config.contrastive_method, wandb_run=self.wandb_run)
        self.logger = self.ctx.logger
        self.ctx.save_config(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

    def setup(self):
        # Data Module
        self.data_module = create_data_module(
            file_path=self.config.file_path,
            dataset_name=self.config.dataset, # Fix hardcoded dataset_name
            batch_size=self.config.batch_size,
        )
        self.data_module.setup()
        self.input_dim = self.data_module.get_input_dim()
        
        # Encoder creation
        # We need a TrainingConfig for create_model
        t_cfg = TrainingConfig(
            model=self.config.encoder_type,
            hidden_dimension=self.config.embedding_dim,
            num_layers=4, # Default layers
            num_heads=4   # Default heads
        )
        encoder = create_model(t_cfg, self.input_dim, self.config.embedding_dim).to(self.device)
        
        # Contrastive Model
        if self.config.contrastive_method == "simclr":
            self.model = SimCLRModel(encoder, self.config).to(self.device)
            self.criterion = SimCLRLoss(temperature=self.config.temperature)
        else:
            raise ValueError(f"Unsupported contrastive method: {self.config.contrastive_method}")

        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        
        # Prepare Siamese Dataset for contrastive
        samples = self.data_module.get_dataset().samples.cpu().numpy()
        labels = self.data_module.get_dataset().labels.cpu().numpy()
        self.siamese_dataset = SiameseDataset(samples, labels)
        self.train_loader = DataLoader(
            self.siamese_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )

    def train(self):
        self.model.train()
        history = {"loss": []}
        for epoch in range(self.config.num_epochs):
            total_loss = 0
            for batch_idx, (x1, x2, _) in enumerate(self.train_loader):
                x1, x2 = x1.to(self.device), x2.to(self.device)
                
                self.optimizer.zero_grad()
                z1, z2 = self.model(x1, x2)
                loss = self.criterion(z1, z2)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            history["loss"].append(avg_loss)
            self.ctx.log_metric(epoch + 1, {"loss": avg_loss})
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                self.logger.info(f"Epoch [{epoch+1}/{self.config.num_epochs}], Loss: {avg_loss:.4f}")

        # Save results
        self.ctx.save_results(history, filename="training_history.json")
        torch.save(self.model.state_dict(), self.ctx.get_checkpoint_path("final_model.pth"))
        self.logger.info("Contrastive training finished and model saved.")

def run_contrastive_experiment(config: ContrastiveConfig):
    """
    Orchestrates contrastive learning experiments.
    """
    trainer = ContrastiveTrainer(config)
    try:
        trainer.setup()
        trainer.train()
    finally:
        if trainer.wandb_run:
            trainer.wandb_run.finish()
