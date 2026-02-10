# -*- coding: utf-8 -*-
"""
Orchestrator for pre-training tasks.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fishy._core.factory import create_model
from fishy.experiments.pre_training import PreTrainer, PreTrainingConfig
from fishy._core.config import TrainingConfig
from fishy._core.utils import RunContext
from fishy._core.config_loader import load_config

class PreTrainingOrchestrator:
    """
    Handles the orchestration of multiple pre-training tasks.
    Uses external configuration for task definitions.
    """

    def __init__(
        self,
        config: TrainingConfig,
        device: torch.device,
        input_dim: int,
        ctx: RunContext,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.device = device
        self.input_dim = input_dim
        self.ctx = ctx
        self.logger = logger if logger else logging.getLogger(__name__)
        
        # Load tasks from configuration
        self.task_configs = load_config("pre_training")["tasks"]

    def run_all(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Optional[nn.Module]:
        """Runs all enabled pre-training tasks sequentially."""
        enabled_tasks = [
            task
            for task in self.task_configs
            if getattr(self.config, task["name"], False)
        ]
        
        if not enabled_tasks:
            self.logger.info("No pre-training tasks enabled.")
            return None

        self.logger.info(f"Enabled pre-training tasks: {', '.join(t['name'] for t in enabled_tasks)}")

        pre_train_cfg = PreTrainingConfig(
            num_epochs=self.config.epochs,
            file_path=self.config.file_path,
            device=self.device,
            n_features=self.input_dim,
        )

        model_after_last_task: Optional[nn.Module] = None
        
        for task in enabled_tasks:
            flag = task["name"]
            self.logger.info(f"Starting pre-training task: {flag}")
            
            # Determine output dimension
            if task["output_dim_type"] == "n_features":
                output_dim = self.input_dim
            else:
                output_dim = task["output_dim"]

            current_model = create_model(self.config, self.input_dim, output_dim).to(self.device)
            
            if model_after_last_task:
                self._handle_weight_chaining(current_model, model_after_last_task)

            pre_trainer = PreTrainer(
                model=current_model,
                config=pre_train_cfg,
                optimizer=torch.optim.AdamW(
                    current_model.parameters(), lr=self.config.learning_rate
                ),
            )

            call_args = [train_loader]
            if task["requires_val"]:
                if val_loader is None:
                    self.logger.warning(f"Validation loader for {flag} not found, passing None.")
                call_args.append(val_loader)

            start_time = time.time()
            trained_model = getattr(pre_trainer, task["method"])(*call_args, **task["kwargs"])
            self.logger.info(f"{flag} training time: {time.time() - start_time:.2f}s")

            # Save pre-trained checkpoint
            checkpoint_path = self.ctx.get_checkpoint_path(f"pretrained_{flag}.pth")
            torch.save(trained_model.state_dict(), checkpoint_path)
            self.logger.info(f"Pre-trained weights for {flag} saved to {checkpoint_path}")

            model_after_last_task = trained_model

        return model_after_last_task

    def _handle_weight_chaining(self, current_model: nn.Module, prev_model: nn.Module):
        """Copies compatible weights from the previous model to the current one."""
        self.logger.info(f"Attempting weight chaining for {self.config.model}")
        try:
            prev_state_dict = prev_model.state_dict()
            current_model_dict = current_model.state_dict()

            load_state_dict = {
                k: v
                for k, v in prev_state_dict.items()
                if k in current_model_dict and v.shape == current_model_dict[k].shape
            }
            
            missing_keys, unexpected_keys = current_model.load_state_dict(
                load_state_dict, strict=False
            )
            
            if missing_keys:
                self.logger.debug(f"Chaining: Missing keys: {missing_keys}")
            if unexpected_keys:
                self.logger.debug(f"Chaining: Unexpected keys: {unexpected_keys}")
                
            self.logger.info("Weight chaining: successfully loaded compatible weights.")
        except Exception as e:
            self.logger.warning(f"Weight chaining failed: {e}. Model will train from scratch.")

    def adapt_for_finetuning(self, model: nn.Module, pre_trained_model: nn.Module):
        """Adapts a pre-trained model for fine-tuning by loading compatible weights."""
        self.logger.info("Adapting pre-trained model for fine-tuning...")
        self._handle_weight_chaining(model, pre_trained_model)
