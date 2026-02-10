# -*- coding: utf-8 -*-
"""
Benchmarking module for deep learning models.
Uses external configuration for model registries.
"""
import time
import torch
import numpy as np
import pandas as pd
from typing import List, Optional
import wandb
from dataclasses import asdict
from fishy.data.classic_loader import load_dataset
from fishy.engine.training_loops import train_model
from fishy._core.factory import create_model
from fishy._core.config import TrainingConfig
from fishy._core.utils import RunContext
from fishy._core.config_loader import load_config


def get_device() -> torch.device:
    """Selects the best available device."""
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


def run_benchmark(
    model_names: List[str],
    warmup_epochs: int = 0,
    output_file: str = "benchmark_results.csv",
    file_path: str = None,
    wandb_project: Optional[str] = "fishy-business",
    wandb_entity: Optional[str] = "victoria-university-of-wellington",
    wandb_log: bool = False,
) -> pd.DataFrame:
    """Benchmarks specified models on classification tasks."""
    wandb_run = None
    if wandb_log:
        wandb_config_dict = {"model_names": model_names, "warmup_epochs": warmup_epochs, "output_file": output_file, "file_path": file_path}
        wandb_run = wandb.init(project=wandb_project, entity=wandb_entity, config=wandb_config_dict, reinit=True, group="benchmark_suite", job_type="benchmarking")
    
    ctx = RunContext(dataset="summary", method="benchmark", model_name="orchestrator", wandb_run=wandb_run)
    logger = ctx.logger
    device = get_device()
    
    # Load available datasets from config
    datasets_cfg = load_config("datasets")
    datasets = [d for d in ["species", "part", "oil", "cross-species"] if d in datasets_cfg]
    
    all_results = []
    try:
        for model_name in model_names:
            model_results = []
            for dataset_name in datasets:
                logger.info(f"Benchmarking {model_name} on {dataset_name}...")
                X, y, _ = load_dataset(dataset_name, file_path=file_path)
                n_features, n_classes = X.shape[1], len(np.unique(y))
                
                config = TrainingConfig(model=model_name, dataset=dataset_name, epochs=1, k_folds=1)
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long()), batch_size=32)
                
                if warmup_epochs > 0:
                    warmup_model = create_model(config, n_features, n_classes).to(device)
                    train_model(warmup_model, train_loader, torch.nn.CrossEntropyLoss(), torch.optim.Adam(warmup_model.parameters()), num_epochs=warmup_epochs, n_splits=1, n_runs=1, device=str(device))
                
                model = create_model(config, n_features, n_classes).to(device)
                start_time = time.time()
                train_model(model, train_loader, torch.nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters()), num_epochs=1, n_splits=1, n_runs=1, device=str(device))
                training_time = time.time() - start_time
                
                start_time = time.time()
                with torch.no_grad(): model(torch.from_numpy(X).float().to(device))
                inference_time = time.time() - start_time
                
                res = {
                    "model": model_name, "dataset": dataset_name, "training_time": training_time, "inference_time": inference_time,
                    "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6,
                    "num_params": sum(p.numel() for p in model.parameters()),
                }
                model_results.append(res)
                all_results.append(res)
            
            ctx.save_dataframe(pd.DataFrame(model_results), f"benchmark_results_{model_name}.csv")
        
        final_df = pd.DataFrame(all_results)
        ctx.save_dataframe(final_df, output_file)
        return final_df
    finally:
        if wandb_run: wandb_run.finish()