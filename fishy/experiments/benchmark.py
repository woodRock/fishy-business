# -*- coding: utf-8 -*-
"""
Benchmarking module for deep learning models.
Uses DataModule for standardized data loading and TrainingEngine for execution.
"""
import time
import torch
import numpy as np
import pandas as pd
from typing import List, Optional
import wandb
from dataclasses import asdict

from fishy.data.module import create_data_module
from fishy.engine.trainer import TrainingEngine
from fishy._core.factory import create_model
from fishy._core.config import TrainingConfig
from fishy._core.utils import RunContext, get_device
from fishy._core.config_loader import load_config


def run_benchmark(
    model_names: List[str],
    warmup_epochs: int = 0,
    output_file: str = "benchmark_results.csv",
    file_path: Optional[str] = None,
    wandb_project: Optional[str] = "fishy-business",
    wandb_entity: Optional[str] = "victoria-university-of-wellington",
    wandb_log: bool = False,
) -> pd.DataFrame:
    """
    Benchmarks specified models on classification tasks across multiple datasets.

    Args:
        model_names (List[str]): List of model architecture names to benchmark.
        warmup_epochs (int, optional): Number of epochs to warm up. Defaults to 0.
        output_file (str, optional): Output CSV path. Defaults to "benchmark_results.csv".
        file_path (str, optional): Path to the source data file. Defaults to None.
        wandb_project (str, optional): W&B project name. Defaults to "fishy-business".
        wandb_entity (str, optional): W&B entity name. Defaults to "victoria-university-of-wellington".
        wandb_log (bool, optional): Enable W&B logging. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the benchmark results.
    """
    wandb_run = None
    if wandb_log:
        wandb_config_dict = {"model_names": model_names, "warmup_epochs": warmup_epochs, "output_file": output_file, "file_path": file_path}
        wandb_run = wandb.init(project=wandb_project, entity=wandb_entity, config=wandb_config_dict, reinit=True, group="benchmark_suite", job_type="benchmarking")
    
    ctx = RunContext(dataset="summary", method="benchmark", model_name="orchestrator", wandb_run=wandb_run)
    logger = ctx.logger
    device = get_device()
    
    datasets_cfg = load_config("datasets")
    datasets = [d for d in ["species", "part", "oil", "cross-species"] if d in datasets_cfg]
    
    all_results = []
    try:
        for model_name in model_names:
            model_results = []
            for dataset_name in datasets:
                logger.info(f"Benchmarking {model_name} on {dataset_name}...")
                
                # Standardized Data Loading to get dimensions
                data_module = create_data_module(dataset_name=dataset_name, file_path=file_path)
                data_module.setup()
                X, _ = data_module.get_numpy_data()
                n_features = X.shape[1]
                n_classes = data_module.get_num_classes()
                
                config = TrainingConfig(model=model_name, dataset=dataset_name, epochs=1, k_folds=1)
                
                # We use TrainingEngine for the actual run
                start_time = time.time()
                TrainingEngine.run_deep(config)
                training_time = time.time() - start_time
                
                # Re-create model briefly for size/inference stats
                model = create_model(config, n_features, n_classes).to(device)
                start_time = time.time()
                with torch.no_grad(): 
                    input_sample = torch.from_numpy(X[:32]).float().to(device)
                    model(input_sample)
                inference_time = (time.time() - start_time) / 32
                
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
