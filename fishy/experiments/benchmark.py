# -*- coding: utf-8 -*-
"""
Benchmarking module for deep learning models.

This module provides tools to benchmark the performance of various deep learning models
across multiple datasets. It measures training time, inference time, model size, and
the number of parameters, facilitating a comparative analysis of different architectures.
"""

import time
import torch
import numpy as np
import pandas as pd
from typing import List, Optional # Added Optional

import wandb # Added import

from dataclasses import asdict # Added import



from fishy.data.classic_loader import load_dataset

from fishy.engine.training_loops import train_model

from fishy._core.factory import create_model, MODEL_REGISTRY

from fishy._core.config import TrainingConfig

from fishy._core.utils import RunContext



def get_device() -> torch.device:

    """

    Automatically selects the best available device (MPS, CUDA, or CPU).



    Returns:

        torch.device: The selected device.

    """

    if torch.backends.mps.is_available():

        return torch.device("mps")

    elif torch.cuda.is_available():

        return torch.device("cuda")

    else:

        return torch.device("cpu")



def run_benchmark(

    model_names: List[str], 

    warmup_epochs: int = 0, 

    output_file: str = "benchmark_results.csv", 

    file_path: str = None,

    # New W&B parameters

    wandb_project: Optional[str] = "fishy-business",

    wandb_entity: Optional[str] = "victoria-university-of-wellington",

    wandb_log: bool = False,

) -> pd.DataFrame:

    """

    Benchmarks specified models on standard classification tasks.



    For each model and dataset, it performs:

    1.  Optional warm-up epochs.

    2.  Measurement of 1 epoch of training time.

    3.  Measurement of a single inference pass.

    4.  Calculation of model disk size and parameter count.



    Args:

        model_names (List[str]): List of model architecture names to benchmark.

        warmup_epochs (int): Number of warm-up epochs to run before measurement.

        output_file (str): Path to save the final results as a CSV.

        file_path (str): Optional path to the dataset excel file.

        wandb_project (str, optional): Weights & Biases project name. Defaults to "fishy-business".

        wandb_entity (str, optional): Weights & Biases entity name. Defaults to "victoria-university-of-wellington".

        wandb_log (bool): Whether to log to Weights & Biases. Defaults to False.



    Returns:

        pd.DataFrame: A DataFrame containing the benchmarking metrics for all models and datasets.

    """

    wandb_run = None

    if wandb_log:

        # Create a dict for W&B config from function arguments

        wandb_config_dict = {

            "model_names": model_names,

            "warmup_epochs": warmup_epochs,

            "output_file": output_file,

            "file_path": file_path,

        }

        wandb_run = wandb.init(

            project=wandb_project,

            entity=wandb_entity,

            config=wandb_config_dict,

            reinit=True,

            group="benchmark_suite",

            job_type="benchmarking"

        )



    ctx = RunContext(dataset="summary", method="benchmark", model_name="orchestrator", wandb_run=wandb_run)

    logger = ctx.logger

    device = get_device()

    datasets = ["species", "part", "oil", "cross-species"]

    all_results = []



    try: # Start try block for wandb.finish

        for model_name in model_names:

            model_results = []

            for dataset_name in datasets:

                logger.info(f"Benchmarking {model_name} on {dataset_name}...")



                # Load data

                X, y, _ = load_dataset(dataset_name, file_path=file_path)

                n_features = X.shape[1]

                n_classes = len(np.unique(y))

                

                # Create a minimal config for create_model

                config = TrainingConfig(

                    file_path="", model=model_name, dataset=dataset_name, 

                    run=0, output="", data_augmentation=False, 

                    masked_spectra_modelling=False, next_spectra_prediction=False,

                    next_peak_prediction=False, spectrum_denoising_autoencoding=False,

                    peak_parameter_regression=False, spectrum_segment_reordering=False,

                    contrastive_transformation_invariance_learning=False,

                    early_stopping=0, dropout=0.2, label_smoothing=0.1,

                    epochs=1, learning_rate=1e-4, batch_size=32,

                    hidden_dimension=128, num_layers=4, num_heads=4,

                    num_augmentations=0, noise_level=0.0, shift_enabled=False,

                    scale_enabled=False, k_folds=1

                )



                train_loader = torch.utils.data.DataLoader(

                    torch.utils.data.TensorDataset(

                        torch.from_numpy(X).float(), torch.from_numpy(y).long()

                    ),

                    batch_size=32,

                )



                # --- Warm-up ---

                if warmup_epochs > 0:

                    logger.info(f"Running {warmup_epochs} warm-up epochs...")

                    warmup_model = create_model(config, n_features, n_classes).to(device)

                    train_model(

                        warmup_model,

                        train_loader,

                        torch.nn.CrossEntropyLoss(),

                        torch.optim.Adam(warmup_model.parameters()),

                        num_epochs=warmup_epochs,

                        n_splits=1,

                        n_runs=1,

                        device=str(device)

                    )



                # --- Training Time Measurement ---

                model = create_model(config, n_features, n_classes).to(device)

                start_time = time.time()

                train_model(

                    model,

                    train_loader,

                    torch.nn.CrossEntropyLoss(),

                    torch.optim.Adam(model.parameters()),

                    num_epochs=1,

                    n_splits=1,

                    n_runs=1,

                    device=str(device)

                )

                training_time = time.time() - start_time



                # --- Inference Time Measurement ---

                start_time = time.time()

                with torch.no_grad():

                    model(torch.from_numpy(X).float().to(device))

                inference_time = time.time() - start_time



                # Model metrics

                model_size = sum(p.numel() * p.element_size() for p in model.parameters())

                num_params = sum(p.numel() for p in model.parameters())



                res = {

                    "model": model_name,

                    "dataset": dataset_name,

                    "training_time": training_time,

                    "inference_time": inference_time,

                    "model_size_mb": model_size / 1e6,

                    "num_params": num_params,

                }

                model_results.append(res)

                all_results.append(res)

            

            df = pd.DataFrame(model_results)

            ctx.save_dataframe(df, f"benchmark_results_{model_name}.csv")



        final_df = pd.DataFrame(all_results)

        ctx.save_dataframe(final_df, output_file)

        logger.info(f"All benchmark results saved to {ctx.result_dir / output_file}")

        return final_df

    finally:

        if wandb_run:

            wandb_run.finish()
