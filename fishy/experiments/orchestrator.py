# -*- coding: utf-8 -*-
"""
Batch experiment orchestrator for large-scale benchmarking and statistical analysis.
Uses external configuration for registries.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from scipy import stats
from dataclasses import asdict

from fishy.experiments.classic_training import run_classic_experiment
from fishy.experiments.deep_training import run_training_pipeline
from fishy.experiments.evolutionary import run_gp_experiment
from fishy._core.config import TrainingConfig
from fishy._core.utils import RunContext, set_seed
from fishy._core.config_loader import load_config

logger = logging.getLogger(__name__)

def get_fold_count(dataset: str) -> int:
    """
    Returns the fold count for a specific dataset.

    Args:
        dataset (str): The name of the dataset.

    Returns:
        int: Number of folds to use for cross-validation.
    """
    return 3 if dataset == "part" else 5

def run_all_experiments(
    num_runs: int = 30,
    wandb_log: bool = False,
    wandb_project: str = "fishy-business",
    wandb_entity: str = "victoria-university-of-wellington",
    quick: bool = False,
    file_path: str = "data/REIMS.xlsx"
) -> pd.DataFrame:
    """
    Runs all experiments across all datasets and models for n runs.

    Performs statistical analysis (paired t-test) against OPLS-DA baseline
    and saves a summary report.

    Args:
        num_runs (int, optional): Number of independent runs. Defaults to 30.
        wandb_log (bool, optional): Enable W&B logging. Defaults to False.
        wandb_project (str, optional): W&B project name. Defaults to "fishy-business".
        wandb_entity (str, optional): W&B entity name. Defaults to "victoria-university-of-wellington".
        quick (bool, optional): If True, runs a minimal set of experiments for testing. Defaults to False.
        file_path (str, optional): Path to the source data file. Defaults to "data/REIMS.xlsx".

    Returns:
        pd.DataFrame: A statistical summary of all experiment results.
    """
    # Load registries from config
    models_cfg = load_config("models")
    datasets_cfg = load_config("datasets")
    
    classic_models = list(models_cfg["classic_models"].keys())
    deep_models = list(models_cfg["deep_models"].keys()) + ["transformer_msm"]
    datasets = [d for d in ["species", "part", "oil", "cross-species"] if d in datasets_cfg]

    if quick:
        num_runs, datasets = 2, ["species"]
        classic_models, deep_models = ["opls-da", "svm"], ["transformer"]
        active_evo = True
        logger.info("QUICK MODE: Running reduced set of experiments.")
    else:
        active_evo = True

    orchestrator_ctx = RunContext(dataset="summary", method="orchestrator", model_name="benchmark_suite")
    main_logger = orchestrator_ctx.logger
    
    val_results_map = {d: {m: [] for m in classic_models + deep_models + (["evolutionary"] if active_evo else [])} for d in datasets}
    train_results_map = {d: {m: [] for m in classic_models + deep_models + (["evolutionary"] if active_evo else [])} for d in datasets}

    master_seeds = [(i + 1) * 123 for i in range(num_runs)]

    for dataset in datasets:
        main_logger.info(f"--- BENCHMARKING DATASET: {dataset} ---")
        folds = get_fold_count(dataset)
        
        # 1. Classic Models
        for model_name in classic_models:
            main_logger.info(f"Running Classic Model: {model_name} on {dataset}")
            for seed in master_seeds:
                set_seed(seed)
                config = TrainingConfig(dataset=dataset, model=model_name, k_folds=folds, wandb_log=wandb_log, wandb_project=wandb_project, wandb_entity=wandb_entity, run=seed, file_path=file_path)
                try:
                    stats_res = run_classic_experiment(config, model_name, dataset, run_id=seed, file_path=file_path)
                    val_results_map[dataset][model_name].append(stats_res.get("val_balanced_accuracy", 0))
                    train_results_map[dataset][model_name].append(stats_res.get("train_balanced_accuracy", 0))
                except Exception as e:
                    main_logger.error(f"Failed {model_name} on {dataset}: {e}")
                    val_results_map[dataset][model_name].extend([0])
                    train_results_map[dataset][model_name].extend([0])

        # 2. Deep Models
        for model_name in deep_models:
            main_logger.info(f"Running Deep Model: {model_name} on {dataset}")
            for seed in master_seeds:
                set_seed(seed)
                config = TrainingConfig(dataset=dataset, model=model_name, k_folds=folds, epochs=50 if not quick else 2, wandb_log=wandb_log, wandb_project=wandb_project, wandb_entity=wandb_entity, run=seed, file_path=file_path)
                if model_name == "transformer_msm":
                    config.model, config.masked_spectra_modelling = "transformer", True
                
                try:
                    stats_res = run_training_pipeline(config)
                    val_results_map[dataset][model_name].append(stats_res.get("val_balanced_accuracy", 0))
                    train_results_map[dataset][model_name].append(stats_res.get("train_balanced_accuracy", 0))
                except Exception as e:
                    main_logger.error(f"Failed {model_name} on {dataset}: {e}")
                    val_results_map[dataset][model_name].extend([0])
                    train_results_map[dataset][model_name].extend([0])

        # 3. Evolutionary Models
        if active_evo:
            main_logger.info(f"Running Evolutionary (GP) on {dataset}")
            for seed in master_seeds:
                set_seed(seed)
                try:
                    stats_res = run_gp_experiment(dataset=dataset, generations=10 if not quick else 1, population=100 if not quick else 10, run=seed, wandb_log=wandb_log, wandb_project=wandb_project, wandb_entity=wandb_entity, data_file_path=file_path)
                    val_results_map[dataset]["evolutionary"].append(stats_res.get("val_balanced_accuracy", 0))
                    train_results_map[dataset]["evolutionary"].append(stats_res.get("train_balanced_accuracy", 0))
                except Exception as e:
                    main_logger.error(f"Failed Evolutionary on {dataset}: {e}")
                    val_results_map[dataset]["evolutionary"].append(0)
                    train_results_map[dataset]["evolutionary"].append(0)

    # Statistical Analysis
    summary_data = []
    for dataset in datasets:
        baseline = "opls-da"
        if baseline not in val_results_map[dataset] or not val_results_map[dataset][baseline]: continue
        
        b_vals, b_train_vals = val_results_map[dataset][baseline], train_results_map[dataset][baseline]
        b_mean = np.mean(b_vals)
        
        summary_data.append({"dataset": dataset, "model": baseline, "val_ba_mean": b_mean, "val_ba_std": np.std(b_vals), "train_ba_mean": np.mean(b_train_vals), "p_value_val": 1.0, "sig_symbol": "≈"})
        
        for model in val_results_map[dataset]:
            if model == baseline or not val_results_map[dataset][model]: continue
            m_vals = val_results_map[dataset][model]
            m_mean = np.mean(m_vals)
            _, p_val = stats.ttest_rel(b_vals, m_vals)
            symbol = "≈"
            if p_val < 0.05: symbol = "+" if m_mean > b_mean else "-"
            summary_data.append({"dataset": dataset, "model": model, "val_ba_mean": m_mean, "val_ba_std": np.std(m_vals), "train_ba_mean": np.mean(train_results_map[dataset][model]), "p_value_val": p_val, "sig_symbol": symbol})

    summary_df = pd.DataFrame(summary_data)
    orchestrator_ctx.save_dataframe(summary_df, "statistical_summary.csv")
    return summary_df