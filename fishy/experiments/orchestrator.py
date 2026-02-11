# -*- coding: utf-8 -*-
"""
Batch experiment orchestrator for large-scale benchmarking and statistical analysis.
Uses TrainingEngine for unified execution.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from scipy import stats
from dataclasses import asdict

from fishy.experiments.unified_trainer import run_unified_training
from fishy._core.config import TrainingConfig
from fishy._core.utils import RunContext, set_seed
from fishy._core.config_loader import load_config

logger = logging.getLogger(__name__)

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
    Uses UnifiedTrainer for consistent execution.
    """
    # Load registries from config
    models_cfg = load_config("models")
    datasets_cfg = load_config("datasets")
    
    classic_models = list(models_cfg["classic_models"].keys())
    deep_models = list(models_cfg["deep_models"].keys())
    evolutionary_models = list(models_cfg["evolutionary_models"].keys())
    
    datasets = [d for d in ["species", "part", "oil", "cross-species"] if d in datasets_cfg]

    if quick:
        num_runs, datasets = 2, ["species"]
        classic_models, deep_models, evolutionary_models = ["opls-da"], ["transformer"], ["ga"]
        logger.info("QUICK MODE: Running reduced set of experiments.")

    orchestrator_ctx = RunContext(dataset="summary", method="orchestrator", model_name="benchmark_suite")
    main_logger = orchestrator_ctx.logger
    
    all_models = [
        (m, "classic") for m in classic_models
    ] + [
        (m, "deep") for m in deep_models
    ] + [
        (m, "evolutionary") for m in evolutionary_models
    ]

    # Track results: dataset -> model -> list of accuracies
    val_results_map = {d: {m[0]: [] for m in all_models} for d in datasets}
    train_results_map = {d: {m[0]: [] for m in all_models} for d in datasets}

    master_seeds = [(i + 1) * 123 for i in range(num_runs)]

    for dataset in datasets:
        main_logger.info(f"--- BENCHMARKING DATASET: {dataset} ---")
        
        for model_name, method in all_models:
            main_logger.info(f"Running {method.upper()} Model: {model_name} on {dataset}")
            
            for seed in master_seeds:
                set_seed(seed)
                config = TrainingConfig(
                    dataset=dataset, 
                    model=model_name, 
                    method=method,
                    wandb_log=wandb_log, 
                    wandb_project=wandb_project, 
                    wandb_entity=wandb_entity, 
                    run=seed, 
                    file_path=file_path,
                    epochs=50 if not quick and method == "deep" else (2 if quick else None)
                )
                
                try:
                    res = run_unified_training(config)
                    val_results_map[dataset][model_name].append(res.get("val_balanced_accuracy", 0))
                    train_results_map[dataset][model_name].append(res.get("train_balanced_accuracy", 0))
                except Exception as e:
                    main_logger.error(f"Failed {model_name} on {dataset}: {e}")
                    val_results_map[dataset][model_name].append(0)
                    train_results_map[dataset][model_name].append(0)

    # Statistical Analysis
    summary_data = []
    baseline = "opls-da"
    
    for dataset in datasets:
        if baseline not in val_results_map[dataset] or not val_results_map[dataset][baseline]:
            main_logger.warning(f"Baseline {baseline} not found for dataset {dataset}. Skipping stats.")
            continue
        
        b_vals = val_results_map[dataset][baseline]
        b_mean = np.mean(b_vals)
        
        summary_data.append({
            "dataset": dataset, "model": baseline, 
            "val_ba_mean": b_mean, "val_ba_std": np.std(b_vals), 
            "train_ba_mean": np.mean(train_results_map[dataset][baseline]), 
            "p_value_val": 1.0, "sig_symbol": "≈"
        })
        
        for model_name, _ in all_models:
            if model_name == baseline: continue
            m_vals = val_results_map[dataset][model_name]
            if not m_vals: continue
            
            m_mean = np.mean(m_vals)
            _, p_val = stats.ttest_rel(b_vals, m_vals)
            symbol = "≈"
            if p_val < 0.05: symbol = "+" if m_mean > b_mean else "-"
            
            summary_data.append({
                "dataset": dataset, "model": model_name, 
                "val_ba_mean": m_mean, "val_ba_std": np.std(m_vals), 
                "train_ba_mean": np.mean(train_results_map[dataset][model_name]), 
                "p_value_val": p_val, "sig_symbol": symbol
            })

    summary_df = pd.DataFrame(summary_data)
    orchestrator_ctx.save_dataframe(summary_df, "statistical_summary.csv")
    return summary_df
