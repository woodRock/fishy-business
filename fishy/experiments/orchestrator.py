# -*- coding: utf-8 -*-
"""
Batch experiment orchestrator for large-scale benchmarking and statistical analysis.
"""

import logging
import time
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
from fishy._core.factory import MODEL_REGISTRY

logger = logging.getLogger(__name__)

# Experiment Constants
DATASETS = ["species", "part", "oil", "cross-species"]
CLASSIC_MODELS = ["opls-da", "knn", "dt", "lr", "lda", "nb", "rf", "svm"]
DEEP_MODELS = list(MODEL_REGISTRY.keys()) + ["transformer_msm"]
DEFAULT_DATA_PATH = "data/REIMS.xlsx"

def get_fold_count(dataset: str) -> int:
    """Returns the fold count for a specific dataset as requested."""
    if dataset == "part":
        return 3
    return 5

def run_all_experiments(
    num_runs: int = 30,
    wandb_log: bool = False,
    wandb_project: str = "fishy-business",
    wandb_entity: str = "victoria-university-of-wellington",
    quick: bool = False,
    file_path: str = DEFAULT_DATA_PATH
):
    """
    Runs all experiments across all datasets and models for n runs.
    Performs statistical analysis (paired t-test) against OPLS-DA baseline.
    """
    if quick:
        num_runs = 2
        active_datasets = ["species"]
        active_classic = ["opls-da", "svm"]
        active_deep = ["transformer"]
        active_evo = True
        logger.info("QUICK MODE: Running reduced set of experiments.")
    else:
        active_datasets = DATASETS
        active_classic = CLASSIC_MODELS
        active_deep = DEEP_MODELS
        active_evo = True

    # Main Orchestrator Context
    orchestrator_ctx = RunContext(
        dataset="summary", 
        method="orchestrator", 
        model_name="benchmark_suite"
    )
    main_logger = orchestrator_ctx.logger
    
    # Structure to hold results: {dataset: {model: [run1_val_acc, run2_val_acc, ...]}}
    val_results_map = {d: {m: [] for m in active_classic + active_deep + (["evolutionary"] if active_evo else [])} for d in active_datasets}
    train_results_map = {d: {m: [] for m in active_classic + active_deep + (["evolutionary"] if active_evo else [])} for d in active_datasets}

    # Generate consistent master seeds for each of the n trials
    # Every method will be tested on these exact same seeds in order.
    master_seeds = [(i + 1) * 123 for i in range(num_runs)]

    for dataset in active_datasets:
        main_logger.info(f"--- BENCHMARKING DATASET: {dataset} ---")
        folds = get_fold_count(dataset)
        
        # 1. Classic Models
        for model_name in active_classic:
            main_logger.info(f"Running Classic Model: {model_name} on {dataset}")
            for run_idx in range(num_runs):
                current_seed = master_seeds[run_idx]
                set_seed(current_seed)
                
                config = TrainingConfig(
                    dataset=dataset,
                    model=model_name,
                    k_folds=folds,
                    wandb_log=wandb_log,
                    wandb_project=wandb_project,
                    wandb_entity=wandb_entity,
                    run=current_seed,
                    file_path=file_path
                )
                try:
                    stats_res = run_classic_experiment(config, model_name, dataset, run_id=current_seed, file_path=file_path)
                    val_results_map[dataset][model_name].append(stats_res.get("val_balanced_accuracy", 0))
                    train_results_map[dataset][model_name].append(stats_res.get("train_balanced_accuracy", 0))
                except Exception as e:
                    main_logger.error(f"Failed {model_name} on {dataset} run {run_idx} (seed {current_seed}): {e}")
                    val_results_map[dataset][model_name].append(0)
                    train_results_map[dataset][model_name].append(0)

        # 2. Deep Models
        for model_name in active_deep:
            main_logger.info(f"Running Deep Model: {model_name} on {dataset}")
            for run_idx in range(num_runs):
                current_seed = master_seeds[run_idx]
                set_seed(current_seed)
                
                config = TrainingConfig(
                    dataset=dataset,
                    model=model_name,
                    k_folds=folds,
                    epochs=50 if not quick else 2,
                    wandb_log=wandb_log,
                    wandb_project=wandb_project,
                    wandb_entity=wandb_entity,
                    run=current_seed,
                    file_path=file_path
                )
                
                # Special handling for MSM
                if model_name == "transformer_msm":
                    config.model = "transformer"
                    config.masked_spectra_modelling = True
                
                try:
                    stats_res = run_training_pipeline(config)
                    val_results_map[dataset][model_name].append(stats_res.get("val_balanced_accuracy", 0))
                    train_results_map[dataset][model_name].append(stats_res.get("train_balanced_accuracy", 0))
                except Exception as e:
                    main_logger.error(f"Failed {model_name} on {dataset} run {run_idx} (seed {current_seed}): {e}")
                    val_results_map[dataset][model_name].append(0)
                    train_results_map[dataset][model_name].append(0)

        # 3. Evolutionary Models
        if active_evo:
            main_logger.info(f"Running Evolutionary (GP) on {dataset}")
            for run_idx in range(num_runs):
                current_seed = master_seeds[run_idx]
                set_seed(current_seed)
                try:
                    # Evolutionary uses a different signature
                    stats_res = run_gp_experiment(
                        dataset=dataset,
                        generations=10 if not quick else 1,
                        population=100 if not quick else 10,
                        run=current_seed,
                        wandb_log=wandb_log,
                        wandb_project=wandb_project,
                        wandb_entity=wandb_entity,
                        data_file_path=file_path
                    )
                    # For evolutionary, we need to extract from return dict. 
                    # Assuming it returns {'stats': {...}} or similar as per recent fixes
                    # Check run_gp_experiment implementation - it returns stats dict directly now
                    val_results_map[dataset]["evolutionary"].append(stats_res.get("val_balanced_accuracy", 0))
                    train_results_map[dataset]["evolutionary"].append(stats_res.get("train_balanced_accuracy", 0))
                except Exception as e:
                    main_logger.error(f"Failed Evolutionary on {dataset} run {run_idx}: {e}")
                    val_results_map[dataset]["evolutionary"].append(0)
                    train_results_map[dataset]["evolutionary"].append(0)

    # --- STATISTICAL ANALYSIS ---
    main_logger.info("Performing Statistical Significance Tests (Paired T-Test vs OPLS-DA)...")
    
    summary_data = []
    for dataset in active_datasets:
        baseline_model = "opls-da"
        if baseline_model not in val_results_map[dataset] or not val_results_map[dataset][baseline_model]:
            main_logger.warning(f"Baseline {baseline_model} not found for {dataset}. Skipping dataset analysis.")
            continue
            
        baseline_vals = val_results_map[dataset][baseline_model]
        baseline_train_vals = train_results_map[dataset][baseline_model]
        baseline_mean = np.mean(baseline_vals)
        
        # 1. Add baseline method row itself
        summary_data.append({
            "dataset": dataset,
            "model": baseline_model,
            "val_ba_mean": baseline_mean,
            "val_ba_std": np.std(baseline_vals),
            "train_ba_mean": np.mean(baseline_train_vals),
            "p_value_val": 1.0,
            "p_value_train": 1.0,
            "sig_symbol": "≈" # Baseline is equivalent to itself
        })
        
        for model in val_results_map[dataset]:
            if model == baseline_model:
                continue
                
            model_vals = val_results_map[dataset][model]
            if not model_vals:
                continue
            
            model_mean = np.mean(model_vals)
                
            # Validation T-Test
            t_stat, p_val = stats.ttest_rel(baseline_vals, model_vals)
            
            # Training T-Test
            t_stat_train, p_val_train = stats.ttest_rel(
                baseline_train_vals, 
                train_results_map[dataset][model]
            )
            
            # Determine significance symbol (+, -, ≈)
            symbol = "≈"
            if p_val < 0.05:
                symbol = "+" if model_mean > baseline_mean else "-"
            
            res = {
                "dataset": dataset,
                "model": model,
                "val_ba_mean": model_mean,
                "val_ba_std": np.std(model_vals),
                "train_ba_mean": np.mean(train_results_map[dataset][model]),
                "p_value_val": p_val,
                "p_value_train": p_val_train,
                "sig_symbol": symbol
            }
            summary_data.append(res)

    summary_df = pd.DataFrame(summary_data)
    orchestrator_ctx.save_dataframe(summary_df, "statistical_summary.csv")
    
    # Log to W&B if enabled
    if wandb_log:
        import wandb
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create a final summary run
        summary_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=f"benchmark_summary_{orchestrator_ctx.timestamp}",
            job_type="summary_analysis"
        )
        
        # 1. Log results table
        summary_run.log({"statistical_results": wandb.Table(dataframe=summary_df)})
        
        # 2. Log Comparison Heatmap
        try:
            # Pivot data for heatmap: Rows=Models, Columns=Datasets, Values=Val Acc
            pivot_df = summary_df.pivot(index="model", columns="dataset", values="val_ba_mean")
            plt.figure(figsize=(12, 10))
            sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f")
            plt.title("Mean Balanced Accuracy across Datasets")
            plt.tight_layout()
            summary_run.log({"model_performance_heatmap": wandb.Image(plt)})
            plt.close()
        except Exception as e:
            main_logger.warning(f"Could not generate summary heatmap: {e}")
            
        summary_run.finish()

    main_logger.info("Run All Experiments complete. Summary saved.")
    return summary_df
