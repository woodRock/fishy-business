# -*- coding: utf-8 -*-
"""
Quick experiment comparing TurboQuant (Random + Quantize) vs Normalize vs Default.
"""

import os
import torch
import pandas as pd
import numpy as np
from fishy import TrainingConfig, run_unified_training
from fishy._core.utils import console, set_seed
from rich.table import Table
from rich.panel import Panel

# Constants for the experiment
DATASETS = ["species", "part", "oil", "cross-species"]
MODELS = ["opls-da", "rf", "svm", "dt", "lr", "lstm", "cnn", "transformer"]
CONFIGS = {
    "Default": {"random_projection": False, "quantize": False, "turbo_quant": False, "normalize": False},
    "Sign-RP": {"random_projection": True, "quantize": True, "turbo_quant": False, "normalize": False},
    "TurboQuant": {"random_projection": False, "quantize": False, "turbo_quant": True, "normalize": True},
    "Normalize": {"random_projection": False, "quantize": False, "turbo_quant": False, "normalize": True},
}

def run_experiment():
    results = []
    
    # 3-fold Stratified CV is the default in the trainer for most tasks
    # For a quick experiment, we'll use 1 run and few epochs for deep models
    epochs = 20
    
    total_tasks = len(DATASETS) * len(MODELS) * len(CONFIGS)
    completed = 0

    for ds in DATASETS:
        for model in MODELS:
            for cfg_name, flags in CONFIGS.items():
                completed += 1
                console.print(f"[bold blue][{completed}/{total_tasks}][/] Testing [cyan]{model}[/] on [cyan]{ds}[/] with [green]{cfg_name}[/]...")
                
                # Determine method based on model registry
                method = "classic" if model in ["opls-da", "rf", "svm", "dt", "lr"] else "deep"
                
                config = TrainingConfig(
                    model=model,
                    dataset=ds,
                    method=method,
                    epochs=epochs,
                    k_folds=3,
                    num_runs=1,
                    statistical=False, # We'll handle our own aggregation
                    **flags
                )
                
                try:
                    # Run training
                    res = run_unified_training(config)
                    
                    # Extract metric
                    # For consistency, use val_balanced_accuracy or val_accuracy
                    acc = res.get("val_balanced_accuracy", res.get("val_accuracy", 0.0))
                    
                    results.append({
                        "Dataset": ds,
                        "Model": model,
                        "Config": cfg_name,
                        "Accuracy": acc
                    })
                except Exception as e:
                    console.print(f"[bold red]Failed:[/] {e}")
                    results.append({
                        "Dataset": ds,
                        "Model": model,
                        "Config": cfg_name,
                        "Accuracy": 0.0
                    })
                
                # Cleanup to avoid memory issues
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return pd.DataFrame(results)

def display_results(df):
    # Pivot for the final table: Models as rows, Datasets/Configs as columns
    # Or maybe more readable: Average across datasets?
    # User asked for a big table, let's group by Model and show Configs.
    
    summary = df.groupby(["Model", "Config"])["Accuracy"].mean().unstack()
    
    # Reorder columns
    cols = ["Default", "Normalize", "Sign-RP", "TurboQuant"]
    summary = summary[cols]
    
    # Reorder rows to match MODELS list
    summary = summary.reindex(MODELS)

    table = Table(title="[bold green]Preprocessing Benchmark: Mean Balanced Accuracy Across Datasets[/]")
    table.add_column("Model", style="cyan", no_wrap=True)
    for col in cols:
        table.add_column(col, justify="right")

    for model in MODELS:
        row_vals = summary.loc[model]
        best_val = row_vals.max()
        
        row_str = [model]
        for val in row_vals:
            # Bold if it's the best (or tied for best) in the row
            if val == best_val and val > 0:
                row_str.append(f"[bold green]{val:.4f}[/]")
            else:
                row_str.append(f"{val:.4f}")
        table.add_row(*row_str)

    console.print("\n")
    console.print(table)
    
    # Also show a per-dataset summary if needed
    for ds in DATASETS:
        ds_df = df[df["Dataset"] == ds].pivot(index="Model", columns="Config", values="Accuracy")
        ds_df = ds_df[cols].reindex(MODELS)
        
        ds_table = Table(title=f"[bold blue]Results for {ds.upper()} Dataset[/]")
        ds_table.add_column("Model", style="cyan")
        for col in cols:
            ds_table.add_column(col, justify="right")
            
        for model in MODELS:
            row_vals = ds_df.loc[model]
            best_val = row_vals.max()
            row_str = [model]
            for val in row_vals:
                if val == best_val and val > 0:
                    row_str.append(f"[bold green]{val:.4f}[/]")
                else:
                    row_str.append(f"{val:.4f}")
            ds_table.add_row(*row_str)
        console.print(ds_table)

if __name__ == "__main__":
    set_seed(42)
    df_results = run_experiment()
    display_results(df_results)
