# -*- coding: utf-8 -*-
"""
Benchmark script to compare Transformer performance in three scenarios:
1. Baseline
2. MSM Pre-training
3. MSM as Data Augmentation

Configurations:
- 5-fold stratified cross-validation (standard)
- 3-fold stratified cross-validation (for 'part' dataset)
- Currently focused on 'cross-species'
"""

import os
import torch
import numpy as np
import pandas as pd
from fishy._core.config import TrainingConfig
from fishy.experiments.unified_trainer import run_unified_training
from fishy._core.utils import console, set_seed
from rich.table import Table

def run_benchmark(dataset="cross-species"):
    model = "transformer"
    runs = 1 # We use k-folds instead of multiple independent runs for variance
    
    # Determine k_folds
    k_folds = 3 if dataset == "part" else 5
    
    console.print(f"[bold yellow]Starting Benchmark on dataset: {dataset} ({k_folds}-fold CV)[/]")

    scenarios = [
        {"name": "Baseline", "config": {}},
        {"name": "MSM Pre-training", "config": {"masked_spectra_modelling": True}},
        {"name": "MSM Augmentation", "config": {
            "data_augmentation": True, 
            "mask_enabled": True, 
            "mask_size": 50
        }}
    ]
    
    all_results = []

    for scenario in scenarios:
        console.print(f"\n[bold blue]Scenario: {scenario['name']}[/]")
        
        # In k-fold mode, run_unified_training returns the aggregated stats
        # from all folds.
        seed = 42
        set_seed(seed)
        
        # Create config
        cfg = TrainingConfig(
            model=model,
            dataset=dataset,
            run=seed,
            epochs=50, 
            k_folds=k_folds,
            normalize=True,
            **scenario.get("config", {})
        )
        
        try:
            res = run_unified_training(cfg)
            
            # Extract balanced accuracy metrics
            # trainer._compile_best_metrics uses 'train_balanced_accuracy' and 'val_balanced_accuracy'
            train_bal_acc = res.get("train_balanced_accuracy", 0)
            val_bal_acc = res.get("val_balanced_accuracy", 0)
            
            # Fallback for validation if needed (unified trainer/dispatch stats)
            if val_bal_acc == 0:
                val_bal_acc = res.get("test_balanced_accuracy", res.get("balanced_accuracy", 0))
            
            all_results.append({
                "Scenario": scenario["name"],
                "Train Balanced Acc": train_bal_acc,
                "Val Balanced Acc": val_bal_acc
            })
            
            console.print(f"[green]Finished {scenario['name']}. Train: {train_bal_acc:.4f}, Val: {val_bal_acc:.4f}[/]")

        except Exception as e:
            console.print(f"[red]Error in {scenario['name']}: {e}[/]")
            import traceback
            console.print(traceback.format_exc())

    # Display final results table
    table = Table(title=f"MSM Benchmark Results ({dataset}, {k_folds}-fold CV)")
    table.add_column("Scenario", style="cyan")
    table.add_column("Train Balanced Acc", justify="right", style="magenta")
    table.add_column("Val Balanced Acc", justify="right", style="green")

    for res in all_results:
        table.add_row(
            res["Scenario"], 
            f"{res['Train Balanced Acc']:.4f}", 
            f"{res['Val Balanced Acc']:.4f}"
        )

    console.print("\n")
    console.print(table)

if __name__ == "__main__":
    import sys
    # Allow passing dataset as argument, default to cross-species
    target_dataset = sys.argv[1] if len(sys.argv) > 1 else "cross-species"
    run_benchmark(target_dataset)
