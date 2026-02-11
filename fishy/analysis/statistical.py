# -*- coding: utf-8 -*-
"""
Statistical analysis tools for experiment results.
"""

import numpy as np
import pandas as pd
import warnings
from scipy import stats
from typing import List, Dict, Any, Optional
from rich.table import Table
from rich.console import Console
from fishy._core.utils import console

def perform_significance_test(model_results: List[float], baseline_results: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    if len(model_results) != len(baseline_results) or len(model_results) < 2:
        return {"error": "Invalid lengths", "symbol": "≈", "significant": False}
    t_stat, p_val = stats.ttest_rel(baseline_results, model_results)
    mean_m, mean_b = np.mean(model_results), np.mean(baseline_results)
    symbol = "≈"
    if p_val < alpha: symbol = "+" if mean_m > mean_b else "-"
    return {"p_value": p_val, "t_statistic": t_stat, "significant": p_val < alpha, "symbol": symbol}

def summarize_results(results_map: Dict[str, List[Dict[str, Any]]], baseline_model: Optional[str] = None):
    """Summarizes results into a DataFrame."""
    datasets = {}
    for key, results in results_map.items():
        dataset, model = key.split("|||", 1) if "|||" in key else ("unknown", key)
        if dataset not in datasets: datasets[dataset] = {}
        datasets[dataset][model] = {
            "val": [r.get("val_balanced_accuracy", 0) for r in results],
            "train": [r.get("train_balanced_accuracy", 0) for r in results]
        }

    summary_data = []
    for dataset, models in datasets.items():
        actual_baseline = baseline_model if (baseline_model and baseline_model in models) else ("opls-da" if "opls-da" in models else list(models.keys())[0])
        b_vals = models[actual_baseline]["val"]
        
        for model_name, m_data in models.items():
            m_val, m_train = m_data["val"], m_data["train"]
            test_res = perform_significance_test(m_val, b_vals) if model_name != actual_baseline else {"symbol": " "}
            summary_data.append({
                "Dataset": dataset, "Method": model_name,
                "Train": np.mean(m_train), "Train Std": np.std(m_train),
                "Test": np.mean(m_val), "Test Std": np.std(m_val),
                "Sig": test_res["symbol"], "is_baseline": model_name == actual_baseline
            })
    return pd.DataFrame(summary_data)

def display_statistical_summary(df: pd.DataFrame, show_significance: bool = True):
    """Displays a pretty table grouped by dataset."""
    table = Table(title="[bold green]Statistical Analysis Summary[/]", box=None)
    table.add_column("Dataset", style="dim")
    table.add_column("Method", style="bold cyan")
    table.add_column("Train Acc", justify="right", style="magenta")
    table.add_column("Std", justify="right", style="dim")
    table.add_column("Test Acc", justify="right", style="green")
    table.add_column("Std", justify="right", style="dim")
    if show_significance: table.add_column("Sig", justify="center", style="bold yellow")

    # Sort to ensure same order for each dataset
    df = df.sort_values(by=["Dataset", "Method"])
    
    current_ds = None
    for _, row in df.iterrows():
        ds_name = row["Dataset"] if row["Dataset"] != current_ds else ""
        current_ds = row["Dataset"]
        
        row_data = [
            ds_name, row["Method"],
            f"{row['Train']:.4f}", f"{row['Train Std']:.4f}",
            f"{row['Test']:.4f}", f"{row['Test Std']:.4f}"
        ]
        if show_significance: row_data.append(row["Sig"])
        table.add_row(*row_data)
        
    console.print("\n", table, "\n")
