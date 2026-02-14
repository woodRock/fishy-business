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
from rich.panel import Panel
from fishy._core.utils import console


def perform_significance_test(
    model_results: List[float], baseline_results: List[float], alpha: float = 0.05
) -> Dict[str, Any]:
    """Performs a paired t-test between model and baseline results."""
    m_res = np.array(model_results)
    b_res = np.array(baseline_results)

    # Pad shorter array with its mean to allow comparison if lengths differ
    if len(m_res) != len(b_res):
        if len(m_res) > len(b_res):
            b_res = np.pad(
                b_res,
                (0, len(m_res) - len(b_res)),
                mode="constant",
                constant_values=np.mean(b_res),
            )
        else:
            m_res = np.pad(
                m_res,
                (0, len(b_res) - len(m_res)),
                mode="constant",
                constant_values=np.mean(m_res),
            )

    if len(m_res) < 2:
        return {"error": "Single sample", "symbol": "s", "significant": False}

    # Handle zero variance case (e.g. all padded or all same)
    if np.all(m_res == b_res):
        return {"p_value": 1.0, "significant": False, "symbol": "≈"}

    try:
        t_stat, p_val = stats.ttest_rel(b_res, m_res)
        mean_m, mean_b = np.mean(m_res), np.mean(b_res)

        symbol = "≈"
        if p_val < alpha:
            symbol = "+" if mean_m > mean_b else "-"
        return {
            "p_value": p_val,
            "t_statistic": t_stat,
            "significant": p_val < alpha,
            "symbol": symbol,
        }
    except:
        return {"error": "Test failed", "symbol": " ", "significant": False}


def summarize_results(
    results_map: Dict[str, List[Dict[str, Any]]], baseline_model: Optional[str] = None
):
    """Summarizes results and calculates significance for both Train and Test."""
    import logging

    logger = logging.getLogger(__name__)

    datasets = {}
    for key, raw_results in results_map.items():
        dataset, model = key.split("|||", 1) if "|||" in key else ("unknown", key)
        if dataset not in datasets:
            datasets[dataset] = {}

        # Collect all unique metric keys from the raw results
        all_metric_keys = set()
        for r in raw_results:
            src = r.get("stats", r)
            all_metric_keys.update(src.keys())
        
        # We'll specifically look for and normalize core metrics
        metric_lists = {k: [] for k in all_metric_keys if isinstance(raw_results[0].get("stats", raw_results[0]).get(k), (int, float, np.number))}
        
        # Core metric mapping for consistency
        core_map = {
            "val_balanced_accuracy": ["val_balanced_accuracy", "balanced_accuracy", "test_balanced_accuracy", "accuracy"],
            "train_balanced_accuracy": ["train_balanced_accuracy", "train_accuracy"]
        }

        for r in raw_results:
            src = r.get("stats", r)
            for k in metric_lists.keys():
                val = src.get(k, 0.0)
                try:
                    metric_lists[k].append(float(val))
                except:
                    metric_lists[k].append(0.0)

        # Ensure we have normalized names for the significance tests
        normalized = {"val": [], "train": []}
        for target, aliases in core_map.items():
            for alias in aliases:
                if alias in metric_lists and metric_lists[alias]:
                    normalized["val" if "val" in target or "test" in target or target == "val_balanced_accuracy" else "train"] = metric_lists[alias]
                    break
        
        if not normalized["val"] and "val_balanced_accuracy" in metric_lists: normalized["val"] = metric_lists["val_balanced_accuracy"]
        if not normalized["train"] and "train_balanced_accuracy" in metric_lists: normalized["train"] = metric_lists["train_balanced_accuracy"]

        datasets[dataset][model] = {"metrics": metric_lists, "val": normalized["val"], "train": normalized["train"]}

    summary_data = []
    for dataset, models in datasets.items():
        actual_baseline = (
            baseline_model
            if (baseline_model and baseline_model in models)
            else ("opls-da" if "opls-da" in models else list(models.keys())[0])
        )

        b_val_accs = models[actual_baseline]["val"]
        b_train_accs = models[actual_baseline]["train"]

        for model_name, m_data in models.items():
            m_val, m_train = m_data["val"], m_data["train"]
            
            if model_name == actual_baseline:
                sig_test, sig_train = " ", " "
            else:
                sig_test = perform_significance_test(m_val, b_val_accs)["symbol"] if m_val and b_val_accs else " "
                sig_train = perform_significance_test(m_train, b_train_accs)["symbol"] if m_train and b_train_accs else " "

            row = {
                "Dataset": dataset,
                "Method": model_name,
                "Train": np.mean(m_train) if m_train else 0.0,
                "Train Std": np.std(m_train) if m_train else 0.0,
                "Sig Tr": sig_train,
                "Test": np.mean(m_val) if m_val else 0.0,
                "Test Std": np.std(m_val) if m_val else 0.0,
                "Sig Te": sig_test,
                "is_baseline": model_name == actual_baseline,
                "Baseline": actual_baseline,
            }
            
            # Add all other metrics as means
            for m_key, m_values in m_data["metrics"].items():
                # Skip core ones we already handled
                if m_key in ["val_balanced_accuracy", "train_balanced_accuracy", "val", "train"]:
                    continue
                row[m_key] = np.mean(m_values)
                row[f"{m_key}_std"] = np.std(m_values)

            summary_data.append(row)

    return pd.DataFrame(summary_data)


def display_statistical_summary(df: pd.DataFrame, show_significance: bool = True):
    """Displays a pretty table grouped by dataset."""
    table = Table(title="[bold green]Full Statistical Analysis Summary[/]", box=None)
    table.add_column("Dataset", style="dim")
    table.add_column("Method", style="bold cyan")
    table.add_column("Train Acc", justify="right", style="magenta")
    table.add_column("Std", justify="right", style="dim")
    if show_significance:
        table.add_column("Sig", justify="center", style="bold yellow")
    table.add_column("Test Acc", justify="right", style="green")
    table.add_column("Std", justify="right", style="dim")
    if show_significance:
        table.add_column("Sig", justify="center", style="bold yellow")

    df = df.sort_values(by=["Dataset", "Method"])
    current_ds = None
    for _, row in df.iterrows():
        ds_name = row["Dataset"] if row["Dataset"] != current_ds else ""
        current_ds = row["Dataset"]
        row_data = [
            ds_name,
            row["Method"],
            f"{row['Train']:.4f}",
            f"{row['Train Std']:.4f}",
        ]
        if show_significance:
            row_data.append(row["Sig Tr"])
        row_data.extend([f"{row['Test']:.4f}", f"{row['Test Std']:.4f}"])
        if show_significance:
            row_data.append(row["Sig Te"])
        table.add_row(*row_data)
    console.print("\n")
    console.print(Panel(table, expand=False, border_style="green"))
    console.print("\n")
