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

        val_accs, train_accs = [], []
        for r in raw_results:
            src = r.get("stats", r)
            v = (
                src.get("val_balanced_accuracy")
                or src.get("balanced_accuracy")
                or src.get("test_balanced_accuracy")
                or 0.0
            )
            t = src.get("train_balanced_accuracy") or src.get("train_accuracy") or 0.0

            # Ensure we have floats, even if the source was a string
            try:
                v = float(v)
            except (ValueError, TypeError):
                v = 0.0
            try:
                t = float(t)
            except (ValueError, TypeError):
                t = 0.0

            val_accs.append(v)
            train_accs.append(t)

        datasets[dataset][model] = {"val": val_accs, "train": train_accs}

    summary_data = []
    for dataset, models in datasets.items():
        # Priority for baseline: 1. Passed arg, 2. opls-da, 3. first available
        actual_baseline = (
            baseline_model
            if (baseline_model and baseline_model in models)
            else ("opls-da" if "opls-da" in models else list(models.keys())[0])
        )

        logger.debug(
            f"Dataset {dataset}: Using {actual_baseline} as baseline. Available: {list(models.keys())}"
        )

        b_val_accs = models[actual_baseline]["val"]
        b_train_accs = models[actual_baseline]["train"]

        for model_name, m_data in models.items():
            m_val, m_train = m_data["val"], m_data["train"]

            logger.debug(
                f"  - Testing {model_name} (n={len(m_val)}) against {actual_baseline} (n={len(b_val_accs)})"
            )

            if model_name == actual_baseline:
                sig_test, sig_train = " ", " "
            else:
                sig_res = perform_significance_test(m_val, b_val_accs)
                sig_test = sig_res["symbol"]
                sig_train = perform_significance_test(m_train, b_train_accs)["symbol"]
                if sig_test == " ":
                    logger.debug(
                        f"    * No significance symbol for {model_name}. Reason: {sig_res.get('error', 'None')}"
                    )

            summary_data.append(
                {
                    "Dataset": dataset,
                    "Method": model_name,
                    "Train": np.mean(m_train),
                    "Train Std": np.std(m_train),
                    "Sig Tr": sig_train,
                    "Test": np.mean(m_val),
                    "Test Std": np.std(m_val),
                    "Sig Te": sig_test,
                    "is_baseline": model_name == actual_baseline,
                    "Baseline": actual_baseline,
                }
            )

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
