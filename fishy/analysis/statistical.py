# -*- coding: utf-8 -*-
"""
Statistical analysis tools for experiment results.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Any, Optional, Union

def perform_significance_test(
    model_results: List[float], 
    baseline_results: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Performs a paired t-test between model results and baseline results.

    Examples:
        >>> model = [0.9, 0.95, 0.92]
        >>> baseline = [0.8, 0.82, 0.81]
        >>> res = perform_significance_test(model, baseline)
        >>> bool(res['significant'])
        True
        >>> res['symbol']
        '+'
    """
    if len(model_results) != len(baseline_results):
        return {"error": "Result lengths do not match for paired test"}
    
    if len(model_results) < 2:
        return {"error": "Not enough samples for statistical significance"}

    t_stat, p_val = stats.ttest_rel(baseline_results, model_results)
    mean_m = np.mean(model_results)
    mean_b = np.mean(baseline_results)
    
    symbol = "≈"
    if p_val < alpha:
        symbol = "+" if mean_m > mean_b else "-"
        
    return {
        "p_value": p_val,
        "t_statistic": t_stat,
        "significant": p_val < alpha,
        "symbol": symbol,
        "mean_diff": mean_m - mean_b
    }

def summarize_results(results_map: Dict[str, List[Dict[str, Any]]], baseline_model: Optional[str] = None):
    """
    Summarizes results and performs significance tests across models for each dataset.
    
    Args:
        results_map: Dict mapping "dataset|||model" to list of result dicts.
        baseline_model: Name of the model to use as baseline.

    Examples:
        >>> results = {
        ...     "ds1|||m1": [{"val_balanced_accuracy": 0.8}, {"val_balanced_accuracy": 0.82}],
        ...     "ds1|||m2": [{"val_balanced_accuracy": 0.9}, {"val_balanced_accuracy": 0.92}]
        ... }
        >>> df = summarize_results(results, baseline_model="m1")
        >>> len(df)
        2
        >>> df.iloc[1]['significance']
        '+'
    """
    # Group by dataset
    datasets = {}
    for key, results in results_map.items():
        if "|||" in key:
            dataset, model = key.split("|||", 1)
        else:
            # Fallback for old style if any
            dataset, model = "unknown", key
            
        if dataset not in datasets:
            datasets[dataset] = {}
        datasets[dataset][model] = [r.get("val_balanced_accuracy", 0) for r in results]

    summary_data = []
    
    for dataset, models in datasets.items():
        # Determine baseline for this dataset
        actual_baseline = baseline_model
        if not actual_baseline or actual_baseline not in models:
            if "opls-da" in models:
                actual_baseline = "opls-da"
            else:
                actual_baseline = list(models.keys())[0]
        
        b_vals = models[actual_baseline]
        b_mean = np.mean(b_vals)
        
        for model_name, m_vals in models.items():
            m_mean = np.mean(m_vals)
            m_std = np.std(m_vals)
            
            p_val = 1.0
            symbol = "≈"
            
            if model_name != actual_baseline and len(m_vals) == len(b_vals) and len(m_vals) > 1:
                test_res = perform_significance_test(m_vals, b_vals)
                if "error" not in test_res:
                    p_val = test_res["p_value"]
                    symbol = test_res["symbol"]
            
            summary_data.append({
                "dataset": dataset,
                "model": model_name,
                "mean_acc": m_mean,
                "std_acc": m_std,
                "p_value": p_val,
                "significance": symbol,
                "is_baseline": model_name == actual_baseline
            })
            
    return pd.DataFrame(summary_data)

def analyze_regression_predictions(
    predictions: Dict[str, np.ndarray], 
    fold: int, 
    ctx: Any,
    dataset_name: str = "dataset"
) -> None:
    """
    Analyzes and visualizes predictions for regression tasks.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import r2_score, mean_absolute_error

    logger = ctx.logger
    if not predictions:
        logger.warning(f"Fold {fold + 1}: No predictions to analyze.")
        return

    true_labels = predictions["labels"]
    pred_labels = predictions["preds"]

    mae = mean_absolute_error(true_labels, pred_labels)
    r2 = r2_score(true_labels, pred_labels)
    
    logger.info(f"Fold {fold + 1} {dataset_name} Regression - MAE: {mae:.4f}, R2: {r2:.4f}")

    # Prediction Error Distribution
    errors = pred_labels - true_labels
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel("Prediction Error (Predicted - True)")
    plt.ylabel("Frequency")
    plt.title(f"Fold {fold + 1} Prediction Error Distribution - {dataset_name}")
    ctx.save_figure(plt, f"regression_error_dist_fold_{fold + 1}.png")
    plt.close()

    # True vs Predicted Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(true_labels, pred_labels, alpha=0.5)
    plt.plot([true_labels.min(), true_labels.max()], [true_labels.min(), true_labels.max()], 'r--', lw=2)
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")
    plt.title(f"True vs Predicted - {dataset_name} (Fold {fold + 1})")
    ctx.save_figure(plt, f"regression_scatter_fold_{fold + 1}.png")
    plt.close()
