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
