import pandas as pd
import numpy as np
from scipy import stats
import os

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

def analyze_benchmarks(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    
    # Identify the three groups based on their configuration
    # Group 1: Standard (No EMA, No TTT, low epochs)
    # Group 2: EMA (EMA=True, TTT=False)
    # Group 3: Full Recipe (EMA=True, TTT=True)
    
    # Note: Depending on how wandb exports, we might need to filter by epochs or specific flags
    # For this script, we assume the CSV has columns: dataset, balanced_accuracy, use_ema, use_ttt
    
    results = []
    datasets = df['dataset'].unique()

    print(f"{'Dataset':<12} | {'Config':<15} | {'Mean Acc':<10} | {'Std Dev':<10} | {'p-val':<10} | {'Effect (d)':<10}")
    print("-" * 85)

    for ds in datasets:
        ds_df = df[df['dataset'] == ds]
        
        # Split into groups using the correct WandB export columns
        # For Full Recipe (TTT), we use 'val_ttt_balanced_accuracy'
        # For others, we use 'val_balanced_accuracy'
        
        baseline = ds_df[(ds_df['use_ema'] == False) & (ds_df['use_ttt'] == False)]['val_balanced_accuracy'].dropna().values
        ema_only = ds_df[(ds_df['use_ema'] == True) & (ds_df['use_ttt'] == False)]['val_balanced_accuracy'].dropna().values
        
        # For TTT, check if 'val_ttt_balanced_accuracy' exists, fallback to 'val_balanced_accuracy'
        full_recipe_df = ds_df[(ds_df['use_ema'] == True) & (ds_df['use_ttt'] == True)]
        if 'val_ttt_balanced_accuracy' in full_recipe_df.columns:
            full_recipe = full_recipe_df['val_ttt_balanced_accuracy'].dropna().values
        else:
            full_recipe = full_recipe_df['val_balanced_accuracy'].dropna().values

        groups = [
            ("Baseline", baseline),
            ("EMA Only", ema_only),
            ("Full Recipe", full_recipe)
        ]

        # Calculate statistics relative to baseline
        for name, data in groups:
            if len(data) == 0: continue
            
            mean = np.mean(data)
            std = np.std(data)
            
            if name == "Baseline":
                print(f"{ds:<12} | {name:<15} | {mean:.4f}   | {std:.4f}   | {'N/A':<10} | {'N/A':<10}")
            else:
                # Mann-Whitney U Test
                u_stat, p_val = stats.mannwhitneyu(data, baseline, alternative='two-sided')
                d = cohen_d(data, baseline)
                
                sig = "*" if p_val < 0.05 else " "
                print(f"{ds:<12} | {name:<15} | {mean:.4f}   | {std:.4f}   | {p_val:.4f}{sig} | {d:.4f}")
        print("-" * 85)

if __name__ == "__main__":
    # You can run this once you have exported the CSV from WandB
    # Make sure to include 'use_ema' and 'use_ttt' as columns in the export
    analyze_benchmarks("wandb_augformer_ema.csv")
