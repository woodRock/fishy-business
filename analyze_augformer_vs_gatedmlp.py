import pandas as pd
import numpy as np
from scipy import stats
import os

def cohen_d(x, y):
    nx = len(x); ny = len(y)
    if nx < 2 or ny < 2: return 0
    dof = nx + ny - 2
    std_x = np.std(x, ddof=1); std_y = np.std(y, ddof=1)
    if std_x == 0 and std_y == 0: return 0
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*std_x**2 + (ny-1)*std_y**2) / dof)

def run_comparison(file_path):
    df = pd.read_csv(file_path)
    datasets = df['dataset'].unique()

    print(f"\n{'='*95}")
    print(f" STATISTICAL COMPARISON: AugFormer EMA vs GatedMLP Baseline")
    print(f"{'='*95}")
    print(f"{'Dataset':<15} | {'Aug EMA Mean':<12} | {'GMLP Base Mean':<14} | {'p-value':<10} | {'Effect (d)':<10}")
    print(f"{'-'*95}")

    for ds in datasets:
        ds_df = df[df['dataset'] == ds]
        
        aug_ema = ds_df[(ds_df['model'] == 'augformer') & (ds_df['use_ema'] == True) & (ds_df['use_muon'] == False)]['val_balanced_accuracy'].dropna().values
        gmlp_base = ds_df[(ds_df['model'] == 'gatedmlp')]['val_balanced_accuracy'].dropna().values
        
        if len(aug_ema) == 0 or len(gmlp_base) == 0:
            continue

        # One-sided test: is AugFormer EMA GREATER than GatedMLP?
        u_stat, p_val = stats.mannwhitneyu(aug_ema, gmlp_base, alternative='greater')
        d = cohen_d(aug_ema, gmlp_base)
        
        sig = "*" if p_val < 0.05 else " "
        print(f"{ds:<15} | {np.mean(aug_ema):.4f}       | {np.mean(gmlp_base):.4f}         | {p_val:.4f}{sig} | {d:.4f}")

    print(f"{'='*95}\n")

if __name__ == "__main__":
    run_comparison("wandb_muon_experiments.csv")
