import pandas as pd
import numpy as np
from scipy import stats
import os

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2: return 0
    dof = nx + ny - 2
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    # Handle zero variance
    if std_x == 0 and std_y == 0: return 0
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*std_x**2 + (ny-1)*std_y**2) / dof)

def run_analysis(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    
    # Define our four experimental groups
    # 1. Baseline AugFormer (AdamW, no EMA)
    # 2. EMA AugFormer (AdamW + EMA)
    # 3. Baseline GatedMLP (AdamW, no EMA)
    # 4. Muon SOTA (Muon + EMA)
    
    datasets = df['dataset'].unique()

    for ds in datasets:
        ds_df = df[df['dataset'] == ds]
        
        # Extract groups
        groups = {
            "AugFormer Baseline": ds_df[(ds_df['model'] == 'augformer') & (ds_df['use_ema'] == False) & (ds_df['use_muon'] == False)],
            "AugFormer EMA":      ds_df[(ds_df['model'] == 'augformer') & (ds_df['use_ema'] == True)  & (ds_df['use_muon'] == False)],
            "GatedMLP Baseline":  ds_df[(ds_df['model'] == 'gatedmlp')],
            "AugFormer Muon SOTA":ds_df[(ds_df['model'] == 'augformer') & (ds_df['use_ema'] == True)  & (ds_df['use_muon'] == True)]
        }

        # Filter out empty groups and get values
        data = {k: v['val_balanced_accuracy'].dropna().values for k, v in groups.items() if len(v) > 0}
        
        if "AugFormer Muon SOTA" not in data:
            continue

        print(f"\n{'='*90}")
        print(f" DATASET: {ds.upper()}")
        print(f"{'='*90}")
        print(f"{'Configuration':<25} | {'N':<3} | {'Mean':<8} | {'Std':<8} | {'p (vs Muon)':<12} | {'Effect (d)':<10}")
        print(f"{'-'*90}")

        muon_data = data["AugFormer Muon SOTA"]
        muon_mean = np.mean(muon_data)
        muon_std = np.std(muon_data)

        for name, values in data.items():
            n = len(values)
            mean = np.mean(values)
            std = np.std(values)
            
            if name == "AugFormer Muon SOTA":
                print(f"{name:<25} | {n:<3} | {mean:.4f}   | {std:.4f}   | {'Target':<12} | {'N/A':<10}")
            else:
                # Mann-Whitney U test (Muon vs this group)
                # We test if Muon is DIFFERENT (two-sided)
                u_stat, p_val = stats.mannwhitneyu(muon_data, values, alternative='two-sided')
                d = cohen_d(muon_data, values)
                
                sig = "*" if p_val < 0.05 else " "
                p_str = f"{p_val:.4f}{sig}"
                print(f"{name:<25} | {n:<3} | {mean:.4f}   | {std:.4f}   | {p_str:<12} | {d:.4f}")

    print(f"\n{'='*90}")
    print("Note: * indicates statistically significant difference (p < 0.05) vs Muon SOTA.")
    print(f"{'='*90}\n")

if __name__ == "__main__":
    run_analysis("wandb_muon_experiments.csv")
