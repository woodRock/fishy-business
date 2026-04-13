import pandas as pd
import numpy as np
from scipy import stats
import os

def analyze():
    file_path = 'wanb_augformer_gatemlp.csv'
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    # Load data
    df = pd.read_csv(file_path)
    
    # Filter for relevant columns and clean
    df = df[['dataset', 'model', 'val_balanced_accuracy']]
    df['val_balanced_accuracy'] = pd.to_numeric(df['val_balanced_accuracy'], errors='coerce')
    df = df.dropna()

    datasets = df['dataset'].unique()
    models = ['augformer', 'gatedmlp']

    results = []

    print(f"{'Dataset':<15} | {'Model':<12} | {'Mean Acc':<10} | {'Std Dev':<10} | {'P-Value':<10}")
    print("-" * 65)

    for ds in datasets:
        ds_data = df[df['dataset'] == ds]
        
        acc_aug = ds_data[ds_data['model'] == 'augformer']['val_balanced_accuracy'].values
        acc_gate = ds_data[ds_data['model'] == 'gatedmlp']['val_balanced_accuracy'].values
        
        mean_aug = np.mean(acc_aug) if len(acc_aug) > 0 else 0
        mean_gate = np.mean(acc_gate) if len(acc_gate) > 0 else 0
        
        std_aug = np.std(acc_aug) if len(acc_aug) > 0 else 0
        std_gate = np.std(acc_gate) if len(acc_gate) > 0 else 0

        # Perform t-test (Independent)
        if len(acc_aug) > 1 and len(acc_gate) > 1:
            t_stat, p_val = stats.ttest_ind(acc_aug, acc_gate, equal_var=False)
        else:
            p_val = np.nan

        print(f"{ds:<15} | augformer    | {mean_aug:.4f}     | {std_aug:.4f}     | {p_val:.4f}")
        print(f"{'':<15} | gatedmlp     | {mean_gate:.4f}     | {std_gate:.4f}     |")
        print("-" * 65)

if __name__ == "__main__":
    analyze()
