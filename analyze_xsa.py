import pandas as pd
import numpy as np
from scipy import stats
import os


def analyze_xsa():
    file_path = "wandb_augformer_xsa.csv"
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    # Load data
    df = pd.read_csv(file_path)

    # Clean use_xsa column
    if df["use_xsa"].dtype == object:
        df["use_xsa"] = (
            df["use_xsa"].astype(str).str.replace('"', "").str.strip().str.lower()
        )
        df["is_xsa"] = df["use_xsa"] == "true"
    else:
        df["is_xsa"] = df["use_xsa"].astype(bool)

    # Filter for relevant columns and clean
    df = df[["dataset", "model", "is_xsa", "val_balanced_accuracy"]]
    df["val_balanced_accuracy"] = pd.to_numeric(
        df["val_balanced_accuracy"], errors="coerce"
    )
    df = df.dropna(subset=["val_balanced_accuracy"])

    datasets = df["dataset"].unique()

    results = []

    print(
        f"{'Dataset':<15} | {'Mode':<12} | {'Mean Acc':<10} | {'Std Dev':<10} | {'P-Value':<10}"
    )
    print("-" * 75)

    for ds in datasets:
        ds_data = df[df["dataset"] == ds]

        acc_xsa = ds_data[ds_data["is_xsa"] == True]["val_balanced_accuracy"].values
        acc_std = ds_data[ds_data["is_xsa"] == False]["val_balanced_accuracy"].values

        mean_xsa = np.mean(acc_xsa) if len(acc_xsa) > 0 else 0
        mean_std = np.mean(acc_std) if len(acc_std) > 0 else 0

        std_xsa = np.std(acc_xsa) if len(acc_xsa) > 0 else 0
        std_std = np.std(acc_std) if len(acc_std) > 0 else 0

        # Perform Welch's t-test (Independent)
        if len(acc_xsa) > 1 and len(acc_std) > 1:
            t_stat, p_val = stats.ttest_ind(acc_xsa, acc_std, equal_var=False)
        else:
            p_val = np.nan

        sig = " (Significant)" if (not np.isnan(p_val) and p_val < 0.05) else ""

        print(
            f"{ds:<15} | XSA          | {mean_xsa:.4f}     | {std_xsa:.4f}     | {p_val:.4f}{sig}"
        )
        print(f"{'':<15} | Standard     | {mean_std:.4f}     | {std_std:.4f}     |")
        print("-" * 75)


if __name__ == "__main__":
    analyze_xsa()
