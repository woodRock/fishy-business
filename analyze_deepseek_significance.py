import pandas as pd
import numpy as np
from scipy import stats
import os


def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return 0
    dof = nx + ny - 2
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    if std_x == 0 and std_y == 0:
        return 0
    pooled_std = np.sqrt(((nx - 1) * std_x**2 + (ny - 1) * std_y**2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std


def run_analysis(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Load with low_memory=False to avoid DtypeWarning, and treat empty as False for flags
    df = pd.read_csv(file_path)

    # Filter for finished runs and non-null accuracy
    df = df[df["val_balanced_accuracy"].notnull()]

    # Robustly handle boolean flags
    # We treat 'true' as True, and anything else (false, empty, NaN) as False
    bool_cols = ["use_ema", "use_muon", "use_mla", "use_mhc", "use_engram"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip() == "true"

    datasets = df["dataset"].unique()

    for ds in datasets:
        ds_df = df[df["dataset"] == ds]

        # 1. Baseline: AugFormer, No EMA, No Muon, No DeepSeek
        baseline = ds_df[
            (ds_df["model"] == "augformer")
            & (ds_df["use_ema"] == False)
            & (ds_df["use_muon"] == False)
            & (ds_df["use_mla"] == False)
        ]

        # 2. Muon SOTA: AugFormer, Muon + EMA, No DeepSeek
        muon_sota = ds_df[
            (ds_df["model"] == "augformer")
            & (ds_df["use_muon"] == True)
            & (ds_df["use_mla"] == False)
        ]

        # 3. Frankenstein: AugFormer, Muon + EMA + MLA (DeepSeek)
        # Note: We group by use_mla=True since it represents the new architecture
        frankenstein = ds_df[
            (ds_df["model"] == "augformer") & (ds_df["use_mla"] == True)
        ]

        groups = {
            "Baseline": baseline["val_balanced_accuracy"].values,
            "Muon SOTA": muon_sota["val_balanced_accuracy"].values,
            "Frankenstein (MLA+)": frankenstein["val_balanced_accuracy"].values,
        }

        # Filter out empty groups
        groups = {k: v for k, v in groups.items() if len(v) > 0}

        if "Frankenstein (MLA+)" not in groups:
            continue

        print(f"\n{'='*95}")
        print(f" DATASET: {ds.upper()}")
        print(f"{'='*95}")
        print(
            f"{'Configuration':<25} | {'N':<3} | {'Mean':<8} | {'Std':<8} | {'p (vs Frank)':<12} | {'Effect (d)':<10}"
        )
        print(f"{'-'*95}")

        frank_data = groups["Frankenstein (MLA+)"]

        for name, values in groups.items():
            n = len(values)
            mean = np.mean(values)
            std = np.std(values)

            if name == "Frankenstein (MLA+)":
                print(
                    f"{name:<25} | {n:<3} | {mean:.4f}   | {std:.4f}   | {'Target':<12} | {'N/A':<10}"
                )
            else:
                # Mann-Whitney U test (Muon vs this group)
                u_stat, p_val = stats.mannwhitneyu(
                    frank_data, values, alternative="two-sided"
                )
                d = cohen_d(frank_data, values)

                sig = "*" if p_val < 0.05 else " "
                p_str = f"{p_val:.4f}{sig}"
                print(
                    f"{name:<25} | {n:<3} | {mean:.4f}   | {std:.4f}   | {p_str:<12} | {d:.4f}"
                )

    print(f"\n{'='*95}")
    print("Note: * indicates statistically significant difference vs Frankenstein.")
    print(f"{'='*95}\n")


if __name__ == "__main__":
    run_analysis("wandb_deepseek_experiments.csv")
