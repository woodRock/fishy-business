import pandas as pd
import numpy as np
from scipy import stats


def rank_biserial(u_stat, n1, n2):
    """Effect size for Mann-Whitney U: r = 2U/(n1*n2) - 1, range [-1, 1].
    Positive means the first group tends to exceed the second."""
    return (2 * u_stat) / (n1 * n2) - 1


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2)
        / (nx + ny - 2)
    )
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0.0


def main():
    df = pd.read_csv("wandb_ensemble_versus.csv")
    df = df[df["val_balanced_accuracy"].notnull()]

    datasets = sorted(df["dataset"].unique())
    challengers = [m for m in df["model"].unique() if m != "ensemble"]

    # Bonferroni correction over all (dataset, challenger) pairs
    n_comparisons = len(datasets) * len(challengers)
    alpha = 0.05
    alpha_corrected = alpha / n_comparisons

    print(f"Significance threshold (Bonferroni, {n_comparisons} comparisons): "
          f"α = {alpha_corrected:.4f}\n")

    col_w = 12
    header = (
        f"{'Model':<12} | {'N':>4} | {'Mean':>8} | {'Std':>7} | "
        f"{'p (one-tail)':>13} | {'p (corrected)':>14} | {'r (rb)':>7} | {'d':>7} | Sig?"
    )
    sep = "-" * len(header)

    for ds in datasets:
        ds_df = df[df["dataset"] == ds]
        ensemble_vals = ds_df[ds_df["model"] == "ensemble"]["val_balanced_accuracy"].values

        print("=" * len(header))
        print(f" DATASET: {ds.upper()}   (ensemble n={len(ensemble_vals)}, "
              f"mean={np.mean(ensemble_vals):.4f} ± {np.std(ensemble_vals):.4f})")
        print("=" * len(header))
        print(header)
        print(sep)

        for model in sorted(challengers):
            model_vals = ds_df[ds_df["model"] == model]["val_balanced_accuracy"].values
            if len(model_vals) == 0:
                continue

            n1, n2 = len(model_vals), len(ensemble_vals)
            # One-tailed: is model strictly better than ensemble?
            u_stat, p_raw = stats.mannwhitneyu(
                model_vals, ensemble_vals, alternative="greater"
            )
            p_corrected = min(p_raw * n_comparisons, 1.0)
            rb = rank_biserial(u_stat, n1, n2)
            d = cohen_d(model_vals, ensemble_vals)

            sig = "YES ***" if p_corrected < alpha else ("yes *  " if p_raw < alpha else "no     ")
            print(
                f"{model:<12} | {n1:>4} | {np.mean(model_vals):>8.4f} | "
                f"{np.std(model_vals):>7.4f} | {p_raw:>13.4f} | {p_corrected:>14.4f} | "
                f"{rb:>7.3f} | {d:>7.4f} | {sig}"
            )

        print()

    print("=" * len(header))
    print("One-tailed Mann-Whitney U: tests whether model > ensemble.")
    print("'YES ***' = significant after Bonferroni correction.")
    print("'yes *  ' = significant at raw α=0.05 only.")
    print(f"r (rb): rank-biserial correlation [-1, 1].  d: Cohen's d.")
    print("=" * len(header))


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
