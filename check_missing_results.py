# -*- coding: utf-8 -*-
"""
Identifies missing experiment runs in the W&B export CSV.
Expected: 16 models x 4 datasets x 30 runs = 1920 total.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def check_missing(file_path="wanb_export_csv.csv"):
    if not Path(file_path).exists():
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)

    # Handle nested format if necessary (dashboard logic)
    if "summary" in df.columns and "config" in df.columns:
        import ast

        def safe_parse(val):
            if pd.isna(val):
                return {}
            try:
                return ast.literal_eval(val)
            except:
                return {}

        flattened = []
        for _, row in df.iterrows():
            c = safe_parse(row["config"])
            flattened.append({"dataset": c.get("dataset"), "model": c.get("model")})
        df = pd.DataFrame(flattened)

    # Clean up names
    df = df[df["dataset"] != "dataset"]  # Remove header artifact if present
    df["dataset"] = df["dataset"].astype(str)
    df["model"] = df["model"].astype(str)

    expected_datasets = ["species", "part", "oil", "cross-species"]
    expected_models = [
        "moe",
        "ensemble",
        "dt",
        "rf",
        "transformer",
        "lr",
        "xgb",
        "kan",
        "lda",
        "opls-da",
        "nb",
        "svm",
        "knn",
        "cnn",
        "lstm",
        "rcnn",
    ]
    expected_runs = 30

    print("--- Missing Results Report ---")
    print(f"Total runs found: {len(df)}")
    print(f"Target: 1920\n")

    missing_combinations = []

    for ds in expected_datasets:
        for model in expected_models:
            # Count matches
            count = len(df[(df["dataset"] == ds) & (df["model"] == model)])
            if count < expected_runs:
                missing = expected_runs - count
                print(
                    f"❌ {ds.ljust(15)} | {model.ljust(12)} | Missing: {str(missing).rjust(2)} runs (Found {count})"
                )
                missing_combinations.append((ds, model, missing))
            elif count > expected_runs:
                print(
                    f"⚠️  {ds.ljust(15)} | {model.ljust(12)} | Extra:   {str(count-expected_runs).rjust(2)} runs (Found {count})"
                )

    if not missing_combinations:
        print("\n✅ All 1920 runs accounted for!")
    else:
        print(f"\nSummary: {len(missing_combinations)} combinations are incomplete.")


if __name__ == "__main__":
    check_missing()
