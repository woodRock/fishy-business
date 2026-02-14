# -*- coding: utf-8 -*-
"""
Cleans the CSV to exactly 1920 runs by:
1. Removing header artifacts (dataset/model rows)
2. Keeping exactly 30 runs for each valid combination
"""

import pandas as pd
import ast
from pathlib import Path

def clean_to_1920(file_path="wanb_export_csv.csv"):
    df = pd.read_csv(file_path)
    print(f"Initial rows: {len(df)}")

    expected_datasets = ['species', 'part', 'oil', 'cross-species']
    expected_models = [
        'moe', 'ensemble', 'dt', 'rf', 'transformer', 'lr', 'xgb', 'kan', 
        'lda', 'opls-da', 'nb', 'svm', 'knn', 'cnn', 'lstm', 'rcnn'
    ]

    def get_meta(row):
        if "config" in row and isinstance(row["config"], str):
            try:
                c = ast.literal_eval(row["config"])
                return str(c.get("dataset")), str(c.get("model"))
            except: pass
        return str(row.get("dataset")), str(row.get("model"))

    # Add temporary meta columns for easy filtering
    meta = [get_meta(row) for _, row in df.iterrows()]
    df['tmp_ds'] = [m[0] for m in meta]
    df['tmp_mod'] = [m[1] for m in meta]

    # 1. Filter to ONLY expected combinations
    df_clean = df[df['tmp_ds'].isin(expected_datasets) & df['tmp_mod'].isin(expected_models)].copy()
    print(f"Rows after filtering unexpected: {len(df_clean)}")

    # 2. For each combination, keep only the first 30
    final_rows = []
    for ds in expected_datasets:
        for model in expected_models:
            subset = df_clean[(df_clean['tmp_ds'] == ds) & (df_clean['tmp_mod'] == model)]
            if len(subset) > 30:
                print(f"✂️  Trimming {ds}/{model} from {len(subset)} to 30")
                final_rows.append(subset.head(30))
            else:
                final_rows.append(subset)

    df_final = pd.concat(final_rows)
    
    # Remove temp columns
    df_final = df_final.drop(columns=['tmp_ds', 'tmp_mod'])

    print(f"Final rows: {len(df_final)}")
    
    if len(df_final) == 1920:
        df_final.to_csv(file_path, index=False)
        print(f"✅ Success! Exactly 1920 runs saved to {file_path}")
    else:
        print(f"⚠️  Warning: Final count is {len(df_final)}, not 1920. Not overwriting yet.")

if __name__ == "__main__":
    clean_to_1920()
