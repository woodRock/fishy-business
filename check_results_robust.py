# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
import ast

def check_results(file_path="wanb_export_csv.csv"):
    df = pd.read_csv(file_path)
    
    def get_meta(row):
        if "config" in row and isinstance(row["config"], str):
            try:
                c = ast.literal_eval(row["config"])
                return str(c.get("dataset")), str(c.get("model"))
            except: pass
        return str(row.get("dataset")), str(row.get("model"))

    records = []
    for _, row in df.iterrows():
        ds, mod = get_meta(row)
        records.append({"ds": ds, "mod": mod, "name": row.get("Name")})
    
    clean_df = pd.DataFrame(records)
    
    expected_datasets = ['species', 'part', 'oil', 'cross-species']
    expected_models = [
        'moe', 'ensemble', 'dt', 'rf', 'transformer', 'lr', 'xgb', 'kan', 
        'lda', 'opls-da', 'nb', 'svm', 'knn', 'cnn', 'lstm', 'rcnn'
    ]
    
    print(f"Total rows in CSV: {len(clean_df)}")
    
    stats = clean_df.groupby(['ds', 'mod']).size().reset_index(name='count')
    
    print("\n--- EXTRA RUNS ( > 30 ) ---")
    extras = stats[stats['count'] > 30]
    if extras.empty:
        print("None.")
    else:
        for _, row in extras.iterrows():
            print(f"EXTRA: {row['ds']} / {row['mod']} (Found {row['count']})")

    print("\n--- MISSING RUNS ( < 30 ) ---")
    missing_count = 0
    for ds in expected_datasets:
        for model in expected_models:
            mask = (clean_df['ds'] == ds) & (clean_df['mod'] == model)
            count = len(clean_df[mask])
            if count < 30:
                print(f"MISSING: {ds} / {model} (Found {count})")
                missing_count += 30-count
    
    if missing_count == 0:
        print("None.")

    print("\n--- UNEXPECTED COMBINATIONS ---")
    found_unexpected = False
    for _, row in stats.iterrows():
        if row['ds'] not in expected_datasets or row['mod'] not in expected_models:
            print(f"UNEXPECTED: {row['ds']} / {row['mod']} (Count {row['count']})")
            found_unexpected = True
    if not found_unexpected:
        print("None.")

if __name__ == "__main__":
    check_results()
