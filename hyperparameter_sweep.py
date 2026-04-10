import subprocess
import re
import pandas as pd


def run_trial(top_k, num_layers, hidden_dim):
    cmd = [
        "python3",
        "-m",
        "fishy",
        "train",
        "-m",
        "role_filler",
        "-d",
        "oil",
        "-e",
        "5",
        "-n",
        "1",
        "--use-performer",
        "--top-k",
        str(top_k),
        "--num-layers",
        str(num_layers),
        "--hidden-dim",
        str(hidden_dim),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Extract Balanced Accuracy (Val)
    # Example line: "Balanced Accuracy   0.1508   0.1429"
    bal_acc_match = re.search(r"Balanced Accuracy\s+[\d.]+\s+([\d.]+)", result.stdout)
    bal_acc = float(bal_acc_match.group(1)) if bal_acc_match else None

    # Extract Elapsed training time
    # Example line: "Elapsed training time: 12.3768 seconds"
    time_match = re.search(r"Elapsed training time: ([\d.]+) seconds", result.stdout)
    elapsed_time = float(time_match.group(1)) if time_match else None

    return bal_acc, elapsed_time


top_k_list = [128, 256, 512]
num_layers_list = [2, 4]
hidden_dim_list = [64, 128]

results = []

for tk in top_k_list:
    for nl in num_layers_list:
        for hd in hidden_dim_list:
            bal_acc, elapsed_time = run_trial(tk, nl, hd)
            results.append(
                {
                    "top_k": tk,
                    "num_layers": nl,
                    "hidden_dim": hd,
                    "Balanced Accuracy (Val)": bal_acc,
                    "Time (s)": elapsed_time,
                }
            )

df = pd.DataFrame(results)
print("\nHyperparameter Sweep Results:")
print(df.to_markdown(index=False))

best_idx = df["Balanced Accuracy (Val)"].idxmax()
best_settings = df.loc[best_idx]
print("\nRecommendation:")
print(
    f"Best settings: top_k={best_settings['top_k']}, num_layers={best_settings['num_layers']}, hidden_dim={best_settings['hidden_dim']}"
)
print(f"Balanced Accuracy: {best_settings['Balanced Accuracy (Val)']}")
