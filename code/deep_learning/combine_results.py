import pandas as pd
import glob

# Find all benchmark result files
files = glob.glob("/Users/woodj/Desktop/fishy-business/code/benchmark_results_*.csv")

# Read and concatenate all files
all_dfs = [pd.read_csv(f) for f in files]
combined_df = pd.concat(all_dfs, ignore_index=True)

# Sort the results for better readability
combined_df = combined_df.sort_values(by=["dataset", "model"]).reset_index(drop=True)

# Print the combined table
print(combined_df)

# Save the combined table
combined_df.to_csv(
    "/Users/woodj/Desktop/fishy-business/code/benchmark_results_all.csv", index=False
)

print("Combined results saved to benchmark_results_all.csv")
