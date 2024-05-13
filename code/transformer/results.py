# Results - results.py
# Date: 2024-05-13
# Author: Jesse Wood
# 
# This script reads the results from a file and collects
# information about the mean and standard deviation.

import os 
import torch 

if __name__ == "__main__":
    # Set verbose to true for debugging.
    verbose = True
    # Select the dataset to process results for.
    datasets = ["species", "part", "oil", "cross-species"]
    dataset = datasets[3] # Cross-species
    # Path to the logging folder.
    folder = os.path.join("logs",dataset,"nsp")
    # Lists to collect the results with.
    train_accs = []
    val_accs = [] 
    test_accs = []
    runs = 3
    # For each experiment in a batch of 30 independent runs.
    for i in range(1,runs + 1):
        file_name = f"run_{i}.log"
        if verbose:
            print(f"file_name: {file_name}")
        file_path = os.path.join(folder, file_name)
        with open(file_path) as f:
            content = f.readlines()
            # Extract the train, validation and test accuracy.
            print(f"content[240]: {content[332]}")
            # The training accuracy is on line 211.
            train_acc: float = float(content[330].split(sep=' ')[-1])
            # The validation accuracy is on line 213.
            val_acc: float = float(content[332].split(sep=' ')[-1])
            # The test accuracy is on line 215.
            test_acc: float = float(content[334].split(sep=' ')[-1])
            # Append the accuracy to an array.
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            test_accs.append(test_acc)
            
            # Verbose output for debugging purposes
            if verbose: 
                print(f"Train: {train_acc}")
                print(f"Validation: {val_acc}")
                print(f"Test: {test_acc}")

    # Convert to tensors.
    train_accs = torch.tensor(train_accs)
    val_accs = torch.tensor(val_accs)
    test_accs = torch.tensor(test_accs)

    # Mean and standard deviation for each results array.
    mean, std = torch.mean(train_accs), torch.std(train_accs)
    print(f"Training: mean: {mean} +\- {std}")
    mean, std = torch.mean(val_accs), torch.std(val_accs)
    print(f"Validation: mean: {mean} +\- {std}")
    mean, std = torch.mean(test_accs), torch.std(test_accs)
    print(f"Test: mean: {mean} +\- {std}")