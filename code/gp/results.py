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
    verbose = False  
    # Select the dataset to process results for.
    datasets = ["species", "part", "oil", "cross-species"]
    dataset = datasets[1] # Part
    # Path to the logging folder.
    folder = os.path.join("logs",dataset,"tmp_3")
    # Lists to collect the results in.
    train_accs = []
    val_accs = [] 
    test_accs = []
    # For each experiment in a batch of 30 independent runs.
    for i in range(1,30+1):
        file_name = f"run_{i}.log"
        if verbose:
            print(f"file_name: {file_name}")
        file_path = os.path.join(folder, file_name)
        with open(file_path) as f:
            content = f.readlines()
            # Extract the train, validation and test accuracy.
            # The training accuracy is on line 411.
            train_acc: float = float(content[411].split(sep=' ')[2])
            # The validation accuracy is on line 412.
            val_acc: float = float(content[412].split(sep=' ')[2])
            # The test accuracy is on line 413.
            test_acc: float = float(content[413].split(sep=' ')[2])
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