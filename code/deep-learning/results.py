# Results - results.py
# Date: 2024-05-13
# Author: Jesse Wood
# 
# This script reads the results from a file and collects
# information about the mean and standard deviation.
import argparse
import os 
import torch

if __name__ == "__main__":
    # Handle the command line arguments for the script.
    parser = argparse.ArgumentParser(
                    prog='Transformer: results',
                    description='A transformer for fish species classification.',
                    epilog='Implemented in pytorch and written in python.')
    parser.add_argument('-d', '--dataset', type=str, default="species",
                        help="The fish species or part dataset. Defaults to species")
    parser.add_argument('-f', '--folder', type=str, default="tmp",
                        help="The folder to get the results from. Defaults to tmp")
    parser.add_argument('-v', '--verbose',
                    action='store_true', default=False,
                    help="Flag for verbose output in logging. Defaults to False.") 
    args = vars(parser.parse_args())
    
    # Set verbose to true for debugging.
    verbose = args['verbose']
    
    # Select the dataset to process results for.
    datasets = ["species", "part", "oil", "oil_simple", "cross-species", "instance-recognition"]
    dataset = args['dataset'] # Cross-species, 
    
    if dataset not in datasets:
        raise ValueError(f"Not a valid dataset: {dataset}")
    
    # Path to the logging folder.
    folder = os.path.join("logs",dataset,args['folder'])
    # Lists to collect the results with.
    train_accs = []
    val_accs = [] 
    test_accs = []
    # A run for each output file in the logs.
    runs = len(os.listdir(path=folder))
    # runs = 27
    
    # For each experiment in a batch of 30 independent runs.
    for i in range(1,runs + 1):
        file_name = f"run_{i}.log"
        if verbose:
            print(f"file_name: {file_name}")
        file_path = os.path.join(folder, file_name)
        with open(file_path) as f:
            content = f.readlines()
            # Extract the train, validation and test accuracy.
            # The training accuracy is the 8th to last line.
            train_acc: float = float(content[-8].split(sep=' ')[-1])
            # The validation accuracy is the 6th to last line.
            print(f"content[-5]: {content[-5]}")
            val_acc: float = float(content[-5].split(sep=' ')[-1])
            # Append the accuracy to an array.
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Verbose output for debugging purposes
            if verbose: 
                print(f"Train: {train_acc}")
                print(f"Validation: {val_acc}")

    # Convert to tensors.
    train_accs = torch.tensor(train_accs)
    val_accs = torch.tensor(val_accs)
    test_accs = torch.tensor(test_accs)

    # Mean and standard deviation for each results array.
    mean, std = torch.mean(train_accs), torch.std(train_accs)
    print(f"Training: mean: {mean} +\- {std}")
    mean, std = torch.mean(val_accs), torch.std(val_accs)
    print(f"Validation: mean: {mean} +\- {std}")
