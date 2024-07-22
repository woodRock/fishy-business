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
    dataset = args['dataset'] # Part
    print(f"dataset: {dataset}")
    # Path to the logging folder.
    folder = os.path.join("logs",dataset,args['folder'])
    # Lists to collect the results in.
    train_accs = []
    val_accs = [] 
    test_accs = []

    # For each result file in the results log.
    runs = len(os.listdir(folder))

    # For each experiment in a batch of 30 independent runs.
    for i in range(1,runs):
        file_name = f"run_{i}.log"
        if verbose:
            print(f"file_name: {file_name}")
        file_path = os.path.join(folder, file_name)
        with open(file_path) as f:
            content = f.readlines()
            # Extract the train, validation and test accuracy.
            # The training accuracy is on the 10th to last line.
            train_acc: float = float(content[-1].split(sep=' ')[-4])
            # The validation accuracy is on the 9th to last line.
            val_acc: float = float(content[-1].split(sep=' ')[-1])
            # The test accuracy is on the 8th to last line.
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
    # Mean and standard deviation for each results array.
    mean, std = torch.mean(train_accs), torch.std(train_accs)
    print(f"Training: mean: {mean} +\- {std}")
    mean, std = torch.mean(val_accs), torch.std(val_accs)
    print(f"Validation: mean: {mean} +\- {std}")
