"""
Utility - utility.py 
====================

This module contains the utility methods for the FS methods per k features evaluations. 
These are methods that plot and tabulate the results into human readible formats. 
"""

import numpy as np
import matplotlib.pyplot as plt 
from prettytable import PrettyTable


def plot_accuracy(results, dataset, folder="pso/assets"): 
    """ Plot the training and test accuracy per k features for the results. 

    The method plots the accuracies and saves them to respective files in high-resolution (dpi=500). 
    The images are saved to the "pso/assets" folder by default. 
    
    Args: 
        results: a dictionary of results from all FS methods. 
        dataset: the name of the dataset, Fish or Part     
        folder: the path to the desired folder, defaults to "pso/assets". 
    """
    for name, result in results.items():
        k, train, test = zip(*result)
        if name == "pso":
            plt.scatter(k, train, label=name)
        else:
            plt.plot(k, train, label=name)

    plt.title("Train: Accuracy vs. No. Features")
    plt.xlabel("No. Features")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(fname=f"{folder}/accuracy-features-{dataset}-train", dpi=500)
    plt.show()

    for name, result in results.items():
        k, train, test = zip(*result)
        if name == "pso":
            plt.scatter(k, test, label=name)
        else:
            plt.plot(k, test, label=name)

    plt.title("Test: Accuracy vs. No. Features")
    plt.xlabel("No. Features")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(fname=f"{folder}/accuracy-features-{dataset}-test", dpi=500)
    plt.show()


def show_results(results, label='Method'):
    """ Display highlight results from each FS method. 

    Prints a table in a human-friendly format to the console. 
    The table contains the best k, and respective training and test scores. 
    The table includes the performance on the full dataset as a control. 
    
    Args:
        results: a dictionary containing the results for all FS methods. 
        label: A title for the first column, defaults to Method.     
    """
    table = PrettyTable([label, 'Best K', 'Train', 'Test'])

    for name, result in results.items():
        k, train, test = list(zip(*result))
        best_k = np.argmax(test)
        vals = [k[best_k], train[best_k], test[best_k]]
        row = ['%.4f' % elem if i != 0 else elem for i, elem in enumerate(vals) ]
        table.add_row(np.concatenate([[name], row]))

    k, train, test = results['mrmr'][-1]
    vals = [k, train, test]
    row = ['%.4f' % elem if i != 0 else elem for i, elem in enumerate(vals) ]
    table.add_row(np.concatenate([['full'], row]))

    print('\n') # tqdm messses with table border.
    print(table)