"""
Plot - plot.pyW
==============
A utility module for visualizing the genetic algorithm (GA) results.
"""

import matplotlib.pyplot as plt


def plot_error(errors):
    """Plot the error graph of a Genetic Algorithm (GA) training regime.

    Args:
        errors: A list or errors from GA training.
    """
    plt.plot(errors)
    plt.title("Training Error")
    plt.xlabel("Generations")
    plt.ylabel("Error")
    plt.legend()
    plt.show()
