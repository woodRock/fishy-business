import numpy as np
import matplotlib.pyplot as plt
from deap import gp
import pygraphviz as pgv


def plot_class_imbalance(y, filename=None):
    """ 
    Plot the class distribution to see if there is a class imbalance. 

    Args:
        y: The labels.
        filename: Saves the output to file if filename is provided. Defaults to None.
    """
    labels = np.unique(y)
    cts = []
    for label in np.unique(y): 
        cts.append(list(y).count(label))

    plt.bar(labels, cts)

    if filename is not None:
        plt.savefig(filename, dpi=500)

    plt.show()
    

def plot_tree(hof, filename="tree.pdf"):
    """
    Plot the GP tree. Saves the output to a pdf file.

    Args:
        hof: The hall of fame. The best indivuals in the population. 
        filename: The name of the file to save the tree to.
    """
    expr = hof[0]
    nodes, edges, labels = gp.graph(expr)

    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw(filename)