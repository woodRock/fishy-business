
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
from deap import gp
import pygraphviz as pgv
from deap.base import Toolbox
from typing import Iterable

def plot_tsne(
        X: Iterable, 
        features: Iterable, 
        y: Iterable,
        dataset: str = "species",
        file_path: str ="figures/tsne.png", 
        toolbox: Toolbox = None
    ) -> None:
    """ Plot a 2D t-SNE of the original and constructed features.
    
    Args: 
        X (Iterable): the original feature set.
        features (Iterable): the constructed features.
        dataset (str): The fish species, part, oil or cross-species dataset. Defualts to species.
        file_path (str): The filepath to store the figure. Defaults to "figures/tsne.png"
        toolbox: (deap.base.Toolbox): the toolbox contains the terminal and function set.
    """
    logger = logging.getLogger(__name__)
    # Perform t-SNE dimensionality reduction
    perplexity = 10

    for X_set in [X, features]:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X_set)

        # Plot the data
        plt.figure(figsize=(10, 8))

        # Labels for the graph legend are provided for each task.
        labels = []
        if dataset == "species":
            labels = ['Hoki', 'Mackerl']
        elif dataset == "part":
            labels = ['Fillet','Heads','Livers','Skins','Guts','Frames']
        elif dataset == "oil":
            labels = ["Oil", "None"]
        elif dataset == "cross-species": 
            labels = ['Hoki-Mackerel', 'Hoki', 'Mackerel']
        else: 
            raise ValueError(f"Invalid dataset: {dataset}")

        # Plot points belonging to different classes with different colors
        for idx, label in enumerate(np.unique(y)):
            plt.scatter(X_tsne[y == label, 0], X_tsne[y == label, 1], label=labels[idx])

        plt.title('t-SNE Visualization of Fish Parts Dataset')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()

        logger.info(f"Saving t-SNE to file: {file_path}")
        plt.savefig(file_path)
        plt.close()
        # Show for interactive mode.
        # plt.show()


def plot_pca_3D(
        X: Iterable, 
        y: Iterable, 
        features: Iterable, 
        dataset: str = "species",
        file_path: str ="figures/pca_3D.png", 
        toolbox: Toolbox = None
    ) -> None:
    """ Plot a 3D PCA of the original and constructed features.
    
    Args: 
        X (Iterable): the features for the orginal dataset.
        y (Iterable): the class labels for the dataset.
        features (Iterable): the constructed features from genetic programming.
        dataset (str): The fish species, part, oil or cross-species dataset. Defualts to species.
        file_path (str): The filepath to store the figure. Defaults to "figures/pca_3D.png"
        toolbox: (deap.base.Toolbox): the toolbox contains the terminal and function set.
    """
    logger = logging.getLogger(__name__)
    # Assuming you have your own dataset with features X and labels y
    # Replace X and y with your actual data
    # X should be your feature matrix and y should be your labels

    for X_set in [X, features]:
        # Perform PCA dimensionality reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_set)

        # Plot the data
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Labels for the graph legend are provided for each task.
        labels = []
        if dataset == "species":
            labels = ['Hoki', 'Mackerl']
        elif dataset == "part":
            labels = ['Fillet','Heads','Livers','Skins','Guts','Frames']
        elif dataset == "oil":
            labels = ["Oil", "None"]
        elif dataset == "cross-species": 
            labels = ['Hoki-Mackerel', 'Hoki', 'Mackerel']
        else: 
            raise ValueError(f"Invalid dataset: {dataset}")

        # Plot points belonging to different classes with different colors
        for idx, label in enumerate(np.unique(y)):
            ax.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=labels[idx])

        ax.set_title('PCA Visualization of Fish Parts Dataset')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.legend()
        logger.info(f"Saving 3D pca to file: {file_path}")
        plt.savefig(file_path)
        plt.close()
        # Show for interactive mode.
        # plt.show()


def plot_tsne_3D(
        X: Iterable, 
        y: Iterable, 
        features: Iterable, 
        dataset: str = "species",
        file_path: str ="figures/tsne_3D.png", 
        toolbox: Toolbox = None
    ) -> None:
    """ Plot a 3D t-SNE of the original and constructed features.
    
    Args: 
        X (Iterable): the orginal features from the dataset.
        y (Iterable): the class labels from the dataset.
        features (Iterable): the constructed features.
        dataset (str): The fish species, part, oil or cross-species dataset. Defaults to species.
        file_path (str): The filepath to store the figure. Defaults to "figures/tsne_3D.png"
        toolbox: (deap.base.Toolbox): the toolbox contains the terminal and function set.
    """
    logger = logging.getLogger(__name__)
    for X_set in [X, features]:
        # Perform t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, perplexity=10, random_state=42)
        X_tsne = tsne.fit_transform(X_set)

        # Plot the data
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Labels for the graph legend are provided for each task.
        labels = []
        if dataset == "species":
            labels = ['Hoki', 'Mackerl']
        elif dataset == "part":
            labels = ['Fillet','Heads','Livers','Skins','Guts','Frames']
        elif dataset == "oil":
            labels = ["Oil", "None"]
        elif dataset == "cross-species": 
            labels = ['Hoki-Mackerel', 'Hoki', 'Mackerel']
        else: 
            raise ValueError(f"Invalid dataset: {dataset}")

        # Plot points belonging to different classes with different colors
        for idx, label in enumerate(np.unique(y)):
            ax.scatter(X_tsne[y == label, 0], X_tsne[y == label, 1], label=labels[idx])

        ax.set_title('t-SNE Visualization of Fish Parts Dataset')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_zlabel('t-SNE Dimension 3')
        ax.legend()
        logger.info(f"Saving 3D tsne to file: {file_path}")
        plt.savefig(file_path)
        plt.close()
        # Show for interactive mode.
        # plt.show()
    
     
def plot_pair_plot(
        features: Iterable, 
        dataset: str = "species",
        file_path: str ="figures/pairplot.png"
    ) -> None:
    """ Plot a pairplot the constructed features.
    
    Args: 
        features (Iterable): the constructed features.
        dataset (str): The fish species, part, oil or cross-species dataset. Defaults to species.
        file_path (str): The filepath to store the figure. Defaults to "figures/pairplot.png"
    """
    logger = logging.getLogger(__name__)
    feature_no = 2
    data = pd.DataFrame(features[:,:feature_no], columns=[f'feature_{i}' for i in range(feature_no)])
    data['class'] = y[:]

    # Labels for the graph legend are provided for each task.
    labels = []
    if dataset == "species":
        labels = ['Hoki', 'Mackerl']
    elif dataset == "part":
        labels = ['Fillet','Heads','Livers','Skins','Guts','Frames']
    elif dataset == "oil":
        labels = ["Oil", "None"]
    elif dataset == "cross-species": 
        labels = ['Hoki-Mackerel', 'Hoki', 'Mackerel']
    else: 
        raise ValueError(f"Invalid dataset: {dataset}")

    # Create pairplot
    plot = sns.pairplot(data, hue='class', palette='viridis')

    # Modify legend
    handles = plot._legend_data.values()
    plt.legend(handles, labels)
    logger.info(f"Saving 3D tsne to file: {file_path}")
    plt.savefig(file_path)
    plt.close()
    # Show for interactive mode.
    # plt.show()


def plot_evolutionary_process(
        fitness: Iterable, 
        file_path: str ="figures/evolutionary_process.png"
    ) -> None:
    """"
    Plot the evolutionary process for an evolved genetic program.

    Args: 
        fitness (Iterable): the set of fitness values that were evolved.
        file_path (str): The filepath where the figure is saved. Defaults to "figures/pairplot.png".
    """
    logger = logging.getLogger(__name__)
    plt.plot(fitness)
    plt.title("Fitness: evolutionary process")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    logger.info(f"Saving evolutionary process to file: {file_path}")
    plt.savefig(file_path)
    plt.close()
    # Show for interactive mode.
    # plt.show()


def plot_gp_tree(
        multi_tree: Iterable = None
    ) -> None :
    """
    Plot subtrees from a multi-tree evolved using genetic programming.

    Args:
        mutli-tree (Iterable): a solution is represented by a multi-tree.
    """
    logger = logging.getLogger(__name__)
    for t_idx,tree in enumerate(multi_tree):
        nodes, edges, labels = gp.graph(tree)

        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")

        for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]
            
        file_path = f"figures/tree-{t_idx}.pdf"
        logger.info(f"Saving tree to file: {file_path}")
        g.draw(file_path)