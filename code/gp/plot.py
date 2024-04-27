
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
from deap import base, creator, gp
import pygraphviz as pgv


def plot_tsne(X, features, file_path="figures/tsne.png"):
    logger = logging.getLogger(__name__)
    # Perform t-SNE dimensionality reduction
    perplexity = 10
    features = toolbox.compile(expr=hof[0], pset=pset)
    evaluate_classification(hof.items[0], verbose=True)

    for X_set in [X, features]:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X_set)

        # Plot the data
        plt.figure(figsize=(10, 8))

        labels = ['Fillet','Heads','Livers','Skins','Guts','Frames']

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


def plot_pca_3D(X, features,  file_path="figures/pca_3D.png"):
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

        labels = ['Fillet','Heads','Livers','Skins','Guts','Frames']

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


def plot_tsne_3D(X, features, file_path="figures/tsne_3D.png"):
    for X_set in [X, features]:
        # Perform t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, perplexity=10, random_state=42)
        X_tsne = tsne.fit_transform(X_set)

        # Plot the data
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        labels = ['Fillet','Heads','Livers','Skins','Guts','Frames']

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
     
def plot_pair_plot(features, file_path="figures/pairplot.png"):
    feature_no = 2
    data = pd.DataFrame(features[:,:feature_no], columns=[f'feature_{i}' for i in range(feature_no)])
    data['class'] = y[:]

    # Add class labels to the DataFrame
    labels = ['Hoki','Mackerel']

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

def plot_evolutionary_process(fitness, file_path="figures/pairplot.png"):
    plt.plot(fitness)
    plt.title("Fitness: evolutionary process")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.savefig(file_path)
    plt.close()
    # Show for interactive mode.
    # plt.show()

def plot_gp_tree(multi_tree=None):
    for t_idx,tree in enumerate(multi_tree):
        nodes, edges, labels = gp.graph(tree)

        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")

        for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]

        g.draw(f"tree-{t_idx}.pdf")