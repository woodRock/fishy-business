import logging
import numpy as np
from scipy.spatial.distance import pdist, squareform
from deap import gp
from deap.gp import (
    PrimitiveTree,
    Primitive,
    Terminal,
    PrimitiveSetTyped,
    genHalfAndHalf,
)
from deap.base import Toolbox
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import balanced_accuracy_score
from typing import Iterable, Tuple, Any, List, Optional


def quick_evaluate(
    expr: PrimitiveTree, pset: PrimitiveSetTyped, data: np.ndarray, prefix: str = "ARG"
) -> np.ndarray:
    """
    Optimized evaluation of GP trees using a stack-based approach.

    Provides a significant speedup over the default `gp.compile` implementation
    by avoiding expensive string conversions and dynamic code execution.

    Args:
        expr (PrimitiveTree): The uncompiled GP tree.
        pset (PrimitiveSetTyped): The primitive set used by the tree.
        data (np.ndarray): The dataset to evaluate on.
        prefix (str, optional): Variable argument prefix in the pset. Defaults to "ARG".

    Returns:
        np.ndarray: The result of the GP tree evaluation.
    """
    result = None
    stack = []
    for node in expr:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            if isinstance(prim, Primitive):
                result = pset.context[prim.name](*args)
            elif isinstance(prim, Terminal):
                if prefix in prim.name:
                    result = data[:, int(prim.name.replace(prefix, ""))]
                else:
                    result = prim.value
            else:
                raise Exception
            if len(stack) == 0:
                break
            stack[-1][1].append(result)
    return result


def compileMultiTree(
    expr: List[PrimitiveTree], pset: PrimitiveSetTyped, X: np.ndarray
) -> np.ndarray:
    """
    Compiles a multi-tree individual into a feature matrix.

    Args:
        expr (List[PrimitiveTree]): A list of GP trees (one per target feature).
        pset (PrimitiveSetTyped): The primitive set.
        X (np.ndarray): Input feature matrix.

    Returns:
        np.ndarray: The constructed features of shape (n_samples, n_trees).
    """
    funcs = []
    for subexpr in expr:
        gp_tree = gp.PrimitiveTree(subexpr)
        func = quick_evaluate(gp_tree, pset, X, prefix="ARG")
        funcs.append(func)

    features = np.array(funcs).T
    return features


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Min-max normalizes a feature matrix.

    Args:
        x (np.ndarray): The feature matrix to normalize.

    Returns:
        np.ndarray: Normalized feature matrix.
    """
    x_norm = minmax_scale(x, feature_range=(0, 1), axis=0, copy=True)
    return x_norm


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    """
    Computes pairwise Euclidean distances for all points in X.

    Args:
        X (np.ndarray): Input matrix.

    Returns:
        np.ndarray: Square distance matrix.
    """
    return squareform(pdist(X))


def class_distances(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes intraclass and interclass distances.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (intra_distances, inter_distances)
    """
    distances = pairwise_distances(X)
    same_class_mask = np.equal.outer(y, y)
    intra_distances = distances[np.triu(same_class_mask, k=1)]
    inter_distances = distances[np.triu(~same_class_mask, k=1)]
    return intra_distances, inter_distances


def normalized_distances(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Computes normalized means of intraclass and interclass distances.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.

    Returns:
        Tuple[float, float]: (intra_mean, inter_mean)
    """
    intra_distances, inter_distances = class_distances(X, y)

    if len(intra_distances) > 0:
        intra_normalized = minmax_scale(intra_distances)
        intra_mean = np.mean(intra_normalized)
    else:
        intra_mean = 0.0

    if len(inter_distances) > 0:
        inter_normalized = minmax_scale(inter_distances)
        inter_mean = np.mean(inter_normalized)
    else:
        inter_mean = 0.0

    return intra_mean, inter_mean


def wrapper_classification_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 2,
    verbose: bool = False,
    is_normalize: bool = True,
) -> float:
    """
    Evaluates fitness based on classification accuracy and distance regularization.

    Args:
        X (np.ndarray): Constructed features.
        y (np.ndarray): Target labels.
        k (int, optional): Folds for cross-validation. Defaults to 2.
        verbose (bool, optional): If True, logs accuracy. Defaults to False.
        is_normalize (bool, optional): Whether to normalize features. Defaults to True.

    Returns:
        float: The final fitness score in [0, 1].
    """
    logger = logging.getLogger(__name__)

    if is_normalize:
        X = normalize(X)
    
    y_predict = [np.argmax(x) for x in X]
    train_accuracy = balanced_accuracy_score(y, y_predict)

    if verbose:
        logger.info("Balanced Accuracy: %f", train_accuracy)

    train_intraclass_distance, train_interclass_distance = normalized_distances(X, y)

    alpha = 0.5
    beta = 0.8
    train_distance = (
        alpha * (1 - train_intraclass_distance)
        + (1 - alpha) * train_interclass_distance
    )

    fitness = beta * train_accuracy + (1 - beta) * train_distance
    return float(fitness)


def evaluate_classification(
    individual: List[Any],
    X: np.ndarray,
    verbose: bool = False,
    toolbox: Optional[Toolbox] = None,
    pset: Optional[PrimitiveSetTyped] = None,
    y: np.ndarray = None,
) -> Tuple[float]:
    """
    Calculates the fitness of a GP individual.

    Args:
        individual (List[Any]): A candidate solution.
        X (np.ndarray): Input features.
        verbose (bool, optional): Verbosity flag. Defaults to False.
        toolbox (Optional[Toolbox], optional): DEAP toolbox. Defaults to None.
        pset (Optional[PrimitiveSetTyped], optional): Primitive set. Defaults to None.
        y (np.ndarray, optional): Target labels. Defaults to None.

    Returns:
        Tuple[float]: The individual's fitness.
    """
    features = toolbox.compile(expr=individual, X=X, pset=pset)
    fitness = wrapper_classification_accuracy(X=features, y=y, verbose=verbose)
    return (fitness,)
