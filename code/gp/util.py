import logging
import numpy as np
from scipy.spatial.distance import pdist, squareform
from deap import gp
from deap.gp import PrimitiveTree, Primitive, Terminal, PrimitiveSetTyped, genHalfAndHalf
from deap.base import Toolbox
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import balanced_accuracy_score
from typing import Iterable, Tuple


def quick_evaluate(
        expr: PrimitiveTree, 
        pset: PrimitiveSetTyped, 
        data: Iterable, 
        prefix: str = 'ARG'
    ) -> Iterable:
    """ Quick evaluate offers a 500% speedup for the evluation of GP trees.

    The default implementation of gp.compile provided by the DEAP library is
    horrendously inefficient. (Zhang 2022) has shared his code which leads to a
    5x speedup in the compilation and evaluation of GP trees when compared to the
    standard library approach.

    For multi-tree GP, this speedup factor is invaluable! As each individual conists
    of m trees. For the fish dataset we have 4 classes, each with 3 constructed features,
    which corresponds to 4 classes x 3 features = 12 trees for each individual.
    12 trees x 500% speedup = 6,000% overall speedup, or 60 times faster.
    The 500% speedup is fundamental, for efficient evaluation of multi-tree GP.

    Args:
        expr (PrimitiveTree): The uncompiled (gp.PrimitiveTree) GP tree.
        pset (PrimitiveSetTyped): The primitive set.
        data (Iterable): The dataset to evaluate the GP tree for.
        prefix (str): Prefix for variable arguments. Defaults to ARG.

    Returns:
        The (array-like) result of the GP tree evaluate on the dataset .
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
                    result = data[:, int(prim.name.replace(prefix, ''))]
                else:
                    result = prim.value
            else:
                raise Exception
            if len(stack) == 0:
                break # If stack is empty, all nodes should have been seen
            stack[-1][1].append(result)
    return result

def compileMultiTree(
        expr: genHalfAndHalf, 
        pset: PrimitiveSetTyped, 
        X: Iterable = None
    ) -> Iterable:
    """Compile the expression represented by a list of trees.

    A variation of the gp.compileADF method, that handles Multi-tree GP.

    Args:
        expr (genHalfAndHalf): Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
        pset (PrimitiveSetTyped): Primitive Set
        X (Iterable): the features from the dataset.

    Returns:
        A set of functions that correspond for each tree in the Multi-tree.
    """
    funcs = []
    gp_tree = None
    func = None

    for subexpr in expr:
        gp_tree = gp.PrimitiveTree(subexpr)
        # 5x speedup by manually parsing GP tree (Zhang 2022) https://mail.google.com/mail/u/0/#inbox/FMfcgzGqQmQthcqPCCNmstgLZlKGXvbc
        func = quick_evaluate(gp_tree, pset, X, prefix='ARG')
        funcs.append(func)

    # Hengzhe's method returns the features in the wrong rotation for multi-tree
    features = np.array(funcs).T
    return features

def normalize(
        x: Iterable
    ) -> Iterable:
    """
    Normalize a numpy array to interclass/intraclass distance sum to 1.

    Args:
        x (Iterable): numpy array to be normalized.

    Returns:
        numpy array normalized to interclass/intraclass distance sum to 1.
    """
    # x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
    x_norm = minmax_scale(x, feature_range=(0, 1), axis=0, copy=True)
    return x_norm

def pairwise_distances(X):
    """
    Compute pairwise distances for all points in X.
    """
    return squareform(pdist(X))

def class_distances(X, y):
    """
    Compute intraclass and interclass distances efficiently.
    """
    distances = pairwise_distances(X)
    n = len(y)
    
    # Create a mask for same-class pairs
    same_class_mask = np.equal.outer(y, y)
    
    # Intraclass distances (upper triangle only to avoid duplicates)
    intra_distances = distances[np.triu(same_class_mask, k=1)]
    
    # Interclass distances (upper triangle only to avoid duplicates)
    inter_distances = distances[np.triu(~same_class_mask, k=1)]
    
    return intra_distances, inter_distances

def normalized_distances(X, y):
    """
    Compute normalized intraclass and interclass distances.
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
        X: Iterable = None, 
        y: Iterable = None, 
        k: int = 2, 
        verbose: bool = False, 
        is_normalize: bool = True
    ) -> float:
    """ Evaluate balanced classification accuracy over stratified k-fold cross validation.

    This method is our fitness measure for an individual. We measure each individual
    based on its balanced classification accuracy using 10-fold cross-validation on
    the training set.

    If verbose, we evaluate performance on the test set as well, and print the results
    to the standard output. By default, only the train set is evaluated, which
    corresponds to a 2x speedup for training, when compared to the verbose method.

    Args:
        X (Iterable): the features of the evolved tree. Defaults to None.
        y (Iterable): the class labels for comparison. Defaults to None.
        k (int): Number of folds, for cross validation. Defaults to 2.
        verbose (bool): If true, prints stuff. Defaults to false.
        normalize (bool): Normalize the features in the dataset. Defaults to True.

    Returns:
        fitness (Iterable): Averaged balanced accuracy + distance metric.
    """
    logger = logging.getLogger(__name__)

    train_accs = []
    val_accs = []
    test_accs = []

    # Reserve a test set that is not touched by the training algorithm.
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    
    # Shuffle the datasets randomly then take 1,000 samples for evaluation.
    p = np.random.permutation(len(X_train))
    X_train, y_train = X_train[p], y_train[p]
    X_train, y_train = X_train[0:1000], y_train[0:1000]
    p = np.random.permutation(len(X_test))
    X_test, y_test = X_test[p], y_test[p]
    X_test, y_test = X_test[0:1000], y_test[0:1000]

    # Stratified k-fold validation.
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_fold, X_val = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val = y_train[train_idx], y_train[val_idx]
        
        # Normalize features to interclass/intraclass distance sum to 1.
        if is_normalize:
            X_train_fold = normalize(X_train_fold)
        # Class-dependent multi-tree embedded GP (Tran 2019).
        y_predict = [np.argmax(x) for x in X_train_fold]
        train_acc = balanced_accuracy_score(y_train_fold, y_predict)
        train_accs.append(train_acc)

        # 2x speedup: only evaluate test set in verbose mode.
        if verbose:
            if is_normalize:
                X_val, X_test = normalize(X_val), normalize(X_test)

            y_predict = [np.argmax(x) for x in X_val]
            val_acc = balanced_accuracy_score(y_val, y_predict)
            val_accs.append(val_acc)

            y_predict = [np.argmax(x) for x in X_test]
            test_acc = balanced_accuracy_score(y_test, y_predict)
            test_accs.append(test_acc)

    # Mean and standard deviation for training accuracy.
    train_accuracy = np.mean(train_accs)
    train_std = np.std(train_accs)

    # Compute distances once for the entire dataset
    train_intraclass_distance, train_interclass_distance = normalized_distances(X_train, y_train)

    # 2x speedup: only evaluate test set in verbose mode.
    if verbose:
        # Mean and standard deviation for val accuracy.
        val_accuracy = np.mean(val_accs)
        val_std = np.std(val_accs)
        # Mean and standard deviation for test accuracy.
        test_accuracy = np.mean(test_accs)
        test_std = np.std(test_accs)
        # Distance-based regularization method, intra/inter class distance.
        val_intraclass_distance, val_interclass_distance = normalized_distances(X_val, y_val)
        # Distance-based regularization method, intra/inter class distance.
        test_intrarclass_distance, test_interclass_distance = normalized_distances(X_test, y_test)
        # When verbose, give a full evaluation for an individual.
        logger.info(f"Train accuracy: {train_accuracy} +- {train_std}")
        logger.info(f"Val accuracy: {val_accuracy} +- {val_std}")
        logger.info(f"Test accuracy: {test_accuracy} +- {test_std}")

        logger.info(f"Train intra-class: {train_intraclass_distance}, Train Inter-class: {train_interclass_distance}")
        logger.info(f"Val intra-class: {val_intraclass_distance}, Val Inter-class: {val_interclass_distance}")
        logger.info(f"Test intra-class: {test_intrarclass_distance}, Test inter-class: {test_interclass_distance}")

   # Alpha balances the inter-class/intra-class distance.
    alpha = 0.5
    # Beta balances the accuracy and distance regularization term.
    beta = 0.8
    train_distance = alpha * (1 - train_intraclass_distance) + (1 - alpha) * train_interclass_distance

    fitness = beta * train_accuracy + (1 - beta) * train_distance

    # Fitness value must be a tuple.
    assert 0 <= fitness <= 1, f"fitness {fitness} should be normalized between 0 and 1"

    return fitness

def evaluate_classification(
        individual: Iterable, 
        verbose: bool = False, 
        toolbox: Toolbox = None, 
        pset: PrimitiveSetTyped = None, 
        y: Iterable = None
    ) -> Tuple[int]:
    """
    Evalautes the fitness of an individual for multi-tree GP multi-class classification.

    We maxmimize the fitness when we evaluate the accuracy + regularization term.

    Args:
        individual (Individual): A candidate solution to be evaluated.
        verbose (bool): whether or not to print the verbose output.
        toolbox (deap.base.Toolbox): the toolbox that stores all functions required for GP. Defaults to none.
        pset (deap.gp.PrimitiveSet): The set of primitives that contains the terminal and function set(s). 
        X (Iterable): the features for the dataset.
        y (Iterable): the class labels for the dataset.

    Returns:
        accuracy (tuple(int,)): The fitness of the individual.
    """
    features = toolbox.compile(expr=individual, pset=pset)
    fitness = wrapper_classification_accuracy(X=features, y=y, verbose=verbose)
    return fitness,
