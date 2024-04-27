import logging
import numpy as np
import math
from deap import gp
from deap.gp import PrimitiveTree, Primitive, Terminal
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import balanced_accuracy_score


def quick_evaluate(expr: PrimitiveTree, pset, data, prefix='ARG'):
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
        pset: The primitive set.
        data: The dataset to evaluate the GP tree for.
        prefix: Prefix for variable arguments. Defaults to ARG.

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

def compileMultiTree(expr, pset, X=None):
    """Compile the expression represented by a list of trees.

    A variation of the gp.compileADF method, that handles Multi-tree GP.

    Args:
        expr: Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
        pset: Primitive Set

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

def normalize(x):
    """
    Normalize a numpy array to interclass/intraclass distance sum to 1.

    Args:
        x: numpy array to be normalized.

    Returns:
        numpy array normalized to interclass/intraclass distance sum to 1.
    """
    # x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
    x_norm = minmax_scale(x, feature_range=(0, 1), axis=0, copy=True)
    return x_norm

def is_same_class(a,b):
    """
    Return True if a and b are in the same class.

    Args:
        a: first point
        b: second point

    Returns:
        True if a and b are in the same class.
    """
    return a[1] == b[1]

def euclidian_distance(a,b):
    """
    Return the euclidian distance between two points.

    Args:
        a: first point
        b: second point

    Returns:
        Euclidian distance between a and b.
    """
    a,b = a[0], b[0]
    dist = np.linalg.norm(a-b)
    return dist

def intraclass_distance(_X,_y):
    """
    Return the intra-class distance for a dataset.
    The average distance between all pairs of instances that are from the same class.

    Args:
        _X: numpy array of features.
        _y: numpy array of labels.

    Returns:
        Intra-class distance for a dataset.
    """
    data = list(zip(_X, _y))
    pair_length = sum([1 if is_same_class(a,b) else 0 for idx, a in enumerate(data) for b in data[idx + 1:]])
    d = sum([euclidian_distance(a,b) if is_same_class(a,b) else 0 for idx, a in enumerate(data) for b in data[idx + 1:]]) / (pair_length * _X.shape[1])
    return d

def interclass_distance(_X,_y):
    """
    Return the inter-class distance for a dataset.
    The average distance between all pairs of instances that are from different classes.

    Args:
        _X: numpy array of features.
        _y: numpy array of labels.

    Returns:
        Inter-class distance for a dataset.
    """
    data = list(zip(_X, _y))
    pair_length = sum([1 if not is_same_class(a,b) else 0 for idx, a in enumerate(data) for b in data[idx + 1:]])
    d = sum([euclidian_distance(a,b) if not is_same_class(a,b) else 0 for idx, a in enumerate(data) for b in data[idx + 1:]]) / (pair_length * _X.shape[1])
    return d

def wrapper_classification_accuracy(X=None, y=None, k=2, verbose=False):
    """ Evaluate balanced classification accuracy over stratified k-fold cross validation.

    This method is our fitness measure for an individual. We measure each individual
    based on its balanced classification accuracy using 10-fold cross-validation on
    the training set.

    If verbose, we evaluate performance on the test set as well, and print the results
    to the standard output. By default, only the train set is evaluated, which
    corresponds to a 2x speedup for training, when compared to the verbose method.

    Args:
        X: entire dataset, train and test.
        k: Number of folds, for cross validation. Defaults to 10.
        verbose: If true, prints stuff. Defaults to false.

    Returns:
        Average balanced classification accuracy with 10-fold CV on training set.
    """
    logger = logging.getLogger(__name__)

    train_accs = []
    val_accs = []
    test_accs = []

    # Reserve a test set that is not touched by the training algorithm.
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Stratified k-fold validation.
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(X_train,y_train):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        # Normalize features to interclass/intraclass distance sum to 1.
        X_train = normalize(X_train)
        # Class-dependent multi-tree embedded GP (Tran 2019).
        y_predict = [np.argmax(x) for x in X_train]
        train_acc = balanced_accuracy_score(y_train, y_predict)
        train_accs.append(train_acc)

        # 2x speedup: only evaluate test set in verbose mode.
        if verbose:
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
    # Distance-based regularization method, intra/inter class distance.
    train_intraclass_distance = intraclass_distance(X_train, y_train)
    train_interclass_distance = interclass_distance(X_train, y_train)

    # 2x speedup: only evaluate test set in verbose mode.
    if verbose:
        # Mean and standard deviation for val accuracy.
        val_accuracy = np.mean(val_accs)
        val_std = np.std(val_accs)
        # Mean and standard deviation for test accuracy.
        test_accuracy = np.mean(test_accs)
        test_std = np.std(test_accs)
        # Distance-based regularization method, intra/inter class distance.
        val_intraclass_distance = intraclass_distance(X_val, y_val)
        val_interclass_distance = interclass_distance(X_val, y_val)
        # Distance-based regularization method, intra/inter class distance.
        test_intrarclass_distance = intraclass_distance(X_test, y_test)
        test_interclass_distance = interclass_distance(X_test, y_test)
        # When verbose, give a full evaluation for an individual.
        logger.info(f"Train accuracy: {train_accuracy} +- {train_std}")
        logger.info(f"Val accuracy: {val_accuracy} +- {val_std}")
        logger.info(f"Test accuracy: {test_accuracy} +- {test_std}")

        logger.info(f"Train intra-class: {train_intraclass_distance}, Train Inter-class: {train_interclass_distance}")
        logger.info(f"Val intra-class: {val_intraclass_distance}, Val Inter-class: {val_interclass_distance}")
        logger.info(f"Test inter-class: {test_intrarclass_distance}, Test inter-class: {test_interclass_distance}")

    # Alpha balances the inter-class/intra-class distance.
    alpha = 0.5
    #  Beta balances the accuracy and distance regularization term.
    beta = 0.8
    train_distance = alpha * (1 - train_intraclass_distance) + alpha * train_interclass_distance

    fitness = beta * train_accuracy + (1 - beta) * train_distance

    # Fitness value must be a tuple.
    assert fitness <= 1, f"fitness {fitness} should be normalized, and cannot exceed 1"
    if fitness > 1:
        logger.info(f"Train intra-class: {train_intraclass_distance}")
        logger.info(f"Train inter-class: {train_interclass_distance}")

    return fitness

def evaluate_classification(individual, alpha = 0.9, verbose=False, toolbox=None, pset=None, X=None, y=None):
    """
    Evalautes the fitness of an individual for multi-tree GP multi-class classification.

    We maxmimize the fitness when we evaluate the accuracy + regularization term.

    Args:
        individual (Individual): A candidate solution to be evaluated.
        alpha (float): A parameter that balances the accuracy and regularization term. Defaults to 0.98.

    Returns:
        accuracy (tuple): The fitness of the individual.
    """
    features = toolbox.compile(expr=individual, pset=pset)
    fitness = wrapper_classification_accuracy(X=features, y=y, verbose=verbose)
    return fitness,