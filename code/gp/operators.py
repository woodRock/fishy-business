import random
import copy
from functools import wraps, partial
from deap import gp

def xmate(ind1, ind2):
    """ Reproduction operator for multi-tree GP, where trees are represented as a list.

    Crossover happens to a subtree that is selected at random.
    Crossover operations are limited to parents from the same tree.

    FIXME: Have to compile the trees (manually), which is frustrating.

    Args:
        ind1 (Individual): The first parent.
        ind2 (Individual): The second parent

    Returns:
        ind1, ind2 (Individual, Individual): The children from the parents reproduction.
    """
    n = range(len(ind1))
    selected_tree_idx = random.choice(n)
    for tree_idx in n:
        g1, g2 = gp.PrimitiveTree(ind1[tree_idx]), gp.PrimitiveTree(ind2[tree_idx])
        if tree_idx == selected_tree_idx:
            ind1[tree_idx], ind2[tree_idx] = gp.cxOnePoint(g1, g2)
        else:
            ind1[tree_idx], ind2[tree_idx] = g1, g2
    return ind1, ind2


def xmut(ind, expr, pset=None):
    """ Mutation operator for multi-tree GP, where trees are represented as a list.

    Mutation happens to a tree selected at random, when an individual is selected for crossover.

    FIXME: Have to compile the trees (manually), which is frustrating.

    Args:
        ind: The individual, a list of GP trees.
    """
    n = range(len(ind))
    selected_tree_idx = random.choice(n)
    for tree_idx in n:
        g1 = gp.PrimitiveTree(ind[tree_idx])
        if tree_idx == selected_tree_idx:
            indx = gp.mutUniform(g1, expr, pset)
            ind[tree_idx] = indx[0]
        else:
            ind[tree_idx] = g1
    return ind,

def staticLimit(key, max_value):
    """
    A variation of gp.staticLimit that works for Multi-tree representation.
    This works for our altered xmut and xmate genetic operators for mutli-tree GP.
    If tree depth limit is exceeded, the genetic operator is reverted.

    When an invalid (over the limit) child is generated,
    it is simply replaced by one of its parents, randomly selected.

    Args:
        key: The function to use in order the get the wanted value. For
             instance, on a GP tree, ``operator.attrgetter('height')`` may
             be used to set a depth limit, and ``len`` to set a size limit.
        max_value: The maximum value allowed for the given measurement.
             Defaults to 17, the suggested value in (Koza 1992)

    Returns:
        A decorator that can be applied to a GP operator using \
        :func:`~deap.base.Toolbox.decorate`

    References:
        1. Koza, J. R. G. P. (1992). On the programming of computers by means
            of natural selection. Genetic programming.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds = [[copy.deepcopy(tree) for tree in ind] for ind in args]
            new_inds = list(func(*args, **kwargs))
            for ind_idx, ind in enumerate(new_inds):
                for tree_idx, tree in enumerate(ind):
                    if key(tree) > max_value:
                        random_parent = random.choice(keep_inds)
                        new_inds[ind_idx][tree_idx] = random_parent[tree_idx]
            return new_inds
        return wrapper
    return decorator