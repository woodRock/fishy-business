# -*- coding: utf-8 -*-
"""
Genetic Programming (GP) engine for multi-tree evolution.
Standardized to provide a scikit-learn compatible interface.
"""

import logging
import pickle
import random
import copy
import operator
from functools import wraps
from typing import Iterable, Union, Tuple, List, Optional, Any

import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from deap import tools, algorithms, gp, base, creator
from deap.tools import Logbook, HallOfFame
from deap.gp import (
    PrimitiveTree,
    Primitive,
    Terminal,
    PrimitiveSetTyped,
    genHalfAndHalf,
)
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import minmax_scale
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)

# --- Specialized Multi-Tree Operators ---

def xmate(ind1: List[PrimitiveTree], ind2: List[PrimitiveTree]) -> Tuple[List[PrimitiveTree], List[PrimitiveTree]]:
    """Reproduction operator for multi-tree GP individuals."""
    n = range(len(ind1))
    selected_tree_idx = random.choice(n)
    for tree_idx in n:
        g1, g2 = gp.PrimitiveTree(ind1[tree_idx]), gp.PrimitiveTree(ind2[tree_idx])
        if tree_idx == selected_tree_idx:
            ind1[tree_idx], ind2[tree_idx] = gp.cxOnePoint(g1, g2)
        else:
            ind1[tree_idx], ind2[tree_idx] = g1, g2
    return ind1, ind2


def xmut(individual: List[PrimitiveTree], expr: Any, pset: PrimitiveSetTyped) -> Tuple[List[PrimitiveTree]]:
    """Mutation operator for multi-tree GP individuals."""
    n = range(len(individual))
    selected_tree_idx = random.choice(n)
    for tree_idx in n:
        g1 = gp.PrimitiveTree(individual[tree_idx])
        if tree_idx == selected_tree_idx:
            indx = gp.mutUniform(g1, expr, pset)
            individual[tree_idx] = indx[0]
        else:
            individual[tree_idx] = g1
    return (individual,)


def staticLimit(key: Any, max_value: int) -> Any:
    """Depth limit decorator for multi-tree genetic operators."""
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

# --- Optimized Evaluation Utilities ---

def quick_evaluate(expr: PrimitiveTree, pset: PrimitiveSetTyped, data: np.ndarray, prefix: str = "ARG") -> np.ndarray:
    """Optimized stack-based evaluation of GP trees."""
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
            else: raise Exception
            if len(stack) == 0: break
            stack[-1][1].append(result)
    return result


def compileMultiTree(expr: List[PrimitiveTree], pset: PrimitiveSetTyped, X: np.ndarray) -> np.ndarray:
    """Compiles a multi-tree individual into a feature matrix."""
    funcs = [quick_evaluate(gp.PrimitiveTree(sub), pset, X) for sub in expr]
    return np.array(funcs).T

# --- Fitness Measures ---

def normalized_distances(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Computes normalized means of intraclass and interclass distances."""
    dist_mat = squareform(pdist(X))
    mask = np.equal.outer(y, y)
    intra = dist_mat[np.triu(mask, k=1)]
    inter = dist_mat[np.triu(~mask, k=1)]
    intra_mean = np.mean(minmax_scale(intra)) if len(intra) > 0 else 0.0
    inter_mean = np.mean(minmax_scale(inter)) if len(inter) > 0 else 0.0
    return intra_mean, inter_mean


def wrapper_fitness(X: np.ndarray, y: np.ndarray) -> float:
    """Calculates weighted fitness based on accuracy and distance metrics."""
    X_norm = minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
    y_pred = np.argmax(X_norm, axis=1)
    acc = balanced_accuracy_score(y, y_pred)
    intra, inter = normalized_distances(X_norm, y)
    dist = 0.5 * (1 - intra) + 0.5 * inter
    return float(0.8 * acc + 0.2 * dist)

# --- Classifier Interface ---

class GP(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for Multi-tree Genetic Programming.
    """
    def __init__(
        self,
        generations: int = 10,
        population_size: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        elitism: float = 0.1,
        max_depth: int = 6,
        random_state: int = 42
    ):
        self.generations = generations
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.max_depth = max_depth
        self.random_state = random_state
        self.best_individual = None
        self.pset = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        
        self.pset = PrimitiveSetTyped("main", [float] * n_features, float)
        self.pset.addPrimitive(lambda l, r: np.divide(l, r, out=np.ones_like(l), where=r != 0), [float, float], float, name="/")
        self.pset.addPrimitive(lambda x, y: x + y, [float, float], float, name="+")
        self.pset.addPrimitive(lambda x, y: x * y, [float, float], float, name="x")
        self.pset.addPrimitive(lambda x, y: x - y, [float, float], float, name="-")
        self.pset.addPrimitive(lambda x: -x, [float], float, name="-1*")

        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.expr, n=n_classes)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", compileMultiTree, pset=self.pset)
        
        def evaluate(ind):
            feat = compileMultiTree(ind, self.pset, X)
            return (wrapper_fitness(feat, y),)
            
        toolbox.register("evaluate", evaluate)
        toolbox.register("select", tools.selTournament, tournsize=7)
        toolbox.register("mate", xmate)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", xmut, expr=toolbox.expr_mut, pset=self.pset)
        toolbox.decorate("mate", staticLimit(key=lambda t: t.height, max_value=self.max_depth))
        toolbox.decorate("mutate", staticLimit(key=lambda t: t.height, max_value=self.max_depth))

        pop = toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(max(1, int(self.elitism * self.population_size)))
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)

        algorithms.eaSimple(pop, toolbox, self.crossover_rate, self.mutation_rate, self.generations, stats=stats, halloffame=hof, verbose=False)
        
        self.best_individual = hof[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        features = compileMultiTree(self.best_individual, self.pset, X)
        return np.argmax(minmax_scale(features, axis=0), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        features = compileMultiTree(self.best_individual, self.pset, X)
        return minmax_scale(features, axis=0)
