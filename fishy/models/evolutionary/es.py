# -*- coding: utf-8 -*-
"""
Evolution Strategy (ES) for feature weighting.
Standardized to provide a scikit-learn compatible interface.
"""

import logging
import random
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)

class ES(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for Evolution Strategy (mu + lambda).
    """
    def __init__(
        self,
        generations: int = 10,
        mu: int = 50,
        lambda_: int = 100,
        random_state: int = 42
    ):
        self.generations = generations
        self.mu = mu
        self.lambda_ = lambda_
        self.random_state = random_state
        self.best_individual = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        n_features = X.shape[1]

        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate(individual):
            weights = np.array(individual)
            X_weighted = X * weights
            clf = KNeighborsClassifier(n_neighbors=3)
            split_idx = int(len(X) * 0.8)
            clf.fit(X_weighted[:split_idx], y[:split_idx])
            score = balanced_accuracy_score(y[split_idx:], clf.predict(X_weighted[split_idx:]))
            return (score,)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        toolbox.register("select", tools.selBest)

        pop = toolbox.population(n=self.mu)
        hof = tools.HallOfFame(1)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)

        algorithms.eaMuPlusLambda(pop, toolbox, mu=self.mu, lambda_=self.lambda_, 
                                  cxpb=0.6, mutpb=0.3, ngen=self.generations, 
                                  stats=stats, halloffame=hof, verbose=False)
        
        self.best_individual = np.array(hof[0])
        
        # Fit a final classifier on the full weighted data
        self.clf = KNeighborsClassifier(n_neighbors=3)
        self.clf.fit(X * self.best_individual, y)
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X * self.best_individual)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.best_individual
