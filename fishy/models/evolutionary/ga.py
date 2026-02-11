# -*- coding: utf-8 -*-
"""
Genetic Algorithm (GA) for feature weighting/selection.
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

class GA(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible wrapper for Genetic Algorithm."""

    def __init__(self, generations: int = 10, population_size: int = 100, crossover_rate: float = 0.5, mutation_rate: float = 0.2, elitism: float = 0.1, random_state: int = 42):
        self.generations = generations; self.population_size = population_size; self.crossover_rate = crossover_rate; self.mutation_rate = mutation_rate; self.elitism = elitism; self.random_state = random_state; self.best_individual = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        random.seed(self.random_state); np.random.seed(self.random_state); n_features = X.shape[1]
        if not hasattr(creator, "FitnessMax"): creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"): creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox(); toolbox.register("attr_float", random.uniform, 0, 1); toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_features); toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate(ind):
            w = np.array(ind); X_w = X * w; clf = KNeighborsClassifier(n_neighbors=3)
            split = int(len(X) * 0.8); clf.fit(X_w[:split], y[:split])
            return (balanced_accuracy_score(y[split:], clf.predict(X_w[split:])) + 0.01*(1.0 - np.mean(w)),)

        toolbox.register("evaluate", evaluate); toolbox.register("mate", tools.cxTwoPoint); toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1); toolbox.register("select", tools.selTournament, tournsize=3)
        pop = toolbox.population(n=self.population_size); hof = tools.HallOfFame(1); n_elites = max(1, int(self.elitism * self.population_size))
        for ind, fit in zip(pop, map(toolbox.evaluate, pop)): ind.fitness.values = fit
        hof.update(pop)
        for gen in range(self.generations):
            off = [toolbox.clone(ind) for ind in toolbox.select(pop, self.population_size - n_elites)]
            for c1, c2 in zip(off[::2], off[1::2]):
                if random.random() < self.crossover_rate: toolbox.mate(c1, c2); del c1.fitness.values, c2.fitness.values
            for m in off:
                if random.random() < self.mutation_rate: toolbox.mutate(m); del m.fitness.values
            for ind, fit in zip([i for i in off if not i.fitness.valid], map(toolbox.evaluate, [i for i in off if not i.fitness.valid])): ind.fitness.values = fit
            pop[:] = [toolbox.clone(i) for i in tools.selBest(pop, n_elites)] + off; hof.update(pop)
        self.best_individual = np.array(hof[0])
        self.clf = KNeighborsClassifier(n_neighbors=3); self.clf.fit(X * self.best_individual, y)
        self.classes_ = self.clf.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X * self.best_individual)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability estimates for LIME support."""
        return self.clf.predict_proba(X * self.best_individual)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.best_individual
