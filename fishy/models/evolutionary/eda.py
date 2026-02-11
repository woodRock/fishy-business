# -*- coding: utf-8 -*-
"""
Estimation of Distribution Algorithm (EDA) for feature weighting.
"""

import logging
import random
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from deap import base, creator, tools
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)

class EDA(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible wrapper for Estimation of Distribution Algorithm."""

    def __init__(self, generations: int = 10, population_size: int = 100, select_ratio: float = 0.2, elitism: float = 0.1, random_state: int = 42):
        self.generations = generations; self.population_size = population_size; self.select_ratio = select_ratio; self.elitism = elitism; self.random_state = random_state; self.best_individual = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        random.seed(self.random_state); np.random.seed(self.random_state); n_features = X.shape[1]
        if not hasattr(creator, "FitnessMax"): creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"): creator.create("Individual", list, fitness=creator.FitnessMax)
        means, stds = np.full(n_features, 0.5), np.full(n_features, 0.2)
        def sample_individual(): return creator.Individual(np.clip(np.random.normal(means, stds), 0, 1).tolist())
        toolbox = base.Toolbox(); toolbox.register("individual", sample_individual); toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate(ind):
            w = np.array(ind); X_w = X * w; clf = KNeighborsClassifier(n_neighbors=3); split = int(len(X) * 0.8); clf.fit(X_w[:split], y[:split])
            return (balanced_accuracy_score(y[split:], clf.predict(X_w[split:])),)

        toolbox.register("evaluate", evaluate); pop = toolbox.population(n=self.population_size); hof = tools.HallOfFame(1); n_select = int(self.population_size * self.select_ratio); n_elites = max(1, int(self.elitism * self.population_size))
        for gen in range(self.generations):
            for ind in pop:
                if not ind.fitness.valid: ind.fitness.values = toolbox.evaluate(ind)
            hof.update(pop); selected = tools.selBest(pop, n_select); selected_np = np.array(selected); means = np.mean(selected_np, axis=0); stds = np.std(selected_np, axis=0) + 1e-6
            pop[:] = [toolbox.clone(i) for i in tools.selBest(pop, n_elites)] + toolbox.population(n=self.population_size - n_elites)
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
