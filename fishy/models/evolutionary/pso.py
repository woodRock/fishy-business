# -*- coding: utf-8 -*-
"""
Particle Swarm Optimization (PSO) for feature weighting.
"""

import logging
import random
import operator
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from deap import base, creator, tools
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)


class PSO(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible wrapper for Particle Swarm Optimization."""

    def __init__(
        self,
        iterations: int = 10,
        population_size: int = 50,
        phi1: float = 2.0,
        phi2: float = 2.0,
        random_state: int = 42,
    ):
        self.iterations = iterations
        self.population_size = population_size
        self.phi1 = phi1
        self.phi2 = phi2
        self.random_state = random_state
        self.best_individual = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        n_features = X.shape[1]
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Particle"):
            creator.create(
                "Particle",
                list,
                fitness=creator.FitnessMax,
                speed=list,
                smin=None,
                smax=None,
                best=None,
            )

        def initPart(pcls, size, pmin, pmax, smin, smax):
            part = pcls(random.uniform(pmin, pmax) for _ in range(size))
            part.speed = [random.uniform(smin, smax) for _ in range(size)]
            part.smin, part.smax = smin, smax
            return part

        def updatePart(part, best, phi1, phi2):
            u1, u2 = (random.uniform(0, phi1) for _ in range(len(part))), (
                random.uniform(0, phi2) for _ in range(len(part))
            )
            v_u1, v_u2 = map(operator.mul, u1, map(operator.sub, part.best, part)), map(
                operator.mul, u2, map(operator.sub, best, part)
            )
            part.speed = list(
                map(operator.add, part.speed, map(operator.add, v_u1, v_u2))
            )
            for i, s in enumerate(part.speed):
                part.speed[i] = max(part.smin, min(part.smax, s))
            part[:] = list(map(operator.add, part, part.speed))

        toolbox = base.Toolbox()
        toolbox.register(
            "particle",
            initPart,
            creator.Particle,
            size=n_features,
            pmin=0,
            pmax=1,
            smin=-0.1,
            smax=0.1,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.particle)
        toolbox.register("update", updatePart, phi1=self.phi1, phi2=self.phi2)

        def evaluate(part):
            w = np.clip(np.array(part), 0, 1)
            X_w = X * w
            clf = KNeighborsClassifier(n_neighbors=3)
            split = int(len(X) * 0.8)
            clf.fit(X_w[:split], y[:split])
            return (balanced_accuracy_score(y[split:], clf.predict(X_w[split:])),)

        toolbox.register("evaluate", evaluate)
        pop = toolbox.population(n=self.population_size)
        best = None
        for _ in range(self.iterations):
            for p in pop:
                p.fitness.values = toolbox.evaluate(p)
                if not p.best or p.best.fitness < p.fitness:
                    p.best = creator.Particle(p)
                    p.best.fitness.values = p.fitness.values
                if not best or best.fitness < p.fitness:
                    best = creator.Particle(p)
                    best.fitness.values = p.fitness.values
            for p in pop:
                toolbox.update(p, best)

        self.best_individual = np.clip(np.array(best), 0, 1)
        self.clf = KNeighborsClassifier(n_neighbors=3)
        self.clf.fit(X * self.best_individual, y)
        self.classes_ = self.clf.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X * self.best_individual)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability estimates for LIME support."""
        return self.clf.predict_proba(X * self.best_individual)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.best_individual
