# -*- coding: utf-8 -*-
"""
Particle Swarm Optimization (PSO) for feature weighting.
Standardized to provide a scikit-learn compatible interface.
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
    """
    Scikit-learn compatible wrapper for Particle Swarm Optimization.
    """

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

        def initParticle(pcls, size, pmin, pmax, smin, smax):
            part = pcls(random.uniform(pmin, pmax) for _ in range(size))
            part.speed = [random.uniform(smin, smax) for _ in range(size)]
            part.smin = smin
            part.smax = smax
            return part

        def updateParticle(part, best, phi1, phi2):
            u1 = (random.uniform(0, phi1) for _ in range(len(part)))
            u2 = (random.uniform(0, phi2) for _ in range(len(part)))
            v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
            v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
            part.speed = list(
                map(operator.add, part.speed, map(operator.add, v_u1, v_u2))
            )
            for i, speed in enumerate(part.speed):
                if speed < part.smin:
                    part.speed[i] = part.smin
                elif speed > part.smax:
                    part.speed[i] = part.smax
            part[:] = list(map(operator.add, part, part.speed))

        toolbox = base.Toolbox()
        toolbox.register(
            "particle",
            initParticle,
            creator.Particle,
            size=n_features,
            pmin=0,
            pmax=1,
            smin=-0.1,
            smax=0.1,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.particle)
        toolbox.register("update", updateParticle, phi1=self.phi1, phi2=self.phi2)

        def evaluate(part):
            weights = np.array(part)
            # Clip weights to [0, 1]
            weights = np.clip(weights, 0, 1)
            X_weighted = X * weights
            clf = KNeighborsClassifier(n_neighbors=3)
            split_idx = int(len(X) * 0.8)
            clf.fit(X_weighted[:split_idx], y[:split_idx])
            score = balanced_accuracy_score(
                y[split_idx:], clf.predict(X_weighted[split_idx:])
            )
            return (score,)

        toolbox.register("evaluate", evaluate)

        pop = toolbox.population(n=self.population_size)
        best = None

        for _ in range(self.iterations):
            for part in pop:
                part.fitness.values = toolbox.evaluate(part)
                if not part.best or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if not best or best.fitness < part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values
            for part in pop:
                toolbox.update(part, best)

        self.best_individual = np.clip(np.array(best), 0, 1)

        # Fit a final classifier on the full weighted data
        self.clf = KNeighborsClassifier(n_neighbors=3)
        self.clf.fit(X * self.best_individual, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X * self.best_individual)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.best_individual
