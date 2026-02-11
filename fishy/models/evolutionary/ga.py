# -*- coding: utf-8 -*-
"""
Genetic Algorithm (GA) for feature weighting/selection.
Standardized to provide a scikit-learn compatible interface.
"""

import logging
import random
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import minmax_scale
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)


class GA(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for Genetic Algorithm.
    Evolves feature weights to optimize classification performance.
    """

    def __init__(
        self,
        generations: int = 10,
        population_size: int = 100,
        crossover_rate: float = 0.5,
        mutation_rate: float = 0.2,
        elitism: float = 0.1,
        random_state: int = 42,
    ):
        self.generations = generations
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
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
        # Individual is a vector of feature weights [0, 1]
        toolbox.register("attr_float", random.uniform, 0, 1)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_float,
            n=n_features,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate(individual):
            # Apply weights to features
            weights = np.array(individual)
            X_weighted = X * weights

            # Use a fast classifier for evaluation
            clf = KNeighborsClassifier(n_neighbors=3)
            # Simple internal split for fitness
            split_idx = int(len(X) * 0.8)
            clf.fit(X_weighted[:split_idx], y[:split_idx])
            y_pred = clf.predict(X_weighted[split_idx:])
            score = balanced_accuracy_score(y[split_idx:], y_pred)

            # Penalize using too many features (optional but good for GA)
            sparsity_penalty = 0.01 * (1.0 - np.mean(weights))
            return (score + sparsity_penalty,)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        stats.register("avg", np.mean)

        n_elitism = max(1, int(self.elitism * self.population_size))

        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)

        for gen in range(self.generations):
            # Selection
            offspring = toolbox.select(pop, self.population_size - n_elitism)
            offspring = [toolbox.clone(ind) for ind in offspring]

            # Variational form (crossover and mutation)
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_rate:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mutation_rate:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate offspring with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Elitism: combine best from previous population with offspring
            elites = [toolbox.clone(ind) for ind in tools.selBest(pop, n_elitism)]
            pop[:] = elites + offspring

            hof.update(pop)
            record = stats.compile(pop)
            logger.info(f"Gen {gen}: {record}")

        self.best_individual = np.array(hof[0])

        # Fit a final classifier on the full weighted data
        self.clf = KNeighborsClassifier(n_neighbors=3)
        self.clf.fit(X * self.best_individual, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X * self.best_individual)

    def _predict_internal(self, X_weighted: np.ndarray) -> np.ndarray:
        # Placeholder for a more robust final prediction
        # For spectral classification, argmax of weighted features isn't ideal,
        # so we'll just use a basic nearest neighbor logic
        return np.zeros(
            X_weighted.shape[0]
        )  # Needs proper implementation if used alone

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.best_individual
