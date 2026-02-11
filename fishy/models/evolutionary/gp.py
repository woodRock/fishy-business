# -*- coding: utf-8 -*-
"""
Core Genetic Programming (GP) algorithms and utilities.
"""

import logging
import pickle
import random
from typing import Iterable, Union, Tuple, List, Optional, Any

import numpy as np
from tqdm import tqdm
from deap import tools, algorithms
from deap.base import Toolbox
from deap.tools import Logbook, HallOfFame


def SimpleGPWithElitism(
    population: List[Any],
    toolbox: Toolbox,
    cxpb: float,
    mutpb: float,
    ngen: int,
    stats: Optional[tools.Statistics] = None,
    halloffame: Optional[HallOfFame] = None,
    verbose: bool = False,
) -> Tuple[List[Any], Logbook]:
    """
    Executes a simple Genetic Programming algorithm with elitism.

    Elitism ensures the best individuals from each generation are carried over
    to the next, guaranteeing non-decreasing fitness of the best solution.

    Args:
        population (List[Any]): Initial population of individuals.
        toolbox (Toolbox): DEAP toolbox with genetic operators.
        cxpb (float): Crossover probability.
        mutpb (float): Mutation probability.
        ngen (int): Number of generations.
        stats (Optional[tools.Statistics], optional): Statistics object. Defaults to None.
        halloffame (Optional[HallOfFame], optional): Hall of Fame for elitism. Defaults to None.
        verbose (bool, optional): If True, log statistics to console. Defaults to False.

    Returns:
        Tuple[List[Any], Logbook]: The final population and the evolution logbook.

    Raises:
        ValueError: If halloffame is None (required for elitism).
    """
    logger = logging.getLogger(__name__)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)

    if verbose:
        logger.info(logbook.stream)

    for gen in tqdm(range(1, ngen + 1), desc="Training GP"):
        offspring = toolbox.select(population, len(population) - hof_size)
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        offspring.extend(halloffame.items)
        halloffame.update(offspring)
        population[:] = offspring

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if verbose:
            logger.info(logbook.stream)

    return population, logbook


def train(
    generations: int = 100,
    population: int = 1023,
    elitism: float = 0.1,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    run: int = 0,
    toolbox: Optional[Toolbox] = None,
) -> Tuple[List[Any], Logbook, HallOfFame]:
    """
    Trains a Multi-tree GP model with elitism.

    Args:
        generations (int, optional): Generations count. Defaults to 100.
        population (int, optional): Population size. Defaults to 1023.
        elitism (float, optional): Ratio of elites to keep. Defaults to 0.1.
        crossover_rate (float, optional): Probability of crossover. Defaults to 0.8.
        mutation_rate (float, optional): Probability of mutation. Defaults to 0.2.
        run (int, optional): Seed for reproducibility. Defaults to 0.
        toolbox (Optional[Toolbox], optional): DEAP toolbox. Defaults to None.

    Returns:
        Tuple[List[Any], Logbook, HallOfFame]: Final population, stats logbook, and Hall of Fame.
    """
    assert (
        abs(crossover_rate + mutation_rate - 1.0) < 1e-9
    ), "Crossover and mutation rates must sum to 1.0"

    random.seed(run)
    pop = toolbox.population(n=population)

    mu = round(elitism * population)
    hall_of_fame = tools.HallOfFame(mu) if elitism > 0 else None

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    length = lambda a: np.max(list(map(len, a)))
    stats_size = tools.Statistics(length)

    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    population, logbook = SimpleGPWithElitism(
        pop,
        toolbox,
        crossover_rate,
        mutation_rate,
        generations,
        stats=mstats,
        halloffame=hall_of_fame,
        verbose=True,
    )
    return population, logbook, hall_of_fame


def save_model(
    file_path: str,
    generations: int,
    population: List[Any],
    hall_of_fame: HallOfFame,
    toolbox: Toolbox,
    logbook: Logbook,
    run: int,
) -> None:
    """
    Saves the evolved GP model and state to a pickle file.

    Args:
        file_path (str): Output path.
        generations (int): Generations completed.
        population (List[Any]): Final population.
        hall_of_fame (HallOfFame): Hall of Fame.
        toolbox (Toolbox): DEAP toolbox.
        logbook (Logbook): Statistics log.
        run (int): Run identifier.
    """
    cp = dict(
        population=population,
        generation=generations,
        halloffame=hall_of_fame,
        logbook=logbook,
        rndstate=random.getstate(),
        run=run,
    )
    with open(file_path, "wb") as cp_file:
        pickle.dump(cp, cp_file)


def load_model(
    file_path: str,
    generations: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    toolbox: Optional[Toolbox] = None,
) -> Tuple[List[Any], Logbook, HallOfFame]:
    """
    Loads a GP model state and continues evolution.

    Args:
        file_path (str): Input path.
        generations (int, optional): Additional generations to run. Defaults to 100.
        crossover_rate (float, optional): Crossover rate. Defaults to 0.8.
        mutation_rate (float, optional): Mutation rate. Defaults to 0.2.
        toolbox (Optional[Toolbox], optional): DEAP toolbox. Defaults to None.

    Returns:
        Tuple[List[Any], Logbook, HallOfFame]: Population, Logbook, and Hall of Fame.
    """
    with open(file_path, "rb") as cp_file:
        cp = pickle.load(cp_file)

    population = cp["population"]
    halloffame = cp["halloffame"]
    random.setstate(cp["rndstate"])
    run = cp["run"]

    assert (
        abs(crossover_rate + mutation_rate - 1.0) < 1e-9
    ), "Crossover and mutation rates must sum to 1.0"

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    length = lambda a: np.max(list(map(len, a)))
    stats_size = tools.Statistics(length)

    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    random.seed(run)
    pop, log = SimpleGPWithElitism(
        population,
        toolbox,
        crossover_rate,
        mutation_rate,
        generations,
        stats=mstats,
        halloffame=halloffame,
        verbose=True,
    )

    return pop, log, halloffame
