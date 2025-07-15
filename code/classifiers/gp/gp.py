import logging
import pickle
import random
import numpy as np
from tqdm import tqdm
from deap import tools
from deap import algorithms
from deap import tools
from deap.base import Toolbox
from deap.tools import Logbook, HallOfFame
from typing import Iterable, Union


def SimpleGPWithElitism(
    population: int,
    toolbox: Toolbox,
    cxpb: float,
    mutpb: float,
    ngen: int,
    stats=None,
    halloffame=None,
    verbose: bool = False,
) -> Union[Iterable, Logbook]:
    """
    Elitism for Multi-Tree GP for Multi-Class classification.
    A variation of the eaSimple method from the DEAP library that supports

    Elitism ensures the best individuals (the elite) from each generation are
    carried onto the next without alteration. This ensures the quality of the
    best solution monotonically increases over time.

    Args:
        population (int): The number of individuals to evolve.
        toolbox (deap.base.Toolbox): The toolbox containing the genetic operators.
        cxpb (float): The probability of a crossover between two individuals.
        mutpb (float): The probability of a random mutation within an individual.
        ngen (int): The number of genetations to evolve the population for.
        stats: That can be used to collect statistics on the evolution.
        halloffame: The hall of fame contains the best individual solutions.
        verbose (bool): Whether or not to print the logbook.

    Returns:
        population (deap.base.Toolbox.population): The final population the algorithm has evolved.
        logbook (deap.tools.Logbook): The logbook which can record important statistics.
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
    toolbox: Toolbox = None,
) -> Union[Iterable, Logbook, HallOfFame]:
    """
    This is a Multi-tree GP with Elitism for Multi-class classification.

    An assertion error will be raised if the crossover_rate and mutation_rate do not sum to 1.

    Args:
        generations (int): The number of generations to evolve the populaiton for. Defaults to 100.
        population (int): The number of individuals for the population. Defaults to 1023.
        elitism (float): The ratio of elites to be kept between generations. Defaults to 0.1
        crossover_rate (float): The probability of a crossover between two individuals. Defaults to 0.8.
        mutation_rate (float): The probability of a random mutation within an individual. Defualts to 0.2
        run (int): the number for the experimental run. Defaults to 0.
        toolbox (deap.base.Toolbox): the toolbox that stores all functions required for GP. Defaults to none.

    Returns:
        population (deap.base.Toolbox.population): The final population the algorithm has evolved.
        logbook (deap.tools.Logbook): The logbook which can record important statistics.
        hall_of_fame (deap.tools.tools.HallOfFame): The hall of fame contains the best individual solutions.

    References:
        1. Koza, J. R. (1994). Genetic programming II: automatic discovery of
          reusable programs.
        2. Tran, B., Xue, B., & Zhang, M. (2019).
          Genetic programming for multiple-feature construction on
          high-dimensional classification. Pattern Recognition, 93, 404-417.
        3. Patil, V. P., & Pawar, D. D. (2015). The optimal crossover or mutation
          rates in genetic algorithm: a review. International Journal of Applied
          Engineering and Technology, 5(3), 38-41.
    """
    assert (
        crossover_rate + mutation_rate == 1
    ), "Crossover and mutation sums to 1 (to please the Gods!)"

    # Reproducuble results for each run.
    random.seed(run)

    pop = toolbox.population(n=population)

    # Elitism (Koza 1994)
    mu = round(elitism * population)
    if elitism > 0:
        # See https://www.programcreek.com/python/example/107757/deap.tools.HallOfFame
        hall_of_fame = tools.HallOfFame(mu)
    else:
        hall_of_fame = None

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    length = lambda a: np.max(list(map(len, a)))
    stats_size = tools.Statistics(length)

    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # Run the genetic program.
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
    file_path: str = "checkpoint_name.pkl",
    generations: int = 0,
    population: Iterable = None,
    hall_of_fame: HallOfFame = None,
    toolbox: Toolbox = None,
    logbook: Logbook = None,
    run: int = 0,
) -> None:
    """
    Save the model to a file.

    This is a Multi-tree GP with Elitism for Multi-class classification.

    Args:
        file_path (str): The filepath to store the model checkpoints to. Defaults to "checkpoint_name.pkl".
        generations (int): The number of generations to evolve the populaiton for. Defaults to 100.
        population (int): The number of individuals for the population. Defaults to 1023.
        elitism (float): The ratio of elites to be kept between generations. Defaults to 0.1
        crossover_rate (float): The probability of a crossover between two individuals. Defaults to 0.8.
        mutation_rate (float): The probability of a random mutation within an individual. Defualts to 0.2
        run (int): the number for the experimental run. Defaults to 0.
        toolbox (deap.base.Toolbox): the toolbox that stores all functions required for GP. Defaults to none.
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
    file_path: str = "checkpoint_name.pkl",
    generations: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    toolbox: Toolbox = None,
) -> Union[Iterable, Logbook, HallOfFame]:
    """
    Load a model from a file.

    An assertion error will be raised if the crossover_rate and mutation_rate do not sum to 1.

    Args:
        file_path (str): The filepath to store the model checkpoints to. Defaults to "checkpoint_name.pkl".
        generations (int): The number of generations to evolve the populaiton for. Defaults to 100.
        crossover_rate (float): The probability of a crossover between two individuals. Defaults to 0.8.
        mutation_rate (float): The probability of a random mutation within an individual. Defualts to 0.2
        toolbox (deap.base.Toolbox): the toolbox that stores all functions required for GP. Defaults to none.

    Returns:
        population (deap.base.Toolbox.population): The final population the algorithm has evolved.
        logbook (deap.tools.Logbook): The logbook which can record important statistics.
        hall_of_fame (deap.tools.tools.HallOfFame): The hall of fame contains the best individual solutions.
    """
    with open(file_path, "rb") as cp_file:
        cp = pickle.load(cp_file)

    population = cp["population"]
    # start_gen = cp["generation"]
    halloffame = cp["halloffame"]
    # logbook = cp["logbook"]
    random.setstate(cp["rndstate"])
    run = cp["run"]

    assert (
        crossover_rate + mutation_rate == 1
    ), "Crossover and mutation sums to 1 (to please the Gods!)"

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    length = lambda a: np.max(list(map(len, a)))
    stats_size = tools.Statistics(length)

    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # Reproducible results each run.
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
