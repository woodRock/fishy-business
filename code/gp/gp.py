import logging
import pickle
import random
import numpy as np
from tqdm import tqdm 
from deap import tools
from deap import algorithms
from deap import tools

def SimpleGPWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """
    Elitism for Multi-Tree GP for Multi-Class classification.
    A variation of the eaSimple method from the DEAP library that supports

    Elitism ensures the best individuals (the elite) from each generation are
    carried onto the next without alteration. This ensures the quality of the
    best solution monotonically increases over time.

    Args:
        population: The number of individuals to evolve.
        toolbox: The toolbox containing the genetic operators.
        cxpb: The probability of a crossover between two individuals.
        mutpb: The probability of a random mutation within an individual.
        ngen: The number of genetations to evolve the population for.
        stats: That can be used to collect statistics on the evolution.
        halloffame: The hall of fame contains the best individual solutions.
        verbose: Whether or not to print the logbook.

    Returns:
        population: The final population the algorithm has evolved.
        logbook: The logbook which can record important statistics.
    """
    logger = logging.getLogger(__name__)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

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


def train(generations=100, population=100, elitism=0.1, crossover_rate=0.5, mutation_rate=0.1, run=0, toolbox=None):
    """
    This is a Multi-tree GP with Elitism for Multi-class classification.

    Args:
        generations: The number of generations to evolve the populaiton for.
        elitism: The ratio of elites to be kept between generations.
        crossover_rate: The probability of a crossover between two individuals.
        mutation_rate: The probability of a random mutation within an individual.

    Returns:
        pop: The final population the algorithm has evolved.
        log: The logbook which can record important statistics.
        hof: The hall of fame contains the best individual solutions.

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
    # Reproducuble results for each run.
    random.seed(run)

    pop = toolbox.population(n=population)

    # Elitism (Koza 1994)
    mu = round(elitism * population)
    if elitism > 0:
        # See https://www.programcreek.com/python/example/107757/deap.tools.HallOfFame
        hof = tools.HallOfFame(mu)
    else:
        hof = None

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    length = lambda a: np.max(list(map(len, a)))
    stats_size = tools.Statistics(length)

    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # Run the genetic program.
    pop, log = SimpleGPWithElitism(pop, toolbox, crossover_rate, mutation_rate,
                                   generations, stats=mstats, halloffame=hof,
                                   verbose=True)
    return pop, log, hof


def save_model(file_path="checkpoint_name.pkl", population=None, generations=None, hall_of_fame=None, toolbox=None, logbook=None, run=None):
    """
    Save the model to a file.

    Args:
        file_path: The path to save the model to. Default is "checkpoint_name.pkl".
    """
    cp = dict(population=population, generation=generations, halloffame=hall_of_fame, logbook=logbook, rndstate=random.getstate(), run=run)
    with open(file_path, "wb") as cp_file:
        pickle.dump(cp, cp_file)


def load_model(file_path="checkpoint_name.pkl", toolbox=None, generations=100, crossover_rate=0.8, mutation_rate=0.2):
    """
    Load a model from a file.

    Args:
        file_path: The path to load the model from. Default is "checkpoint_name.pkl".
        generations: The number of generations to train for. Default is 100.
    """
    with open(file_path, "rb") as cp_file:
        cp = pickle.load(cp_file)

    population = cp["population"]
    start_gen = cp["generation"]
    halloffame = cp["halloffame"]
    logbook = cp["logbook"]
    random.setstate(cp["rndstate"])
    run = cp["run"]

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
    pop, log = SimpleGPWithElitism(population, toolbox, crossover_rate, mutation_rate, generations, stats=mstats, halloffame=halloffame, verbose=True)

    return pop, log, halloffame