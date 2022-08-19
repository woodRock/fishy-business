""" 
Genetic Program - Main.py 
=======================

An implementation of a Genetic Program (Koza 1994), a simple GP tree with elitism and a classification map (CM) (Smart 2005) for multi-class classification. 

References:
1. Koza, J. R. (1994). Genetic programming as a means for programming 
    computers by natural selection. Statistics and computing, 4(2), 87-112.
2. Smart, W., & Zhang, M. (2005, March). Using genetic programming for 
    multiclass classification by simultaneously solving component binary 
    classification problems. In European Conference on Genetic Programming 
    (pp. 227-239). Springer, Berlin, Heidelberg.
3. Tran, B., Xue, B., & Zhang, M. (2019). Genetic programming for 
    multiple-feature construction on high-dimensional classification. 
    Pattern Recognition, 93, 404-417.
"""
import random
import operator

import numpy as np
from deap import algorithms
from deap import gp, base, creator, tools

from .data import load, prepare, normalize, encode_labels

def protectedDiv(left, right):
    """
    Protected division operator that avoids zero division.

    Arg:
        left: The left hand side expression. 
        right: The right hand side expression. 

    Returns: 
        left/right, when right is non-zero value. Otherwise, returns 1. 
    """
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def if_then_else(condition, left, right):
    """
    The if then else operator allows for conditional branching within the GP tree. 
    Conditional logic allows for more complex program structures to emerge.

    Args: 
        condition: The expression which determines the chosen branch.
        left: The left branch, taken if condition is positive.
        right: The right branch is returned if condition is negative.

    Return: 
        left, if condition is postive. Otherwise, right. 
    """
    if condition > 0: 
        return left 
    else: 
        return right 


def classification_map(y_pred):
    """ 
    Maps a float to a classification label using a classification map (Smart 2005).

    This variation situates class regions sequentially on the floating point number line. 
    The object image will be classified to the class of the region that the program output with
    the object image input falls into. Class region boundaries start at some negative
    number, and end at the same positive number. Boundaries between the starting point and the 
    end point are allocated with an identical interval of 1.0

    Args:
        y_pred (float): The floating point prediction on the floating point number line. 

    Returns:
        y_pred (int): The predicted class. 

    References:
        1. Smart, W., & Zhang, M. (2005, March). Using genetic programming for multiclass 
        classification by simultaneously solving component binary classification problems. 
        In European Conference on Genetic Programming (pp. 227-239). Springer, Berlin, Heidelberg.
    """
    a = [float('-inf'), -1, 0, 1, float('inf')]
    for i, (a,b) in enumerate(zip(a, a[1:])):
        if y_pred > a and y_pred < b:
            return i 


def evaluate_classification(individual):
    """ 
    Evalautes the fitness of an individual by its classification accuracy. 

    Args:  
        individual: A candidate solution to be evaluated. 

    Returns: 
        error: A fraction of incorrectly classified instances. 
    """
    n_instances = X.shape[0]
    func = toolbox.compile(expr=individual)
    acc = sum(classification_map(func(*x)) == y_ for x,y_ in zip(X,y)) / n_instances 
    error = 1 - acc 
    return error,


def SimpleGPWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """
    A variation of the eaSimple method from the DEAP library that allows for elitism. 

    Elitism ensures the best individuals (the elite) from each generation are 
    carried onto the next without alteration. This ensures the quality of the 
    best solution monotonically increases over time. 
    """
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
        print(logbook.stream)

    for gen in range(1, ngen + 1):
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
            print(logbook.stream)

    return population, logbook


def train(generations=100, population=100, elitism=0.1, crossover_rate=0.5, mutation_rate=0.1):
    """
    This is a SimpleGA with Elitism and a Classification Map (CM) for Multi-class classification. 

    Args:
        generations: The number of generations to evolve the populaiton for. 
        elitism: The ratio of elites to be kept between generations. 
        crossover_rate: The probability of a crossover between two individuals. 
        mutation_rate: The probability of a random mutation within an individual.  

    Returns:
        pop: The final population the algorithm has evolved. 
        log: The logbook which can record important statistics. 
        hof: The hall of fame contains the best individual solutions.
    """
    random.seed(420)
    pop = toolbox.population(n=population)
    
    mu = round(elitism * population)
    if elitism > 0:
        hof = tools.HallOfFame(mu)
    else:
        hof = None
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = SimpleGPWithElitism(pop, toolbox, crossover_rate, mutation_rate, generations, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof

# Load the dataset
folder = './data/matlab/' 
dataset = "Fish.mat"
file = load(dataset, folder=folder)
X,y = prepare(file)
X,_ = normalize(X,X)
y, _, le = encode_labels(y)
labels = le.inverse_transform(np.unique(y))
n_features = X.shape[1]
n_instances = X.shape[0]

pset = gp.PrimitiveSet("MAIN", n_features)

# Arithmetic Operators
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)

# More complex operators. 
# pset.addPrimitive(if_then_else, 3)

# Constants 
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))

# Do we intent to minimize or maximize the fitness?
minimized = True  

if minimized: 
    # Minimize when fitness is error.
    weights = weights=(-1.0,)
else: 
    # Maximize when fitness is accuracy. 
    weights = weights=(1.0,)

creator.create("FitnessMin", base.Fitness, weights=weights)
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)

"""
In work by (Tran 2019), they propose a mutli-tree aproach for multiple-feature construction (MFC). 
This idea can be borrowed to handle multi-class classification problems with multi-tree GP. 
We use a one-vs-all approach, to break the problem into a series of binary classification problems.
This should see marked improvement over the classification map (Smart 2005) approach used currently.

TODO [ ] - Multi-tree GP with one-vs-all for multi-label classification

See if we can make the indivudal a multi-tree representation. 
With on expression per class for one-vs-all multi-class classification. 
(See DEAP documentation for hint https://deap.readthedocs.io/en/master/tutorials/basic/part1.html#funky)

References: 
    1. Tran, B., Xue, B., & Zhang, M. (2019). Genetic programming for 
    multiple-feature construction on high-dimensional classification. 
    Pattern Recognition, 93, 404-417.
"""

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", evaluate_classification)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# Hyperparameters 
generations = 300
population = 100
elitism = 0.1
crossover_rate = 0.5
mutation_rate = 0.1

pop, log, hof = train(generations, population, elitism, crossover_rate, mutation_rate)