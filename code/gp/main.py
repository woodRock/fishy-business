import logging
import argparse
import math
import copy
import random
import operator
import os
from re import I
from operator import attrgetter
from functools import wraps, partial
import numpy as np
from deap import algorithms
from deap.algorithms import varAnd
from deap import base, creator, tools, gp
from deap.gp import PrimitiveTree, Primitive, Terminal
from sklearn.metrics import balanced_accuracy_score
from util import quick_evaluate, compileMultiTree, wrapper_classification_accuracy, normalize, evaluate_classification
from operators import xmate, xmut, staticLimit
from gp import SimpleGPWithElitism, train, save_model, load_model
from data import load_dataset
# Disable the warnings.
# Source: https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Handle the command line arguments for the script.
    parser = argparse.ArgumentParser(
                    prog='Embedded Genetic Programming',
                    description='An embedded GP for fish species classification.',
                    epilog='Implemented in deap and written in python.')
    parser.add_argument('-f', '--file-path', type=str, default="checkpoints/embedded-gp.pth")
    parser.add_argument('-d', '--dataset', type=str, default="species")
    parser.add_argument('-l', '--load', type=bool, default=False)
    parser.add_argument('-r', '--run', type=int, default=0)
    parser.add_argument('-o', '--output', type=str, default=f"logs/results")
    parser.add_argument('-p', '--population', type=int, default=100)
    parser.add_argument('-b', '--beta', type=int, default=1)
    parser.add_argument('-g', '--generations', type=int, default=10)
    parser.add_argument('-mx', '--mutation-rate', type=float, default=0.2)
    parser.add_argument('-cx', '--crossover-rate', type=int, default=0.8)
    parser.add_argument('-e', '--elitism', type=int, default=0.1)

    args = vars(parser.parse_args())

    # Freeze the seed for reproduceability.
    run = args['run'] # @param {type: "integer"}
    dataset = args['dataset']
    file_path = args['file_path']

    # Logging output to a file.
    logger = logging.getLogger(__name__)
    # Run argument for numbered log files.
    output = f"{args['output']}_{args['run']}.log"
    # Filemode is write, so it clears the file, then appends output.
    logging.basicConfig(filename=output, level=logging.INFO, filemode='w')

    # The number of features in the dataset.
    n_features = 1023
    
    # Hyperparameters
    beta = args['beta'] # @param {type: "integer"}
    population = n_features * beta
    # population = args['population']
    generations = args['generations'] # @param {type: "integer"}
    elitism = args['elitism'] # @param {type: "number"}
    crossover_rate = args['crossover_rate'] # @param {type: "number"}
    mutation_rate = args['mutation_rate'] # @param {type: "number"}

    assert crossover_rate + mutation_rate == 1, "Crossover and mutation sums to 1 (to please the Gods!)"

    X,y = load_dataset(dataset=dataset)
    
    n_features = 1023
    n_classes = 2
    if dataset == "species":
        n_classes = 2
    elif dataset == "part":
        n_classes = 6

    pset = gp.PrimitiveSet("MAIN", n_features)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(operator.neg, 1)
    # pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
        
    global toolbox
    toolbox = base.Toolbox()

    minimized = False
    if minimized:
        weight = -1.0
    else:
        weight = 1.0

    weights = (weight,)

    if minimized:
        creator.create("FitnessMin", base.Fitness, weights=weights)
        creator.create("Individual", list, fitness=creator.FitnessMin)
    else:
        creator.create("FitnessMax", base.Fitness, weights=weights)
        creator.create("Individual", list, fitness=creator.FitnessMax)

    # MCIFC constructs 8 feautres for a (c=4) multi-class classification problem (Tran 2019).
    # c - number of classes, r - construction ratio, m - total number of constructed features.
    # m = r * c = 2 ratio * 4 classes = 8 features

    r = 1
    c = n_classes
    m = r * c

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.expr, n=m)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", compileMultiTree, X=X)

    toolbox.register('evaluate', evaluate_classification, toolbox=toolbox, pset=pset, X=X, y=y)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", xmate)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", xmut, expr=toolbox.expr_mut, pset=pset)

    # See https://groups.google.com/g/deap-users/c/pWzR_q7mKJ0
    toolbox.decorate("mate", staticLimit(key=operator.attrgetter("height"), max_value=8))
    toolbox.decorate("mutate", staticLimit(key=operator.attrgetter("height"), max_value=8))

    # File path for saved model.
    pop, log, hof = None, None, None

    # If a saved model exists?
    if args['load'] and os.path.isfile(file_path):
        s = f"Loading model from file: {file_path}"
        logger.info(s)
        print(s)
        pop, log, hof = load_model(file_path=file_path, toolbox=toolbox, generations=10)
    else:
        s = f"No model found. Train from scratch."
        logger.info(s)
        print(s)
        pop, log, hof = train(generations=generations, population=population, elitism=elitism, 
                                crossover_rate=crossover_rate, mutation_rate=mutation_rate, run=run, toolbox=toolbox)

    logger.info(f"Saving model to file: {file_path}")
    save_model(file_path=file_path, population=pop, generations=generations, hall_of_fame=hof, toolbox=toolbox, logbook=log, run=run) # Best accuracy: 0.911423
     
    best = hof[0]
    evaluate_classification(best, toolbox=toolbox, pset=pset, verbose=True, X=X, y=y)
