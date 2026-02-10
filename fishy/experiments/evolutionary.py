# -*- coding: utf-8 -*-
"""
Evolutionary experiments module (Genetic Programming).

This module provides an orchestrator for running Genetic Programming (GP) experiments
on spectral data. It uses the DEAP library to evolve multi-tree individuals for
multi-class classification, including support for elitism and checkpointing.
"""

import logging
import operator
import os
import numpy as np
from deap import base, creator, tools, gp as deap_gp
from deap.gp import PrimitiveSetTyped
from sklearn.model_selection import StratifiedKFold

from fishy.models.evolutionary.gp_util import compileMultiTree, evaluate_classification
from fishy.models.evolutionary.operators import xmate, xmut, staticLimit
from fishy.models.evolutionary.gp import train, save_model, load_model
from fishy.models.evolutionary.gp_data import load_dataset
from fishy.models.evolutionary.gp_plot import plot_tsne, plot_gp_tree

def run_gp_experiment(
    dataset: str = "species",
    generations: int = 10,
    population: int = 1023,
    run: int = 0,
    file_path: str = "outputs/checkpoints/embedded-gp.pth",
    output_log: str = "outputs/logs/evolutionary/results",
    load_checkpoint: bool = False,
    data_file_path: str = None
):
    """
    Runs a Genetic Programming experiment.

    This function sets up the GP environment (primitives, fitness, toolbox), loads the 
    specified dataset, and performs stratified k-fold cross-validation. In each fold,
    it evolves a population of multi-tree individuals to solve the classification task.

    Args:
        dataset (str): Name of the dataset to use ('species', 'part', etc.).
        generations (int): Number of generations to evolve the population.
        population (int): Size of the population.
        run (int): Run identifier for random seeding and logging.
        file_path (str): File path to save/load model checkpoints.
        output_log (str): Base path for output log files.
        load_checkpoint (bool): If True, attempts to resume from a checkpoint at ``file_path``.
        data_file_path (str): Optional path to the dataset excel file.
    """
    os.makedirs(os.path.dirname(output_log), exist_ok=True)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f"{output_log}_{run}.log", level=logging.INFO, filemode="w")

    np.random.seed(run)

    n_features = 1023
    if dataset == "instance-recognition": n_features = 2046
    
    n_classes_per_dataset = {
        "species": 2, "part": 6, "oil": 7, "cross-species": 3,
        "cross-species-hard": 15, "instance-recognition": 2,
    }
    n_classes = n_classes_per_dataset[dataset]

    X, y = load_dataset(dataset=dataset, file_path=data_file_path)
    pset = PrimitiveSetTyped("main", [float] * n_features, float)

    def protectedDiv(left, right): return np.divide(left, right, out=np.ones_like(left, dtype=float), where=right != 0)
    def add(x, y): return x.astype(float) + y.astype(float)
    def sub(x, y): return x.astype(float) - y.astype(float)
    def mul(x, y): return x.astype(float) * y.astype(float)
    def neg(x): return -x.astype(float)

    pset.addPrimitive(protectedDiv, [float, float], float, name="/")
    pset.addPrimitive(add, [float, float], float, name="+")
    pset.addPrimitive(mul, [float, float], float, name="x")
    pset.addPrimitive(sub, [float, float], float, name="-")
    pset.addPrimitive(neg, [float], float, name="-1*")

    # Avoid duplicate creation if called multiple times in same process
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    k = 3 if dataset in ["part", "cross-species-hard"] else 5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        m = n_classes # Construction ratio r=1
        toolbox.register("expr", deap_gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.expr, n=m)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", compileMultiTree, X=X_train)
        toolbox.register("evaluate", evaluate_classification, toolbox=toolbox, pset=pset, X=X_train, y=y_train)
        toolbox.register("select", tools.selTournament, tournsize=7)
        toolbox.register("mate", xmate)
        toolbox.register("expr_mut", deap_gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", xmut, expr=toolbox.expr_mut, pset=pset)

        toolbox.decorate("mate", staticLimit(key=operator.attrgetter("height"), max_value=6))
        toolbox.decorate("mutate", staticLimit(key=operator.attrgetter("height"), max_value=6))

        if load_checkpoint and os.path.isfile(file_path):
            pop, log, hof = load_model(file_path=file_path, toolbox=toolbox, generations=10)
        else:
            pop, log, hof = train(generations=generations, population=population, run=run, toolbox=toolbox)

        save_model(file_path=file_path, population=pop, generations=generations, hall_of_fame=hof, toolbox=toolbox, logbook=log, run=run)
        
        best = hof[0]
        evaluate_classification(best, toolbox=toolbox, pset=pset, verbose=True, X=X_test, y=y_test)