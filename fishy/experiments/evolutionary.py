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
from typing import Optional # Added import
from deap import base, creator, tools, gp as deap_gp
from deap.gp import PrimitiveSetTyped
from sklearn.model_selection import StratifiedKFold

from fishy.models.evolutionary.gp_util import compileMultiTree, evaluate_classification
from fishy.models.evolutionary.operators import xmate, xmut, staticLimit
from fishy.models.evolutionary.gp import train, save_model, load_model
from fishy.models.evolutionary.gp_data import load_dataset
from fishy.models.evolutionary.gp_plot import plot_tsne, plot_gp_tree
from fishy._core.utils import RunContext
import wandb # Added import
from dataclasses import asdict # Added import

def run_gp_experiment(
    dataset: str = "species",
    generations: int = 10,
    population: int = 1023,
    run: int = 0,
    file_path: str = None,
    output_log: str = "evolutionary",
    load_checkpoint: bool = False,
    data_file_path: str = None,
    # New W&B parameters
    wandb_project: Optional[str] = "fishy-business", # Default to user's project
    wandb_entity: Optional[str] = "victoria-university-of-wellington", # Default to user's entity
    wandb_log: bool = False,
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
        output_log (str): experiment name for RunContext.
        load_checkpoint (bool): If True, attempts to resume from a checkpoint at ``file_path``.
        data_file_path (str): Optional path to the dataset excel file.
        wandb_project (str, optional): Weights & Biases project name. Defaults to "fishy-business".
        wandb_entity (str, optional): Weights & Biases entity name. Defaults to "victoria-university-of-wellington".
        wandb_log (bool): Whether to log to Weights & Biases. Defaults to False.
    """
    wandb_run = None
    if wandb_log:
        # Create a dict for W&B config from function arguments
        wandb_config_dict = {
            "dataset": dataset,
            "generations": generations,
            "population": population,
            "run": run,
            "output_log": output_log,
            "load_checkpoint": load_checkpoint,
            "file_path": file_path,
        }
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config=wandb_config_dict,
            reinit=True,
            group=f"{dataset}_evolutionary",
            job_type="evolutionary_training"
        )

    ctx = RunContext(dataset=dataset, method="evolutionary", model_name="evolutionary_algorithm", wandb_run=wandb_run)
    logger = ctx.logger
    
    try: # Start try block for wandb.finish
        # If file_path is not provided, use a default in the checkpoint dir
        if file_path is None:
            file_path = str(ctx.get_checkpoint_path("gp_model.pth"))

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

        fold_metrics = []

        for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            logger.info(f"Starting Fold {fold_idx + 1}/{k}")
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

            fold_checkpoint = ctx.get_checkpoint_path(f"gp_model_fold_{fold_idx+1}.pth")

            if load_checkpoint and os.path.isfile(file_path):
                pop, log, hof = load_model(file_path=file_path, toolbox=toolbox, generations=10)
            else:
                pop, log, hof = train(generations=generations, population=population, run=run, toolbox=toolbox)

            save_model(file_path=str(fold_checkpoint), population=pop, generations=generations, hall_of_fame=hof, toolbox=toolbox, logbook=log, run=run)
            
            best = hof[0]
            acc = evaluate_classification(best, toolbox=toolbox, pset=pset, verbose=True, X=X_test, y=y_test)
            fold_metrics.append({"fold": fold_idx + 1, "accuracy": acc})
            ctx.log_metric(fold_idx + 1, {"test_accuracy": acc})

        ctx.save_results({"fold_results": fold_metrics, "average_accuracy": np.mean([m["accuracy"] for m in fold_metrics])})
        logger.info(f"GP Experiment finished. Average Accuracy: {np.mean([m['accuracy'] for m in fold_metrics]):.4f}")
    finally:
        if wandb_run:
            wandb_run.finish()
