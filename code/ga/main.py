import argparse
import logging
import numpy as np
from util import preprocess_dataset
from ga import train

if __name__ == "__main__":

    # Handle the command line arguments for the script.
    parser = argparse.ArgumentParser(
                    prog='Genetic Algorithm',
                    description='An genetic algorithm (GP) for fish species classification.',
                    epilog='Implemented in deap and written in python.')
    parser.add_argument('-f', '--file-path', type=str, default="checkpoints/embedded-gp.pth", 
                        help="The filepath to store the checkpoints. Defaults to checkpoints/embedded-gp.pth")
    parser.add_argument('-d', '--dataset', type=str, default="species", 
                        help="The fish species or part dataset. Defaults to species.")
    parser.add_argument('-l', '--load', type=bool, action=argparse.BooleanOptionalAction, default=False,
                        help="To load a checkpoint from a file. Defaults to false")
    parser.add_argument('-r', '--run', type=int, default=0,
                        help="The number for the run, this effects the random seed. Defaults to 0")
    parser.add_argument('-o', '--output', type=str, default=f"logs/results",
                        help="Partial filepath for the output logging. Defaults to 'logs/results'.")
    parser.add_argument('-p', '--population', type=int, default=1023,
                        help="The number of individuals in the population. Defaults to 1023.")
    parser.add_argument('-b', '--beta', type=int, default=-1,
                        help="Specify beta * num_features as population size. Defaults to -1.")
    parser.add_argument('-g', '--generations', type=int, default=10,
                        help="The number of generations, or epochs, to train for. Defaults to 10.")
    parser.add_argument('-mx', '--mutation-rate', type=float, default=0.2,
                        help="The probability of a mutation operations occuring. Defaults to 0.2")
    parser.add_argument('-cx', '--crossover-rate', type=int, default=0.8,
                        help="The probability of a mutation operations occuring. Defaults to 0.2")
    parser.add_argument('-e', '--elitism', type=int, default=0.1,
                        help="The ratio of elitists to be kept each generation.")

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

    train_loader, val_loader, _, _ , _ = preprocess_dataset(
        dataset=dataset,
        batch_size=64,
        is_data_augmentation=True,
        is_pre_train=False
    )

    # The number of features in the dataset.
    n_features = 1023

    # Hyperparameters
    beta = args['beta'] # @param {type: "integer"}
    population = beta * n_features
    if beta == -1:
        population = args['population']
    generations = args['generations'] # @param {type: "integer"}
    elitism = args['elitism'] # @param {type: "number"}
    crossover_rate = args['crossover_rate'] # @param {type: "number"}
    mutation_rate = args['mutation_rate'] # @param {type: "number"}
    tree_depth = 6 # Manually set the maximum tree depth.
    n_classes_per_dataset = {"species": 2, "part": 6, "oil": 7, "cross-species": 3}
    if dataset not in n_classes_per_dataset.keys():
        raise ValueError(f"Invalid dataset: {dataset} not in {n_classes_per_dataset.keys()}")
    n_classes = n_classes_per_dataset[dataset]

    assert crossover_rate + mutation_rate == 1, "Crossover and mutation sums to 1 (to please the Gods!)"

    train(
        train_loader, val_loader, n_classes, population, crossover_rate, mutation_rate, generations
    )
