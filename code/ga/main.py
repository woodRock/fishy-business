import argparse
import logging
import time
import numpy as np
from util import preprocess_dataset
from ga import GeneticAlgorithm

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='Genetic Algorithm',
        description='A genetic algorithm (GA) for fish species classification.',
        epilog='Implemented in deap and written in Python.')
    
    parser.add_argument('-f', '--file-path', type=str, default="checkpoints/embedded-gp.pth", 
                        help="The filepath to store the checkpoints. Defaults to checkpoints/embedded-gp.pth")
    parser.add_argument('-d', '--dataset', type=str, default="species", 
                        help="The fish species or part dataset. Defaults to species.")
    parser.add_argument('-l', '--load', action='store_true',
                        help="To load a checkpoint from a file. Defaults to false")
    parser.add_argument('-r', '--run', type=int, default=0,
                        help="The number for the run, this affects the random seed. Defaults to 0")
    parser.add_argument('-o', '--output', type=str, default="logs/results",
                        help="Partial filepath for the output logging. Defaults to 'logs/results'.")
    parser.add_argument('-p', '--population', type=int, default=100,
                        help="The number of individuals in the population. Defaults to 1023.")
    parser.add_argument('-b', '--beta', type=int, default=-1,
                        help="Specify beta * num_features as population size. Defaults to -1.")
    parser.add_argument('-g', '--generations', type=int, default=10,
                        help="The number of generations, or epochs, to train for. Defaults to 10.")
    parser.add_argument('-mx', '--mutation-rate', type=float, default=0.2,
                        help="The probability of a mutation operation occurring. Defaults to 0.2")
    parser.add_argument('-cx', '--crossover-rate', type=float, default=0.8,
                        help="The probability of a crossover operation occurring. Defaults to 0.8")
    parser.add_argument('-e', '--elitism', type=float, default=0.1,
                        help="The ratio of elitists to be kept each generation.")
    
    return parser.parse_args()

def setup_logging(args):
    logger = logging.getLogger(__name__)
    output = f"{args.output}_{args.run}.log"
    logging.basicConfig(filename=output, level=logging.INFO, filemode='w')
    return logger

def main():
    args = parse_arguments()
    logger = setup_logging(args)

    n_features = 1023
    if args.dataset == "instance-recognition":
        n_features = 2046
    n_classes_per_dataset = {"species": 2, "part": 6, "oil": 7, "cross-species": 3, "instance-recognition": 2}
    
    if args.dataset not in n_classes_per_dataset:
        raise ValueError(f"Invalid dataset: {args.dataset} not in {n_classes_per_dataset.keys()}")
    
    n_classes = n_classes_per_dataset[args.dataset]

    train_loader, val_loader = preprocess_dataset(
        dataset=args.dataset,
        batch_size=64,
        is_data_augmentation=True,
        is_pre_train=False
    )

    population = args.beta * n_features if args.beta != -1 else args.population

    assert args.crossover_rate + args.mutation_rate == 1, "Crossover and mutation rates should sum to 1"

    ga = GeneticAlgorithm(
        n_features=n_features, 
        n_classes=n_classes, 
        population_size=population, 
        crossover_rate=args.crossover_rate, 
        mutation_rate=args.mutation_rate, 
        generations=args.generations
    )

    start_time = time.time()
    best_individual = ga.train(train_loader, val_loader)
    end_time = time.time()
    logger.info(f"Training time: {end_time - start_time:.4f} seconds")
    
    train_accuracy = ga.evaluate_model(best_individual, train_loader)
    test_accuracy = ga.evaluate_model(best_individual, val_loader)
    
    final_message = f"Training accuracy: {train_accuracy:.4f} Test accuracy: {test_accuracy:.4f}"
    logger.info(final_message)
    print(final_message)

if __name__ == "__main__":
    main()