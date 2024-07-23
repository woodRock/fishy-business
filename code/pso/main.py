import argparse
import logging
import time
from util import preprocess_dataset
from pso import PSO


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
    parser.add_argument('-bs', '--batch-size', type=int, default=64,
                        help="The batch size for the data loaders. Defaults to 64.")
    parser.add_argument('-p', '--population', type=int, default=100,
                        help="The number of individuals in the population. Defaults to 100.")
    parser.add_argument('-b', '--beta', type=int, default=-1,
                        help="Specify beta * num_features as population size. Defaults to -1.")
    parser.add_argument('-g', '--generations', type=int, default=100,
                        help="The number of generations, or epochs, to train for. Defaults to 100.")
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
    n_classes_per_dataset = {"species": 2, "part": 6, "oil": 7, "cross-species": 3}

    if args.dataset not in n_classes_per_dataset:
        raise ValueError(f"Invalid dataset: {args.dataset} not in {n_classes_per_dataset.keys()}")
    
    n_classes = n_classes_per_dataset[args.dataset]

    train_loader, val_loader = preprocess_dataset(
        dataset=args.dataset,
        batch_size=args.batch_size,
        is_data_augmentation=True,
        is_pre_train=False
    )

    population = args.beta * n_features if args.beta != -1 else args.population
    
    # Initialize and train PSO classifier
    model = PSO(
        n_particles=population, 
        n_iterations=args.generations, 
        c1=0.4, c2=0.4, 
        w_start=0.9, w_end=0.4,
        n_classes=n_classes,
        n_features=n_features
    )
    
    start_time = time.time()
    model.fit(train_loader, val_loader)
    end_time = time.time()
    logger.info(f"Training time: {end_time - start_time:.4f} seconds")
    
    # Evaluate the model on the training and validation DataLoaders.
    train_accuracy = model.evaluate(train_loader)
    val_accuracy = model.evaluate(val_loader)

    # Display the final train and test accuracy.
    score_str = f"Training Accuracy: {train_accuracy:.4f} Validation Accuracy: {val_accuracy:.4f}"
    logger.info(score_str)
    print(f"{score_str}")

if __name__ == "__main__":
    main()