import logging
import argparse
from gp import GeneticProgram

if __name__ == "__main__":
    # Handle the command line arguments for the script.
    parser = argparse.ArgumentParser(
                    prog='Embedded Genetic Programming',
                    description='An embedded GP for fish species classification.',
                    epilog='Implemented in evotorch and written in python.')
    parser.add_argument('-f', '--file-path', type=str, default="checkpoints/embedded-gp.pth")
    parser.add_argument('-d', '--dataset', type=str, default="species")
    parser.add_argument('-l', '--load', type=bool, default=False)
    parser.add_argument('-r', '--run', type=int, default=0)
    parser.add_argument('-o', '--output', type=str, default=f"logs/results")
    parser.add_argument('-p', '--population', type=int, default=1023)
    parser.add_argument('-b', '--beta', type=int, default=1)
    parser.add_argument('-g', '--generations', type=int, default=10)
    parser.add_argument('-mx', '--mutation-rate', type=float, default=0.2)
    parser.add_argument('-cx', '--crossover-rate', type=int, default=0.8)
    parser.add_argument('-e', '--elitism', type=int, default=0.1)
    args = vars(parser.parse_args())
    logger = logging.getLogger(__name__)
    output = f"{args['output']}_{args['run']}.log"
    logging.basicConfig(filename=output, level=logging.DEBUG, filemode='w')
    
    n_features = 1023
    beta = args['beta']
    population = n_features * beta
    # population = args['population']
    generations = args['generations']
    crossover_rate = args['crossover_rate']
    mutation_rate = args['mutation_rate']
    
    gp = GeneticProgram(
                        population=population, 
                        generations=generations, 
                        crossover_rate=crossover_rate,
                        mutation_rate=mutation_rate)
    gp()