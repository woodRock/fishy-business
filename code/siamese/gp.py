"""
Multi-tree genetic programming for contrastive learning.

References: 
1.  Bromley, J., Guyon, I., LeCun, Y., Säckinger, E., & Shah, R. (1993). 
    Signature verification using a" siamese" time delay neural network. 
    Advances in neural information processing systems, 6.
2.  Koza, J. R. (1994). 
    Genetic programming II: automatic discovery of reusable programs.
"""

import logging
import argparse
import random
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from deap import algorithms, base, creator, gp, tools
from sklearn.metrics import balanced_accuracy_score
from sklearn.decomposition import PCA
from typing import List, Tuple, Callable, Any, Optional


def parse_arguments():
    # Handle the command line arguments for the script.
    parser = argparse.ArgumentParser(
                    prog='Embedded Genetic Programming',
                    description='An embedded GP for fish species classification.',
                    epilog='Implemented in deap and written in python.')
    parser.add_argument('-f', '--file-path', type=str, default="checkpoints/embedded-gp.pth", 
                        help="The filepath to store the checkpoints. Defaults to checkpoints/embedded-gp.pth")
    parser.add_argument('-d', '--dataset', type=str, default="instance-recognition", 
                        help="The fish species or part dataset. Defaults to instance-recognition.")
    parser.add_argument('-l', '--load', type=bool, action=argparse.BooleanOptionalAction, default=False,
                        help="To load a checkpoint from a file. Defaults to false")
    parser.add_argument('-r', '--run', type=int, default=0,
                        help="The number for the run, this effects the random seed. Defaults to 0")
    parser.add_argument('-o', '--output', type=str, default=f"logs/results",
                        help="Partial filepath for the output logging. Defaults to 'logs/results'.")
    parser.add_argument('-p', '--population', type=int, default=100,
                        help="The number of individuals in the population. Defaults to 100.")
    parser.add_argument('-g', '--generations', type=int, default=10,
                        help="The number of generations, or epochs, to train for. Defaults to 10.")
    parser.add_argument('-mx', '--mutation-rate', type=float, default=0.2,
                        help="The probability of a mutation operations occuring. Defaults to 0.2")
    parser.add_argument('-cx', '--crossover-rate', type=int, default=0.8,
                        help="The probability of a mutation operations occuring. Defaults to 0.2")
    parser.add_argument('-e', '--elitism', type=int, default=5,
                        help="The number of elitists to be kept each generation.")
    parser.add_argument('-td', '--tree-depth', type=int, default=6,
                        help="The maximum tree depth for GP trees. Defaults to 6.")
    parser.add_argument('-nt', '--num-trees', type=int, default=20,
                        help="The number of trees for multi-tree GP. Defaults to 20.")

    return parser.parse_args()


def setup_logging(args):
    logger = logging.getLogger(__name__)
    output = f"{args.output}_{args.run}.log"
    logging.basicConfig(filename=output, level=logging.INFO, filemode='w')
    return logger


args = parse_arguments()
logger = setup_logging(args)

# Define primitives that work with numpy arrays and return float arrays
def protectedDiv(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.divide(left, right, out=np.ones_like(left, dtype=float), where=right!=0)

def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x.astype(float) + y.astype(float)

def sub(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x.astype(float) - y.astype(float)

def mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x.astype(float) * y.astype(float)

def neg(x: np.ndarray) -> np.ndarray:
    return -x.astype(float)

def sin(x: np.ndarray) -> np.ndarray:
    return np.sin(x.astype(float))

def cos(x: np.ndarray) -> np.ndarray:
    return np.cos(x.astype(float))

def rand101(x: np.ndarray) -> np.ndarray:
    return np.random.uniform(-1, 1, size=x.shape)

pset = gp.PrimitiveSet("MAIN", 1023)  # 1 input for individual feature evaluation
pset.addPrimitive(add, 2)
pset.addPrimitive(sub, 2)
pset.addPrimitive(mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(neg, 1)
pset.addPrimitive(sin, 1)
pset.addPrimitive(cos, 1)
pset.addPrimitive(rand101, 1)
pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Define the number of trees per individual
NUM_TREES = args.num_trees

# Toolbox initialization
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=args.tree_depth)
toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.tree, n=NUM_TREES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Compile function
def compile_trees(individual: List[gp.PrimitiveTree]) -> List[Callable]:
    return [gp.compile(expr, pset) for expr in individual]

# Contrastive loss function
def contrastive_loss(z1, z2, y, temperature=0.5):
    similarity = nn.functional.cosine_similarity(z1, z2)
    loss = y * torch.pow(1 - similarity, 2) + (1 - y) * torch.pow(torch.clamp(similarity - 0.1, min=0.0), 2)
    return loss.mean()

# Evaluation function
def evalContrastive(individual: List[gp.PrimitiveTree], data: List[Tuple[np.ndarray, np.ndarray, float]], alpha: float = 0.5) -> Tuple[float]:
    trees = compile_trees(individual)
    total_loss = 0
    predictions = []
    labels = []
    
    for x1, x2, label in data:
        # Evaluate both inputs using the same set of trees (Siamese approach)
        outputs1 = torch.tensor(np.array([tree(*x1) for tree in trees]), dtype=torch.float32)
        outputs2 = torch.tensor(np.array([tree(*x2) for tree in trees]), dtype=torch.float32)
        
        loss = contrastive_loss(outputs1.unsqueeze(0), outputs2.unsqueeze(0), label)
        total_loss += loss.item()
        
        euclidean_distance = F.pairwise_distance(outputs1.unsqueeze(0), outputs2.unsqueeze(0))
        pred = 0 if euclidean_distance < 0.5 else 1  # Adjust threshold as needed
        predictions.append(pred)
        labels.append(label)
    
    avg_loss = total_loss / len(data)
    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    loss_balanced = 1 - balanced_accuracy
    fitness = alpha * loss_balanced + (1 - alpha) * avg_loss  # Combine accuracy and loss
    return (fitness,)

# Custom crossover function
def customCrossover(ind1: List[gp.PrimitiveTree], ind2: List[gp.PrimitiveTree]) -> Tuple[List[gp.PrimitiveTree], List[gp.PrimitiveTree]]:
    for i in range(len(ind1)):
        if random.random() < 0.5:
            ind1[i], ind2[i] = gp.cxOnePoint(ind1[i], ind2[i])
    return ind1, ind2

# Custom mutation function
def customMutate(individual: List[gp.PrimitiveTree]) -> Tuple[List[gp.PrimitiveTree]]:
    for i in range(len(individual)):
        if random.random() < 0.2:  # 20% chance to mutate each tree
            individual[i], = gp.mutUniform(individual[i], expr=toolbox.expr, pset=pset)
    return individual,

# Genetic operators
toolbox.register("mate", customCrossover)
toolbox.register("mutate", customMutate)
toolbox.register("select", tools.selTournament, tournsize=7)

def eaSimpleWithElitism(population: List[List[gp.PrimitiveTree]], 
                        toolbox: base.Toolbox, 
                        cxpb: float, 
                        mutpb: float, 
                        ngen: int, 
                        stats: Optional[tools.Statistics] = None,
                        halloffame: Optional[tools.HallOfFame] = None, 
                        verbose: bool = __debug__, 
                        elite_size: int = 1,
                        train_data: List[Tuple[np.ndarray, np.ndarray, float]] = None, 
                        val_data: List[Tuple[np.ndarray, np.ndarray, float]] = None) -> Tuple[List[List[gp.PrimitiveTree]], tools.Logbook]:
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be None")
    halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        logger.info(logbook.stream)

    # Begin the generational process
    for gen in tqdm(range(1, ngen + 1), desc="Training"):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - elite_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Add the best back to population:
        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            logger.info(logbook.stream)
        
        # Print the best (lowest) fitness in this generation
        best_fit = halloffame[0].fitness.values[0]
        train_balanced_accuracy = evaluate_best_individual(halloffame[0], train_data)
        val_balanced_accuracy = evaluate_best_individual(halloffame[0], val_data)
        logger.info(f"""
                    Generation {gen}: Best Fitness = {best_fit:.4f} 
                    Balanced accuracy: Train: {train_balanced_accuracy:.4f} 
                    Validation: {val_balanced_accuracy:.4f}""")

    return population, logbook

def evaluate_best_individual(individual: List[gp.PrimitiveTree], data: List[Tuple[np.ndarray, np.ndarray, float]]) -> float:
    trees = compile_trees(individual)
    predictions = []
    labels = []
    
    for x1, x2, label in data:
        outputs1 = torch.tensor(np.array([tree(*x1) for tree in trees]), dtype=torch.float32)
        outputs2 = torch.tensor(np.array([tree(*x2) for tree in trees]), dtype=torch.float32)
        
        euclidean_distance = F.pairwise_distance(outputs1.unsqueeze(0), outputs2.unsqueeze(0))
        pred = 0 if euclidean_distance < 0.5 else 1  # Adjust threshold as needed
        predictions.append(pred)
        labels.append(label)
    
    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    return balanced_accuracy

def extract_features(individual: List[gp.PrimitiveTree], data: List[Tuple[np.ndarray, np.ndarray, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    trees = compile_trees(individual)
    features_x1 = []
    features_x2 = []
    labels = []
    
    for x1, x2, label in data:
        output_x1 = np.array([tree(*x1) for tree in trees])
        output_x2 = np.array([tree(*x2) for tree in trees])
        features_x1.append(output_x1)
        features_x2.append(output_x2)
        labels.append(label)
    
    return np.array(features_x1), np.array(features_x2), np.array(labels)

def plot_features(features_x1: np.ndarray, features_x2: np.ndarray, labels: np.ndarray):
    # Use PCA to reduce dimensionality to 2D for visualization
    pca = PCA(n_components=2)
    features_x1_2d = pca.fit_transform(features_x1)
    features_x2_2d = pca.transform(features_x2)

    plt.figure(figsize=(12, 6))
    
    # Plot x1 features
    plt.scatter(features_x1_2d[labels == 0, 0], features_x1_2d[labels == 0, 1], c='blue', marker='o', label='x1 (class 0)', alpha=0.5)
    plt.scatter(features_x1_2d[labels == 1, 0], features_x1_2d[labels == 1, 1], c='red', marker='o', label='x1 (class 1)', alpha=0.5)
    
    # Plot x2 features
    plt.scatter(features_x2_2d[labels == 0, 0], features_x2_2d[labels == 0, 1], c='cyan', marker='s', label='x2 (class 0)', alpha=0.5)
    plt.scatter(features_x2_2d[labels == 1, 0], features_x2_2d[labels == 1, 1], c='magenta', marker='s', label='x2 (class 1)', alpha=0.5)
    
    plt.title("Constructed Features Visualization")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/constructed_features.pdf")
    plt.close()


def main() -> Tuple[List[List[gp.PrimitiveTree]], tools.Logbook, tools.HallOfFame]:
    # Load and preprocess your data
    from util import preprocess_dataset
    train_loader, val_loader = preprocess_dataset(dataset=args.dataset, batch_size=64)
    
    # Convert data loaders to list format for GP
    def loader_to_list(loader: torch.utils.data.DataLoader) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        data_list = []
        for x1, x2, y in loader:
            for i in range(len(y)):
                # Ensure x1 and x2 have NUM_TREES features each
                data_list.append((x1[i].numpy(), x2[i].numpy(), y[i].item()))
        return data_list
    
    train_data = loader_to_list(train_loader)
    val_data = loader_to_list(val_loader)
    
    # Register the evaluation function with the training data
    toolbox.register("evaluate", evalContrastive, data=train_data)
    
    # GP parameters
    pop_size = args.population
    generations = args.generations
    elite_size = 5  # Number of elite individuals to preserve
    
    # Initialize population
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(elite_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    with Pool() as pool:
        toolbox.register("map", pool.map)
        # Run GP with elitism
        pop, log = eaSimpleWithElitism(pop, toolbox, cxpb=args.crossover_rate, mutpb=args.mutation_rate, ngen=generations, 
                                    stats=stats, halloffame=hof, verbose=True, elite_size=elite_size,
                                    train_data=train_data, val_data=val_data)
    
    # Evaluate best individual on validation set
    best_individual = hof[0]
    best_fitness = evalContrastive(best_individual, val_data)
    logger.info(f"Best individual fitness on validation set: {best_fitness[0]}")
    
    # Calculate and print the balanced accuracy score for the best individual
    balanced_accuracy = evaluate_best_individual(best_individual, val_data)
    logger.info(f"Balanced Accuracy Score of the best individual on validation set: {balanced_accuracy:.4f}")
    
    logger.info(f"Printing the GP trees")
    for tree_idx, tree in enumerate(best_individual):
        nodes, edges, labels = gp.graph(tree)

        ### Graphviz Section ###
        import pygraphviz as pgv

        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")

        for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]

        g.draw(f"figures/tree_{tree_idx}.pdf")
    
    logger.info("Plotting the constructed features for the best individual.")
    features_x1, features_x2, labels = extract_features(best_individual, train_data)
    plot_features(features_x1, features_x2, labels)
    logger.info("Feature visualization saved as 'figures/constructed_features.pdf'")

    return pop, log, hof


if __name__ == "__main__":
    pop, log, hof = main()