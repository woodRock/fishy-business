import numpy as np
import operator
from deap import algorithms, base, creator, gp, tools
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import random
from multiprocessing import Pool


# Define primitives that work with numpy arrays and return float arrays
def protectedDiv(left, right):
    return np.divide(left, right, out=np.ones_like(left, dtype=float), where=right!=0)

def add(x, y):
    return x.astype(float) + y.astype(float)

def sub(x, y):
    return x.astype(float) - y.astype(float)

def mul(x, y):
    return x.astype(float) * y.astype(float)

def neg(x):
    return -x.astype(float)

def sin(x):
    return np.sin(x.astype(float))

def cos(x):
    return np.cos(x.astype(float))

def rand101(x):
    return np.random.uniform(-1, 1, size=x.shape)

pset = gp.PrimitiveSet("MAIN", 2)  # 2 inputs for pairwise comparison
pset.addPrimitive(add, 2)
pset.addPrimitive(sub, 2)
pset.addPrimitive(mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(neg, 1)
pset.addPrimitive(sin, 1)
pset.addPrimitive(cos, 1)
pset.addPrimitive(rand101, 1)
pset.renameArguments(ARG0='x1')
pset.renameArguments(ARG1='x2')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Define the number of trees per individual
NUM_TREES = 5

# Toolbox initialization
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.tree, n=NUM_TREES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Compile function
def compile_trees(individual):
    return [gp.compile(expr, pset) for expr in individual]

# Contrastive loss function
def contrastive_loss(output1, output2, label, margin=1.0):
    euclidean_distance = F.pairwise_distance(output1.unsqueeze(0), output2.unsqueeze(0))
    similar_loss = label * torch.pow(euclidean_distance, 2)
    dissimilar_loss = (1 - label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2)
    loss = torch.mean(similar_loss + dissimilar_loss)
    return min(1, max(loss.item(), 0.0))  # Ensure non-negative output

# Evaluation function
def evalContrastive(individual, data, alpha=0.5):
    trees = compile_trees(individual)
    total_loss = 0
    predictions = []
    labels = []
    
    for x1, x2, label in data:
        outputs1 = [np.mean(tree(x1.astype(float), x2.astype(float))) for tree in trees]
        outputs2 = [np.mean(tree(x2.astype(float), x1.astype(float))) for tree in trees]
        output1 = torch.tensor(outputs1, dtype=torch.float32)
        output2 = torch.tensor(outputs2, dtype=torch.float32)
        loss = contrastive_loss(output1, output2, label)
        total_loss += loss
        
        euclidean_distance = F.pairwise_distance(output1.unsqueeze(0), output2.unsqueeze(0))
        pred = 0 if euclidean_distance < 0.5 else 1  # Adjust threshold as needed
        predictions.append(pred)
        labels.append(label)
    
    avg_loss = total_loss / len(data)
    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    loss_balanced = 1 - balanced_accuracy
    fitness = alpha * loss_balanced + (1 - alpha) * avg_loss  # Combine accuracy and loss
    return (fitness,)

# Custom crossover function
def customCrossover(ind1, ind2):
    for i in range(len(ind1)):
        if random.random() < 0.5:
            ind1[i], ind2[i] = gp.cxOnePoint(ind1[i], ind2[i])
    return ind1, ind2

# Custom mutation function
def customMutate(individual):
    for i in range(len(individual)):
        if random.random() < 0.2:  # 20% chance to mutate each tree
            individual[i], = gp.mutUniform(individual[i], expr=toolbox.expr, pset=pset)
    return individual,

# Genetic operators
toolbox.register("mate", customCrossover)
toolbox.register("mutate", customMutate)
toolbox.register("select", tools.selTournament, tournsize=3)


def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
                        halloffame=None, verbose=__debug__, elite_size=1):
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
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
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
            print(logbook.stream)
        
        # Print the best (lowest) fitness in this generation
        best_fit = halloffame[0].fitness.values[0]
        print(f"Generation {gen}: Best Fitness = {best_fit}")

    return population, logbook


def evaluate_best_individual(individual, data):
    trees = compile_trees(individual)
    predictions = []
    labels = []
    
    for x1, x2, label in data:
        outputs1 = [np.mean(tree(x1.astype(float), x2.astype(float))) for tree in trees]
        outputs2 = [np.mean(tree(x2.astype(float), x1.astype(float))) for tree in trees]
        output1 = torch.tensor(outputs1, dtype=torch.float32)
        output2 = torch.tensor(outputs2, dtype=torch.float32)
        
        euclidean_distance = F.pairwise_distance(output1.unsqueeze(0), output2.unsqueeze(0))
        pred = 0 if euclidean_distance < 0.5 else 1  # Adjust threshold as needed
        predictions.append(pred)
        labels.append(label)
    
    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    return balanced_accuracy


def main():
    # Load and preprocess your data
    from util import preprocess_dataset
    train_loader, val_loader = preprocess_dataset(dataset="instance-recognition", batch_size=64)
    
    # Convert data loaders to list format for GP
    def loader_to_list(loader):
        data_list = []
        for x1, x2, y in loader:
            for i in range(len(y)):
                data_list.append((x1[i].numpy(), x2[i].numpy(), y[i].item()))
        return data_list
    
    train_data = loader_to_list(train_loader)
    val_data = loader_to_list(val_loader)
    
    # Register the evaluation function with the training data
    toolbox.register("evaluate", evalContrastive, data=train_data)
    
    # GP parameters
    pop_size = 100
    generations = 50
    elite_size = 5  # Number of elite individuals to preserve
    
    # Initialize population
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(elite_size)  # This will now correctly store the individuals with lowest fitness
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pool = Pool()
    toolbox.register("map", pool.map)
    
    # Run GP with elitism
    pop, log = eaSimpleWithElitism(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, 
                                   stats=stats, halloffame=hof, verbose=True, elite_size=elite_size)
    
    # Evaluate best individual on validation set
    best_individual = hof[0]
    best_fitness = evalContrastive(best_individual, val_data)
    print(f"Best individual fitness on validation set: {best_fitness[0]}")
    
    # Calculate and print the balanced accuracy score for the best individual
    balanced_accuracy = evaluate_best_individual(best_individual, val_data)
    print(f"Balanced Accuracy Score of the best individual on validation set: {balanced_accuracy:.4f}")
    
    pool.close()
    pool.join()
    
    return pop, log, hof

if __name__ == "__main__":
    pop, log, hof = main()