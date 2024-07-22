import logging
import torch
import numpy as np
from deap import base, creator, tools, algorithms
from tqdm import tqdm

# Define the genetic algorithm components
def create_individual(n_features, n_classes):
    return np.random.rand(n_features, n_classes)

def evaluate(individual, data_loader):
    individual_tensor = torch.FloatTensor(individual)
    correct = 0
    total = 0
    for X_batch, y_batch in data_loader:
        predictions = torch.argmax(X_batch @ individual_tensor, dim=1)
        correct += (predictions == y_batch.argmax(1)).sum().item()
        total += len(y_batch)
    return correct / total,

def mutate(individual, mu, sigma, indpb):
    for i in range(individual.shape[0]):
        for j in range(individual.shape[1]):
            if np.random.random() < indpb:
                individual[i, j] += np.random.normal(mu, sigma)
    return individual,

def train(train_loader, val_loader, n_classes, population, crossover_rate, mutation_rate, generations):
    logger = logging.getLogger(__name__)
    # Set up the DEAP toolbox
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, 
                    lambda: create_individual(1023, n_classes))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, data_loader=train_loader)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate, mu=0, sigma=0.2, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def array_equal(arr1, arr2):
        return np.array_equal(arr1, arr2)

    # Create the Hall of Fame with the custom similarity function
    hof = tools.HallOfFame(5, similar=array_equal)

    # Statistics to track
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the genetic algorithm with elitism
    population = toolbox.population(n=population)

    for gen in (pbar := tqdm(range(generations), desc="Training: ")):
        # Select and clone the next generation individuals
        offspring = algorithms.varAnd(population, toolbox, cxpb=crossover_rate, mutpb=mutation_rate)
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update the hall of fame with the generated individuals
        hof.update(offspring)
        
        # Select the next generation population
        population[:] = toolbox.select(offspring, k=len(population) - len(hof))
        population.extend(hof)  # Add the best back to the population
        
        # Compile statistics about the new population
        record = stats.compile(population)
        message = f"Generation {gen+1}: Best={record['max']:.4f}, Avg={record['avg']:.4f}"
        logger.info(message)
        pbar.set_description(message)

    # Get the best individual from the Hall of Fame
    best_individual = hof[0]

    # Evaluate the best individual on the test set
    training_accuracy = evaluate(best_individual, train_loader)[0]
    test_accuracy = evaluate(best_individual, val_loader)[0]

    message = f"Best individual's training accuracy: {training_accuracy:.4f} test accuracy: {test_accuracy:.4f}"
    logger.info(message)    
    print(f"{message}")

    # # Print the top 5 individuals from the Hall of Fame
    # logger.info("\nTop 5 individuals:")
    # for i, ind in enumerate(hof):
    #     logger.info(f"{i+1}. Fitness: {ind.fitness.values[0]:.4f}")