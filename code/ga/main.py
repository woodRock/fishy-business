import numpy as np
import torch
from deap import base, creator, tools, algorithms
from util import preprocess_dataset

train_loader, val_loader, _, _ , _ = preprocess_dataset(
    dataset="species",
    batch_size=64,
    is_data_augmentation=True,
    is_pre_train=False
)

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

# Set up the DEAP toolbox
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, 
                 lambda: create_individual(1023, 2))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate, data_loader=train_loader)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate, mu=0, sigma=0.2, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Set up the genetic algorithm parameters
population_size = 100
n_generations = 100
crossover_prob = 0.7
mutation_prob = 0.2

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
population = toolbox.population(n=population_size)

for gen in range(n_generations):
    # Select and clone the next generation individuals
    offspring = algorithms.varAnd(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob)
    
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
    print(f"Generation {gen+1}: Best={record['max']:.4f}, Avg={record['avg']:.4f}")

# Get the best individual from the Hall of Fame
best_individual = hof[0]

# Evaluate the best individual on the test set
test_accuracy = evaluate(best_individual, val_loader)[0]

print(f"Best individual's test accuracy: {test_accuracy:.4f}")

# Print the top 5 individuals from the Hall of Fame
print("\nTop 5 individuals:")
for i, ind in enumerate(hof):
    print(f"{i+1}. Fitness: {ind.fitness.values[0]:.4f}")