import logging
import torch
import numpy as np
from deap import base, creator, tools, algorithms
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

class GeneticAlgorithm:
    def __init__(self, n_features, n_classes, population_size, crossover_rate, mutation_rate, generations):
        self.n_features = n_features
        self.n_classes = n_classes
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)

        # Set up the DEAP toolbox
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual,
                              lambda: self.create_individual())
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate, mu=0, sigma=0.2, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        self.hof = tools.HallOfFame(5, similar=np.array_equal)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

    def create_individual(self):
        return np.random.rand(self.n_features, self.n_classes)

    def evaluate(self, individual, data_loader):
        individual_tensor = torch.FloatTensor(individual).to(self.device)
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device),  y.to(self.device)
                predictions = torch.argmax(x @ individual_tensor, dim=1)
                correct += (predictions == y.argmax(1)).sum().item()
                total += len(y)
        return correct / total,

    def mutate(self, individual, mu, sigma, indpb):
        for i in range(individual.shape[0]):
            for j in range(individual.shape[1]):
                if np.random.random() < indpb:
                    individual[i, j] += np.random.normal(mu, sigma)
        return individual,

    def train(self, train_loader, val_loader):
        population = self.toolbox.population(n=self.population_size)

        for gen in (pbar := tqdm(range(self.generations), desc="Training: ")):
            # Select and clone the next generation individuals
            offspring = algorithms.varAnd(population, self.toolbox, cxpb=self.crossover_rate, mutpb=self.mutation_rate)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(lambda ind: self.toolbox.evaluate(ind, train_loader), invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            self.hof.update(offspring)

            # Select the next generation population
            population[:] = self.toolbox.select(offspring, k=len(population) - len(self.hof))
            population.extend(self.hof)  # Add the best back to the population

            # Compile statistics about the new population
            record = self.stats.compile(population)
            message = f"Generation {gen+1}: Best={record['max']:.4f}, Avg={record['avg']:.4f}"
            self.logger.info(message)
            pbar.set_description(message)

        # Get the best individual from the Hall of Fame
        best_individual = self.hof[0]

        # Evaluate the best individual on the train and validation sets
        training_accuracy = self.evaluate(best_individual, train_loader)[0]
        val_accuracy = self.evaluate(best_individual, val_loader)[0]
        message = f"Best individual's training accuracy: {training_accuracy:.4f}, validation accuracy: {val_accuracy:.4f}"
        self.logger.info(message)
        print(message)

        return best_individual

    def predict(self, best_individual, data_loader):
        individual_tensor = torch.FloatTensor(best_individual).to(self.device)
        all_predictions = []
        all_true_labels = []

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                predictions = torch.argmax(X_batch @ individual_tensor, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_true_labels.extend(y_batch.argmax(1).numpy())

        return np.array(all_predictions), np.array(all_true_labels)

    def evaluate_model(self, best_individual, data_loader):
        predictions, true_labels = self.predict(best_individual, data_loader)
        balanced_acc = balanced_accuracy_score(true_labels, predictions)
        return balanced_acc