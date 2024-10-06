import numpy as np
import operator
from deap import algorithms, base, creator, gp, tools
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from functools import partial
import random
from multiprocessing import Pool
from typing import List, Tuple, Callable, Any

# Define primitives that work with numpy arrays and return float arrays
def protectedDiv(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.divide(left, right, out=np.ones_like(left), where=right!=0)

def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x + y

def sub(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x - y

def mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x * y

def neg(x: np.ndarray) -> np.ndarray:
    return -x

def sin(x: np.ndarray) -> np.ndarray:
    return np.sin(x)

def cos(x: np.ndarray) -> np.ndarray:
    return np.cos(x)

def rand101(x: np.ndarray) -> np.ndarray:
    return np.random.uniform(-1, 1, size=x.shape)

# Function to create the primitive set
def create_pset(n_features: int) -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("MAIN", n_features * 2)  # n_features for each pair
    pset.addPrimitive(add, 2)
    pset.addPrimitive(sub, 2)
    pset.addPrimitive(mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(neg, 1)
    pset.addPrimitive(sin, 1)
    pset.addPrimitive(cos, 1)
    pset.addPrimitive(rand101, 1)
    
    # Rename arguments
    for i in range(n_features):
        pset.renameArguments(**{f'ARG{i}': f'x1_{i}'})
        pset.renameArguments(**{f'ARG{i+n_features}': f'x2_{i}'})
    
    return pset

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

NUM_TREES: int = 10

def create_toolbox(pset: gp.PrimitiveSet) -> base.Toolbox:
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.tree, n=NUM_TREES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox

def compile_trees(individual: List[gp.PrimitiveTree], pset: gp.PrimitiveSet) -> List[Callable[[np.ndarray], np.ndarray]]:
    return [gp.compile(expr, pset) for expr in individual]

@torch.jit.script
def contrastive_loss(output1: torch.Tensor, output2: torch.Tensor, label: float, margin: float = 1.0) -> torch.Tensor:
    euclidean_distance = F.pairwise_distance(output1.unsqueeze(0), output2.unsqueeze(0))
    loss = label * torch.pow(euclidean_distance, 2) + (1 - label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2)
    return torch.mean(loss)

def evalContrastive(individual: List[gp.PrimitiveTree], data: List[Tuple[np.ndarray, np.ndarray, float]], pset: gp.PrimitiveSet, alpha: float = 0.5) -> Tuple[float]:
    trees = compile_trees(individual, pset)
    total_loss = 0.0
    predictions = []
    labels = []
    
    for x1, x2, label in data:
        combined_input = np.concatenate((x1, x2))
        outputs = torch.tensor([np.mean(tree(*combined_input)) for tree in trees], dtype=torch.float32)
        reverse_input = np.concatenate((x2, x1))
        reverse_outputs = torch.tensor([np.mean(tree(*reverse_input)) for tree in trees], dtype=torch.float32)
        
        loss = contrastive_loss(outputs, reverse_outputs, label)
        total_loss += loss.item()
        
        euclidean_distance = F.pairwise_distance(outputs.unsqueeze(0), reverse_outputs.unsqueeze(0))
        pred = 0 if euclidean_distance < 0.5 else 1
        predictions.append(pred)
        labels.append(label)
    
    avg_loss = total_loss / len(data)
    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    fitness = alpha * (1 - balanced_accuracy) + (1 - alpha) * avg_loss
    return (fitness,)

def customCrossover(ind1: List[gp.PrimitiveTree], ind2: List[gp.PrimitiveTree]) -> Tuple[List[gp.PrimitiveTree], List[gp.PrimitiveTree]]:
    for i in range(len(ind1)):
        if random.random() < 0.5:
            ind1[i], ind2[i] = gp.cxOnePoint(ind1[i], ind2[i])
    return ind1, ind2

def customMutate(individual: List[gp.PrimitiveTree], expr: Callable, pset: gp.PrimitiveSet) -> Tuple[List[gp.PrimitiveTree]]:
    for i in range(len(individual)):
        if random.random() < 0.2:
            individual[i], = gp.mutUniform(individual[i], expr=expr, pset=pset)
    return individual,

def eaSimpleWithElitism(population: List[Any], toolbox: base.Toolbox, cxpb: float, mutpb: float, ngen: int, 
                        stats: tools.Statistics, halloffame: tools.HallOfFame, verbose: bool, elite_size: int,
                        train_dataset: List[Tuple[np.ndarray, np.ndarray, float]], 
                        val_dataset: List[Tuple[np.ndarray, np.ndarray, float]],
                        pset: gp.PrimitiveSet) -> Tuple[List[Any], tools.Logbook]:
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

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
        train_balanced_accuracy = evaluate_best_individual(halloffame[0], train_dataset, pset)
        val_balanced_accuracy = evaluate_best_individual(halloffame[0], val_dataset, pset)
        print(f"Generation {gen}: Best Fitness = {best_fit:.4f}, Balanced Accuracy - Train: {train_balanced_accuracy:.4f} Validation: {val_balanced_accuracy:.4f}")

    return population, logbook

def evaluate_best_individual(individual: List[gp.PrimitiveTree], data: List[Tuple[np.ndarray, np.ndarray, float]], pset: gp.PrimitiveSet) -> float:
    trees = compile_trees(individual, pset)
    predictions = []
    labels = []
    
    for x1, x2, label in data:
        combined_input = np.concatenate((x1, x2))
        outputs = torch.tensor([np.mean(tree(*combined_input)) for tree in trees], dtype=torch.float32)
        reverse_input = np.concatenate((x2, x1))
        reverse_outputs = torch.tensor([np.mean(tree(*reverse_input)) for tree in trees], dtype=torch.float32)
        
        euclidean_distance = F.pairwise_distance(outputs.unsqueeze(0), reverse_outputs.unsqueeze(0))
        pred = 0 if euclidean_distance < 0.5 else 1
        predictions.append(pred)
        labels.append(label)
    
    return balanced_accuracy_score(labels, predictions)

def main() -> Tuple[List[Any], tools.Logbook, tools.HallOfFame]:
    from util import preprocess_dataset
    train_loader, val_loader = preprocess_dataset(dataset="instance-recognition", batch_size=64)
    
    def loader_to_list(loader: torch.utils.data.DataLoader) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        data_list = []
        for x1, x2, y in loader:
            for i in range(len(y)):
                data_list.append((x1[i].numpy(), x2[i].numpy(), y[i].item()))
        return data_list
    
    train_data = loader_to_list(train_loader)
    val_data = loader_to_list(val_loader)
    
    # Get the number of features
    n_features = train_data[0][0].shape[0]
    
    # Create primitive set and toolbox
    pset = create_pset(n_features)
    toolbox = create_toolbox(pset)
    
    toolbox.register("evaluate", evalContrastive, data=train_data, pset=pset)
    toolbox.register("mate", customCrossover)
    toolbox.register("mutate", customMutate, expr=toolbox.expr, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop_size = 100
    generations = 50
    elite_size = 5
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(elite_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    with Pool() as pool:
        toolbox.register("map", pool.map)
        pop, log = eaSimpleWithElitism(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, 
                                       stats=stats, halloffame=hof, verbose=True, elite_size=elite_size,
                                       train_dataset=train_data, val_dataset=val_data, pset=pset)
    
    best_individual = hof[0]
    best_fitness = evalContrastive(best_individual, val_data, pset)
    print(f"Best individual fitness on validation set: {best_fitness[0]}")
    
    balanced_accuracy = evaluate_best_individual(best_individual, val_data, pset)
    print(f"Balanced Accuracy Score of the best individual on validation set: {balanced_accuracy:.4f}")

    print(f"Plotting the GP trees")
    # Source: https://deap.readthedocs.io/en/master/tutorials/advanced/gp.html#plotting-trees
    import pygraphviz as pgv

    for tree_idx in range(NUM_TREES):
        nodes, edges, labels = gp.graph(best_individual[tree_idx])

        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")

        for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]

        g.draw(f"figures/tree_{tree_idx}.pdf")
        
    return pop, log, hof

if __name__ == "__main__":
    pop, log, hof = main()