"""
A basic multi-tree Genetic Programming (GP) algorithm for learning contrastive representations.
"""

import multiprocessing
import torch 
import torch.nn.functional as F
import numpy as np
import random
from deap import base, creator, gp, tools
from sklearn.metrics import balanced_accuracy_score
from typing import List, Tuple
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def prepare_data(data_loader) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Convert data loader into list of (sample1, sample2, label) tuples."""
    data = []
    for batch in data_loader:
        sample1, sample2, labels = batch
        sample1 = F.normalize(sample1.float(), dim=1)
        sample2 = F.normalize(sample2.float(), dim=1)
        data.extend(list(zip(sample1, sample2, labels)))
    return data

class Modi:
    """Container for output index, returns value of its argument on call."""
    def __init__(self, index: int):
        self.index = index

    def __call__(self, x):
        return x

    def __str__(self):
        return f"modi{self.index}"

class MultiOutputTree(gp.PrimitiveTree):
    """Implementation of multiple-output genetic programming tree."""
    num_outputs = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_outputs is None:
            raise Exception("Please initialize class attribute num_outputs")

    def __str__(self):
        """Return the expression in a human-readable string."""
        string_outputs = [""] * self.num_outputs
        stack = []
        for node in self:
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()
                string = prim.format(*args)
                if prim.name[:4] == "modi":
                    index = int(prim.name[4:])
                    if string_outputs[index] != "":
                        string_outputs[index] += "+"
                    string_outputs[index] += string
                if len(stack) == 0:
                    break
                stack[-1][1].append(string)

        string_outputs = [output if output else "0" for output in string_outputs]
        return "[" + ",".join(string_outputs) + "]"

def protected_div(x, y):
    """Protected division function"""
    return np.divide(x, y) if y != 0 else 0

def relu(x):
    """ReLU activation function"""
    return np.maximum(x, 0)

def sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 
                   1 / (1 + np.exp(np.clip(-x, -88.0, 88.0))),
                   np.exp(np.clip(x, -88.0, 88.0)) / (1 + np.exp(np.clip(x, -88.0, 88.0))))

def random_const():
    """Generate random constant between -1 and 1"""
    return random.uniform(-1, 1)

def setup_primitives(n_inputs: int, n_outputs: int) -> gp.PrimitiveSet:
    """Set up primitive set with basic operations."""
    pset = gp.PrimitiveSet("MAIN", n_inputs)
    
    # Add arithmetic operations
    pset.addPrimitive(np.add, 2, name="add")
    pset.addPrimitive(np.subtract, 2, name="sub")
    pset.addPrimitive(np.multiply, 2, name="mul")
    pset.addPrimitive(protected_div, 2, name="div")
    
    # Add activation functions
    pset.addPrimitive(np.tanh, 1, name="tanh")
    pset.addPrimitive(relu, 1, name="relu")
    pset.addPrimitive(sigmoid, 1, name="sigmoid")
    
    # Add constants
    pset.addTerminal(1.0)
    pset.addTerminal(-1.0)
    pset.addEphemeralConstant("rand", random_const)
    
    # Add Modi primitives for each output
    for i in range(n_outputs):
        pset.addPrimitive(Modi(i), 1, name=f"modi{i}")
    
    return pset

def visualize_contrastive_pairs(data, func, save_path="figures/contrastive_pairs.png"):
    """Visualize the cosine similarities between positive pairs colored by class."""
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figures directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    
    similarities = []
    labels = []
    
    # Collect embeddings and compute similarities
    for sample1, sample2, label in data:
        sample1 = sample1.numpy()
        sample2 = sample2.numpy()
        label = label.argmax().item()
        
        # Get embeddings
        out1 = func(*sample1)
        out2 = func(*sample2)
        
        # Convert to numpy arrays if they aren't already
        if not isinstance(out1, np.ndarray):
            out1 = np.array(out1)
        if not isinstance(out2, np.ndarray):
            out2 = np.array(out2)
            
        # Normalize embeddings
        norm1 = np.sqrt(np.sum(out1 * out1))
        norm2 = np.sqrt(np.sum(out2 * out2))
        
        eps = 1e-10
        norm1 = max(norm1, eps)
        norm2 = max(norm2, eps)
        
        out1 = out1 / norm1
        out2 = out2 / norm2
        
        # Compute cosine similarity
        sim = np.sum(out1 * out2)
        sim = (sim + 1) / 2  # Scale to [0,1]
        
        similarities.append(sim)
        labels.append(label)
    
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        plt.scatter(np.random.normal(0, 0.1, size=mask.sum()), 
                   similarities[mask],
                   alpha=0.6,
                   label=f'Class {label}')
    
    plt.axhline(y=similarities.mean(), color='r', linestyle='--', 
                label=f'Mean similarity: {similarities.mean():.3f}')
    
    plt.ylabel('Cosine Similarity')
    plt.xlabel('Jittered x-axis (for visualization)')
    plt.title('Cosine Similarities Between Positive Pairs')
    plt.legend()
    
    # Add text with statistics
    stats_text = (f'Mean: {similarities.mean():.3f}\n'
                 f'Std: {similarities.std():.3f}\n'
                 f'Min: {similarities.min():.3f}\n'
                 f'Max: {similarities.max():.3f}')
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return similarities.mean(), similarities.std()
def evaluate_individual(individual, data, toolbox):
    """Evaluation function that can be pickled."""
    return evaluate(individual, data, toolbox)

def evaluate(individual, data, toolbox):
    """Basic evaluation using balanced accuracy."""
    try:
        func = toolbox.compile(expr=individual)
        
        outputs1 = []
        outputs2 = []
        labels = []
        
        for sample1, sample2, label in data:
            sample1 = sample1.numpy()
            sample2 = sample2.numpy()
            label = label.argmax().item()
            
            out1 = func(*sample1)
            out2 = func(*sample2)
            
            if not isinstance(out1, np.ndarray):
                out1 = np.array(out1)
            if not isinstance(out2, np.ndarray):
                out2 = np.array(out2)
            
            outputs1.append(out1)
            outputs2.append(out2)
            labels.append(label)
        
        outputs1 = np.array(outputs1)
        outputs2 = np.array(outputs2)
        
        # Handle numerical stability
        outputs1 = np.nan_to_num(outputs1, nan=0.0, posinf=1.0, neginf=-1.0)
        outputs2 = np.nan_to_num(outputs2, nan=0.0, posinf=1.0, neginf=-1.0)
        
        eps = 1e-10
        norms1 = np.maximum(np.sqrt(np.sum(outputs1 * outputs1, axis=1, keepdims=True)), eps)
        norms2 = np.maximum(np.sqrt(np.sum(outputs2 * outputs2, axis=1, keepdims=True)), eps)
        
        outputs1 = np.divide(outputs1, norms1, out=np.zeros_like(outputs1), where=norms1>eps)
        outputs2 = np.divide(outputs2, norms2, out=np.zeros_like(outputs2), where=norms2>eps)
        
        similarities = np.sum(outputs1 * outputs2, axis=1)
        similarities = np.clip((similarities + 1) / 2, 0, 1)
        
        # Find best threshold
        thresholds = np.linspace(0, 1, 100)
        accuracies = [balanced_accuracy_score(labels, similarities > t) for t in thresholds]
        best_accuracy = max(accuracies)
        
        return best_accuracy,
        
    except Exception as e:
        return 0.0,

def main():
    # Parameters
    N_INPUTS = 2080
    N_OUTPUTS = 32
    POP_SIZE = 200
    N_GENERATIONS = 500
    N_ELITE = 5
    
    # Load and prepare data
    from util import prepare_dataset, DataConfig
    train_loader, test_loader = prepare_dataset(DataConfig())
    train_data = prepare_data(train_loader)
    test_data = prepare_data(test_loader)
    
    # Setup MultiOutputTree
    MultiOutputTree.num_outputs = N_OUTPUTS
    
    # Create fitness and individual types
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", MultiOutputTree, fitness=creator.FitnessMax)
    
    # Setup toolbox
    pset = setup_primitives(N_INPUTS, N_OUTPUTS)
    toolbox = base.Toolbox()
    
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.register("compile", gp.compile, pset=pset)
    
    # Create initial population
    pop = toolbox.population(n=POP_SIZE)
    
    # Evolution with process pool
    with multiprocessing.Pool() as pool:
        for gen in range(N_GENERATIONS):
            # Evaluate population using starmap for multiple arguments
            fitnesses = pool.starmap(evaluate_individual, 
                                   [(ind, train_data, toolbox) for ind in pop])
            
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit if fit[0] is not None else (0.0,)
            
            # Get statistics
            best_ind = tools.selBest(pop, 1)[0]
            train_acc = evaluate(best_ind, train_data, toolbox)[0]
            test_acc = evaluate(best_ind, test_data, toolbox)[0]
            
            fits = [ind.fitness.values[0] for ind in pop if ind.fitness.values]
            if fits:
                print(f"Gen {gen}: max={max(fits):.3f}, avg={sum(fits)/len(fits):.3f}")
                print(f"Best individual - Train acc: {train_acc:.3f}, Test acc: {test_acc:.3f}")
            else:
                print(f"Gen {gen}: No valid fitness values")

            if gen == N_GENERATIONS - 1:
                break

            # Selection and breeding
            elite = tools.selBest(pop, N_ELITE)
            elite = list(map(toolbox.clone, elite))
            
            offspring = toolbox.select(pop, len(pop) - N_ELITE)
            offspring = list(map(toolbox.clone, offspring))

            # Crossover
            for i in range(1, len(offspring), 2):
                if random.random() < 0.7:
                    offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values

            # Mutation
            for i in range(len(offspring)):
                if random.random() < 0.3:
                    offspring[i], = toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            pop[:] = elite + offspring

    # Visualize best individual
    best_ind = tools.selBest(pop, 1)[0]
    func = toolbox.compile(expr=best_ind)
    
    os.makedirs("figures", exist_ok=True)
    visualize_contrastive_pairs(train_data, func, "figures/gp_train_contrastive_pairs.png")
    visualize_contrastive_pairs(test_data, func, "figures/gp_test_contrastive_pairs.png")

if __name__ == "__main__":
    main()