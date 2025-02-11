"""
A basic multi-tree Genetic Programming (GP) algorithm for learning contrastive representations.
"""

import torch 
import torch.nn.functional as F
import numpy as np
import random
from deap import base, creator, gp, tools
from sklearn.metrics import balanced_accuracy_score
from typing import List, Tuple

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

def setup_primitives(n_inputs: int, n_outputs: int) -> gp.PrimitiveSet:
    """Set up primitive set with basic operations."""
    pset = gp.PrimitiveSet("MAIN", n_inputs)
    
    # Add more complex arithmetic operations
    pset.addPrimitive(np.add, 2, name="add")
    pset.addPrimitive(np.subtract, 2, name="sub")
    pset.addPrimitive(np.multiply, 2, name="mul")
    # Safe division
    pset.addPrimitive(lambda x, y: np.divide(x, y) if y != 0 else 0, 2, name="div")  # Add protected division
    
    # Add more activation functions for non-linearity
    pset.addPrimitive(np.tanh, 1, name="tanh")
    pset.addPrimitive(lambda x: np.maximum(x, 0), 1, name="relu")  # Add ReLU
    # Protected sigmoid to handle overflow
    pset.addPrimitive(lambda x: 1/(1 + np.exp(-np.clip(x, -100, 100))), 1, name="sigmoid")  # Add sigmoid
    
    # Add more diverse constants
    pset.addTerminal(1.0)
    pset.addTerminal(-1.0)
    pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1))
    
    # Add Modi primitives for each output
    for i in range(n_outputs):
        pset.addPrimitive(Modi(i), 1, name=f"modi{i}")
    
    return pset

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
            
            # Convert to numpy arrays if they aren't already
            if not isinstance(out1, np.ndarray):
                out1 = np.array(out1)
            if not isinstance(out2, np.ndarray):
                out2 = np.array(out2)
            
            outputs1.append(out1)
            outputs2.append(out2)
            labels.append(label)
        
        outputs1 = np.array(outputs1)
        outputs2 = np.array(outputs2)
        
        # Replace any NaN or inf values
        outputs1 = np.nan_to_num(outputs1, nan=0.0, posinf=1.0, neginf=-1.0)
        outputs2 = np.nan_to_num(outputs2, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize outputs with better numerical stability
        norms1 = np.sqrt(np.sum(outputs1 * outputs1, axis=1, keepdims=True))
        norms2 = np.sqrt(np.sum(outputs2 * outputs2, axis=1, keepdims=True))
        
        # Handle zero norms with a small epsilon
        eps = 1e-10
        norms1 = np.maximum(norms1, eps)
        norms2 = np.maximum(norms2, eps)
        
        # Safe division
        outputs1 = np.divide(outputs1, norms1, out=np.zeros_like(outputs1), where=norms1>eps)
        outputs2 = np.divide(outputs2, norms2, out=np.zeros_like(outputs2), where=norms2>eps)
        
        # Calculate similarities
        similarities = np.sum(outputs1 * outputs2, axis=1)
        similarities = np.clip((similarities + 1) / 2, 0, 1)  # Scale to [0,1] and clip
        
        # Simple threshold at 0.5
        predictions = (similarities > 0.5).astype(int)
        accuracy = balanced_accuracy_score(labels, predictions)
        
        return accuracy,
        
    except Exception as e:
        return 0.0,

def main():
    # Adjust parameters for better exploration
    N_INPUTS = 2080
    N_OUTPUTS = 2
    POP_SIZE = 200  # Increase population size
    N_GENERATIONS = 100 # Increase number of generations
    N_ELITE = 1  # Number of elite individuals to preserve
    
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
    
    # Setup primitives and toolbox
    pset = setup_primitives(N_INPUTS, N_OUTPUTS)
    toolbox = base.Toolbox()
    
    # Modify genetic operators for better exploration
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    
    # Register genetic operators
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evaluate, data=train_data, toolbox=toolbox)
    
    # Create initial population
    pop = toolbox.population(n=POP_SIZE)
    
    # Simple evolution
    for gen in range(N_GENERATIONS):
        # Evaluate population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            if fit[0] is None:
                ind.fitness.values = (0.0,)
            else:
                ind.fitness.values = fit
        
        # Get best individual
        best_ind = tools.selBest(pop, 1)[0]
        train_acc = toolbox.evaluate(best_ind)[0]
        test_acc = evaluate(best_ind, test_data, toolbox)[0]
        
        # Print statistics
        fits = [ind.fitness.values[0] for ind in pop if ind.fitness.values]
        if fits:
            print(f"Gen {gen}: max={max(fits):.3f}, avg={sum(fits)/len(fits):.3f}")
            print(f"Best individual - Train acc: {train_acc:.3f}, Test acc: {test_acc:.3f}")
        else:
            print(f"Gen {gen}: No valid fitness values")

        # Break if last generation
        if gen == N_GENERATIONS - 1:
            break

        # Select elite individuals
        elite = tools.selBest(pop, N_ELITE)
        elite = list(map(toolbox.clone, elite))

        # Select and clone the rest of the offspring
        offspring = toolbox.select(pop, len(pop) - N_ELITE)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover
        for i in range(1, len(offspring), 2):
            if random.random() < 0.7:
                offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
                del offspring[i-1].fitness.values
                del offspring[i].fitness.values

        # Apply mutation
        for i in range(len(offspring)):
            if random.random() < 0.3:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # Replace population with elite + offspring
        pop[:] = elite + offspring

if __name__ == "__main__":
    main()