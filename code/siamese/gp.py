"""
A multi-tree Genetic Programming (GP) algorithm for learning contrastive representations with parismony pressure.

Genetic Programming (GP) is a type of evolutionary algorithm that automatically creates 
computer programs by mimicking biological evolution, where programs are represented as 
tree structures that can be combined and mutated. In the case of this code, each "individual" 
is a collection of trees that transform input features (the anchor and compare samples) into 
a new representation space where similar items should be close together and different items
should be far apart. The evolution process repeatedly selects the best-performing 
(based on classification accuracy and contrastive loss), combines them through crossover
(mixing parts of different trees), and applies random mutations (small changes to the trees)
to create new generations that incrementally improve at the task.

Parsimony pressure in Genetic Programming is a technique to prevent "bloat" - the tendency 
of GP trees to grow unnecessarily large and complex over generations. In this code, it's 
implemented through the parsimony coefficient parsimony_coeff = 0.1 which adds a  to the 
fitness based on tree size:

References:
1.  Koza, J. R. (1994). 
    Genetic programming as a means for programming computers by natural selection. 
    Statistics and computing, 4, 87-112.
2.  Soule, T., & Foster, J. A. (1998). 
    Effects of code growth and parsimony pressure on populations in genetic programming. 
    Evolutionary computation, 6(4), 293-309.
3.  Patil, V. P., & Pawar, D. D. (2015). 
    The optimal crossover or mutation rates in genetic algorithm: a review. 
    International Journal of Applied Engineering and Technology, 5(3), 38-41.
"""

import traceback
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import random
from deap import base, creator, gp, tools, algorithms
from sklearn.metrics import balanced_accuracy_score
from typing import List, Tuple

from util import prepare_dataset, DataConfig

def prepare_data(data_loader) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    data = []
    for batch in data_loader:
        sample1, sample2, labels = batch
        sample1 = F.normalize(sample1.float(), dim=1)
        sample2 = F.normalize(sample2.float(), dim=1)
        data.extend(list(zip(sample1, sample2, labels)))
    return data

class Modi:
    """Container for associated output index, returns value of its argument on call."""
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
        """Return the expression in a human-readable string.
        """
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
                    break  # If stack is empty, all nodes should have been seen
                stack[-1][1].append(string)

        string_outputs = [output if output else "0" for output in string_outputs]
        return "[" + ",".join(string_outputs) + "]"

def setup_primitives(n_inputs: int, n_outputs: int) -> gp.PrimitiveSet:
    """Set up the primitive set with basic arithmetic operations and Modi nodes."""
    pset = gp.PrimitiveSet("MAIN", n_inputs)
    
    # Add basic arithmetic operations with vector support
    pset.addPrimitive(np.add, 2, name="vadd")
    pset.addPrimitive(np.multiply, 2, name="vmul")
    pset.addPrimitive(np.subtract, 2, name="vsub")
    pset.addPrimitive(lambda x, y: x / (1 + abs(y)), 2, name="vdiv")
    
    # Add more sophisticated activation functions
    pset.addPrimitive(lambda x: np.tanh(x), 1, name="tanh")
    pset.addPrimitive(lambda x: 1 / (1 + np.exp(-x)), 1, name="sigmoid")
    pset.addPrimitive(lambda x: np.maximum(0, x), 1, name="relu")  # Added ReLU
    pset.addPrimitive(lambda x: x / (1 + np.abs(x)), 1, name="softsign")  # Added Softsign
    
    # Add statistical operations
    pset.addPrimitive(lambda x: x - np.mean(x), 1, name="center")  # Feature centering
    pset.addPrimitive(lambda x: (x - np.mean(x)) / (np.std(x) + 1e-8), 1, name="standardize")  # Standardization
    
    # Add more varied constants
    pset.addEphemeralConstant("rand_normal", lambda: random.gauss(0, 0.5))  # Normal distribution
    pset.addEphemeralConstant("rand_small", lambda: random.uniform(-0.1, 0.1))  # Small uniform
    pset.addEphemeralConstant("rand_large", lambda: random.uniform(-2, 2))  # Large uniform
    
    # Add Modi primitives for each output
    for i in range(n_outputs):
        modi_i = Modi(i)
        pset.addPrimitive(modi_i, 1, name=str(modi_i))
        pset.addTerminal(f"out_{i}", name=f"OUT_{i}")
    
    return pset

def setup_toolbox(pset: gp.PrimitiveSet, min_depth: int, max_depth: int, train_data, val_data) -> base.Toolbox:
    """Set up the DEAP toolbox with genetic operators."""
    toolbox = base.Toolbox()
    
    # Modified tree generation to ensure better output distribution
    def protected_expr():
        tree = gp.genHalfAndHalf(pset=pset, min_=min_depth, max_=max_depth)
        # Ensure at least one Modi node for each output
        outputs_used = set()
        for node in tree:
            if isinstance(node.name, str) and node.name.startswith('modi'):
                outputs_used.add(int(node.name[4:]))
        
        # Add missing outputs
        if len(outputs_used) < MultiOutputTree.num_outputs:
            for i in range(MultiOutputTree.num_outputs):
                if i not in outputs_used:
                    # Find the modi primitive for this output
                    modi_primitive = None
                    for prim in pset.primitives[pset.ret]:
                        if prim.name == f"modi{i}":
                            modi_primitive = prim
                            break
                    
                    if modi_primitive is None:
                        continue
                        
                    # Add a simple expression for unused outputs
                    tree.insert(random.randint(0, len(tree)), modi_primitive)
                    # Add a random terminal
                    terminal = random.choice(pset.terminals[pset.ret])
                    tree.insert(random.randint(0, len(tree)), terminal)
        return tree
    
    toolbox.register("expr", protected_expr)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register genetic operators
    toolbox.register("select", tools.selTournament, tournsize=7)

    # Replace the mate registration with a protected version
    def protected_crossover(ind1, ind2):
        try:
            # Make copies to avoid modifying originals if crossover fails
            ind1_copy = gp.PrimitiveTree(ind1)
            ind2_copy = gp.PrimitiveTree(ind2)
            
            # Attempt crossover
            gp.cxOnePoint(ind1_copy, ind2_copy)
            
            # Validate the resulting trees
            if len(ind1_copy) == 0 or len(ind2_copy) == 0:
                return ind1, ind2
                
            # If successful, update the original individuals
            ind1[:] = ind1_copy
            ind2[:] = ind2_copy
            
            return ind1, ind2
        except (IndexError, ValueError, TypeError):
            # Return unchanged individuals if crossover fails
            return ind1, ind2
    
    # Register the protected crossover instead of the default one
    toolbox.register("mate", protected_crossover)
    
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
    
    # Add a protected mutation operator
    def protected_mutation(individual, expr, pset):
        try:
            # Make a copy of the individual
            ind_copy = gp.PrimitiveTree(individual)
            
            # Attempt mutation
            gp.mutUniform(ind_copy, expr=expr, pset=pset)
            
            # Validate the resulting tree
            if len(ind_copy) == 0:
                return individual,
                
            # If successful, update the original individual
            individual[:] = ind_copy
            return individual,
        except (IndexError, ValueError, TypeError):
            # Return unchanged individual if mutation fails
            return individual,
    
    toolbox.register("mutate", protected_mutation, expr=toolbox.expr_mut, pset=pset)
    
    toolbox.register("compile", gp.compile, pset=pset)
    
    # Register evaluation function
    toolbox.register("evaluate", dummy_evaluate, data=train_data, val_data=val_data, toolbox=toolbox)
    
    return toolbox

def dummy_evaluate(individual, data=None, val_data=None, toolbox=None):
    """
    Evaluation function using balanced accuracy and NT-Xent loss as fitness metrics.
    """
    try:
        compiled_func = toolbox.compile(expr=individual)

        def safe_wrapper(*args):
            try:
                result = compiled_func(*args)
                # Convert result to float array and ensure correct shape
                if isinstance(result, (list, tuple)):
                    result = [float(x) if isinstance(x, str) else x for x in result]
                result = np.array(result, dtype=np.float32).reshape(-1)
                if len(result) != MultiOutputTree.num_outputs:
                    return np.zeros(MultiOutputTree.num_outputs, dtype=np.float32)
                return result
            except (TypeError, IndexError, ValueError, AttributeError) as e:
                return np.zeros(MultiOutputTree.num_outputs, dtype=np.float32)

        def nt_xent_loss(outputs1, outputs2, labels, temperature=0.07):  # Reduced temperature
            try:
                outputs1 = np.array(outputs1, dtype=np.float32)
                outputs2 = np.array(outputs2, dtype=np.float32)
                labels = np.array(labels, dtype=np.int32)
                batch_size = len(outputs1)
                
                # Enhanced normalization with epsilon
                outputs1 = outputs1 / (np.linalg.norm(outputs1, axis=1, keepdims=True) + 1e-8)
                outputs2 = outputs2 / (np.linalg.norm(outputs2, axis=1, keepdims=True) + 1e-8)
                
                all_outputs = np.concatenate([outputs1, outputs2], axis=0)
                
                # Use cosine similarity with temperature scaling
                sim_matrix = np.dot(all_outputs, all_outputs.T) / temperature
                
                all_labels = np.concatenate([labels, labels])
                
                pos_mask = (all_labels.reshape(-1, 1) == all_labels.reshape(1, -1)).astype(np.float32)
                neg_mask = (all_labels.reshape(-1, 1) != all_labels.reshape(1, -1)).astype(np.float32)
                
                # Remove self-similarity
                pos_mask = pos_mask - np.eye(2 * batch_size)
                
                # Calculate positive and negative scores with improved numerical stability
                pos_scores = (sim_matrix * pos_mask).sum(axis=1) / (pos_mask.sum(axis=1) + 1e-8)
                neg_scores = (sim_matrix * neg_mask).sum(axis=1) / (neg_mask.sum(axis=1) + 1e-8)
                
                # Adaptive margin based on label distribution
                unique_labels = len(np.unique(labels))
                margin = 0.2 + 0.1 * (unique_labels / batch_size)  # Adaptive margin
                
                loss = np.mean(np.maximum(0, neg_scores - pos_scores + margin))
                return loss
                
            except Exception as e:
                print(f"Error in NT-Xent loss calculation: {e}")
                traceback.print_exc()
                return 1.0

        # Process training data
        outputs1_train = []
        outputs2_train = []
        y_true = []
        
        for sample1, sample2, label in data:
            label = label.argmax(dim=0)
            sample1 = sample1.detach().cpu().numpy()
            sample2 = sample2.detach().cpu().numpy()
            
            output1 = safe_wrapper(*sample1)
            output2 = safe_wrapper(*sample2)
            
            outputs1_train.append(output1)
            outputs2_train.append(output2)
            y_true.append(label.item() if torch.is_tensor(label) else label)

        # Calculate cosine similarities instead of distances
        outputs1_train = np.array(outputs1_train)
        outputs2_train = np.array(outputs2_train)
        similarities = np.sum(outputs1_train * outputs2_train, axis=1) / (
            np.linalg.norm(outputs1_train, axis=1) * np.linalg.norm(outputs2_train, axis=1) + 1e-8
        )
        labels = np.array(y_true)
        
        # Find optimal threshold using training data
        sorted_sims = np.sort(similarities)
        best_threshold = sorted_sims[0]
        best_accuracy = 0
        
        for threshold in sorted_sims:
            y_pred = (similarities > threshold).astype(int)  # Note: changed to > for similarities
            accuracy = balanced_accuracy_score(labels, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        train_accuracy = best_accuracy
        train_contrastive_loss = nt_xent_loss(outputs1_train, outputs2_train, y_true)

        # Process validation data
        if val_data:
            val_y_true = []
            val_y_pred = []
            outputs1_val = []
            outputs2_val = []
            
            for sample1, sample2, label in val_data:
                label = label.argmax(dim=0)
                sample1 = sample1.detach().cpu().numpy()
                sample2 = sample2.detach().cpu().numpy()
                
                output1 = safe_wrapper(*sample1)
                output2 = safe_wrapper(*sample2)
                
                outputs1_val.append(output1)
                outputs2_val.append(output2)
                
                # Calculate cosine similarity
                similarity = np.dot(output1, output2) / (
                    np.linalg.norm(output1) * np.linalg.norm(output2) + 1e-8
                )
                
                val_y_true.append(label.item() if torch.is_tensor(label) else label)
                val_y_pred.append(1 if similarity > best_threshold else 0)
            
            val_accuracy = balanced_accuracy_score(val_y_true, val_y_pred)
            val_contrastive_loss = nt_xent_loss(
                np.array(outputs1_val), 
                np.array(outputs2_val), 
                val_y_true
            )
            
            # Modified fitness calculation with stronger emphasis on accuracy
            train_component = train_accuracy * (1 - 0.3 * train_contrastive_loss)  # Less weight on contrastive loss
            val_component = val_accuracy * (1 - 0.3 * val_contrastive_loss)
            
            # Enhanced diversity bonus
            unique_ops = len(set(str(node) for node in individual))
            total_ops = len(individual)
            diversity_bonus = (unique_ops / total_ops) * (1 - np.exp(-total_ops/50))  # Scaled diversity bonus
            
            # Adjusted fitness components
            fitness = (0.7 * train_component + 
                      0.1 * val_component + 
                      0.2 * diversity_bonus)
        else:
            train_component = train_accuracy * (1 - 0.3 * train_contrastive_loss)
            diversity_bonus = (len(set(str(node) for node in individual)) / len(individual))
            fitness = 0.8 * train_component + 0.2 * diversity_bonus
            val_accuracy = 0.0
        
        # Dynamic parsimony pressure
        base_parsimony = 0.003  # Reduced base coefficient
        tree_size = len(individual)
        
        # Adaptive parsimony pressure based on accuracy and tree size
        if train_accuracy > 0.8:
            parsimony_coeff = base_parsimony * (tree_size / 50)  # Increased pressure for larger trees
        else:
            parsimony_coeff = base_parsimony * 0.5  # Reduced pressure for less accurate solutions
        
        size_penalty = parsimony_coeff * tree_size / 100
        
        # Only apply size penalty for sufficiently accurate solutions
        if train_accuracy > 0.75:
            fitness -= size_penalty
        
        return fitness, train_accuracy, val_accuracy
    
    except Exception as e:
        print(f"Evaluation error: {e}")
        traceback.print_exc()
        return 0.0, 0.0, 0.0

def main():
    # Adjusted parameters for better exploration
    N_INPUTS = 2080
    N_OUTPUTS = 128
    POP_SIZE = 300  # Increased population size
    N_GENERATIONS = 50
    MIN_DEPTH = 2
    MAX_DEPTH = 7  # Slightly increased max depth
    ELITE_SIZE = 15  # Increased elite size

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print("\nPreparing datasets...")
    train_loader, val_loader = prepare_dataset(DataConfig())
    train_data = prepare_data(train_loader)
    val_data = prepare_data(val_loader)
    print(f"Prepared {len(train_data)} training samples and {len(val_data)} validation samples")
    
    # Setup MultiOutputTree
    MultiOutputTree.num_outputs = N_OUTPUTS
    
    # Create fitness and individual types
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 0.01, 0.01))
    creator.create("Individual", MultiOutputTree, num_outputs=N_OUTPUTS, fitness=creator.FitnessMax)
    
    # Setup primitives and toolbox
    pset = setup_primitives(N_INPUTS, N_OUTPUTS)
    toolbox = setup_toolbox(pset, MIN_DEPTH, MAX_DEPTH, train_data, val_data)
    
    # Initialize population and hall of fame
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(5)
    
    # Statistics setup
    stats = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else 0)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Custom elitist evolution
    for gen in range(N_GENERATIONS):
        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Update hall of fame
        hof.update(pop)

        # Get best accuracies
        best_ind = max(pop, key=lambda x: x.fitness.values[0])
        _, best_train_acc, best_val_acc = best_ind.fitness.values

        # Select elite individuals
        pop.sort(key=lambda x: x.fitness.values[0], reverse=True)
        elites = pop[:ELITE_SIZE]

        # Select and clone the next generation individuals
        offspring = toolbox.select(pop, len(pop) - ELITE_SIZE)
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for i in range(1, len(offspring), 2):
            if random.random() < 0.8:  # crossover probability
                if i < len(offspring) - 1:  # Make sure we have a pair
                    toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < 0.4:  # mutation probability
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # The elite individuals pass directly to the next generation
        pop = elites + offspring

        # Compile stats
        record = stats.compile(pop)
        print(f"Generation {gen}: Max = {record['max']:.3f}, Avg = {record['avg']:.3f}, "
              f"Best Train Acc = {best_train_acc:.3f}, Best Val Acc = {best_val_acc:.3f}")

    return pop, hof

if __name__ == "__main__":
    main()