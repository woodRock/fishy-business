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

import functools
import random
import multiprocessing
from dataclasses import dataclass
from typing import List, Tuple, NamedTuple, Dict, Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from deap import algorithms, base, creator, gp, tools
from sklearn.metrics import balanced_accuracy_score
from concurrent.futures import ProcessPoolExecutor

from util import prepare_dataset, DataConfig

class DataPoint(NamedTuple):
    """Single data point for contrastive learning."""
    anchor: np.ndarray
    compare: np.ndarray
    label: float

@dataclass
class GPConfig:
    """Configuration for GP evolution."""
    n_features: int = 2080
    num_trees: int = 10
    population_size: int = 100
    generations: int = 100
    elite_size: int = 10
    crossover_prob: float = 0.8
    mutation_prob: float = 0.3
    tournament_size: int = 7
    distance_threshold: float = 0.5
    margin: float = 2.0
    fitness_alpha: float = 0.6
    max_tree_depth: int = 6
    parsimony_coeff: float = 0.01
    batch_size: int = 128  # Added batch processing
    num_workers: int = 16  # Added parallel processing
    dropout_prob: float = 0.2  # Added dropout probability
    bn_momentum: float = 0.1  # Batch norm momentum
    bn_epsilon: float = 1e-5  # Batch norm epsilon

class BatchNorm:
    """Batch normalization for GP trees with proper shape handling."""
    
    def __init__(self, momentum: float = 0.1, epsilon: float = 1e-5):
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = 0.0
        self.running_var = 1.0
        self.training = True
    
    def set_training(self, mode: bool = True):
        """Set training mode."""
        self.training = mode
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply batch normalization with shape preservation."""
        x = np.asarray(x, dtype=np.float32)
        
        # Handle scalar inputs
        if x.ndim == 0:
            x = x.reshape(1)
            return x
        
        # For batch inputs, preserve batch dimension
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)  # Flatten any extra dimensions
        
        if self.training:
            # Calculate batch statistics along batch dimension
            batch_mean = np.mean(x_flat, axis=0, keepdims=True)
            batch_var = np.var(x_flat, axis=0, keepdims=True) + self.epsilon
            
            # Update running statistics with flattened mean and var
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * np.mean(x_flat)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * np.var(x_flat)
            
            # Normalize
            x_norm = (x_flat - batch_mean) / np.sqrt(batch_var)
        else:
            # Use running statistics
            x_norm = (x_flat - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # Ensure output matches input batch size
        x_norm = x_norm.reshape(batch_size)
        return x_norm.astype(np.float32)

class GPOperations:
    """Primitive operations for GP trees with batch normalization."""
    
    def __init__(self, dropout_prob: float = 0.2):
        self.dropout_prob = dropout_prob
        self.training = True
        # Create batch norm layers for each operation
        self.batch_norms = {
            'add': BatchNorm(),
            'sub': BatchNorm(),
            'mul': BatchNorm(),
            'div': BatchNorm(),
            'sin': BatchNorm(),
            'cos': BatchNorm(),
            'neg': BatchNorm()
        }
    
    def set_training(self, mode: bool = True):
        """Set training mode for dropout and batch norm."""
        self.training = mode
        for bn in self.batch_norms.values():
            bn.set_training(mode)
    
    def _maybe_dropout(self, x):
        """Apply dropout during training."""
        if self.training and random.random() < self.dropout_prob:
            return np.zeros_like(x)
        return x
    
    def add(self, x, y) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if x.ndim == 0:
            x = np.full_like(y, x.item())
        if y.ndim == 0:
            y = np.full_like(x, y.item())
        result = x + y
        result = self.batch_norms['add'](result)
        return self._maybe_dropout(result)
    
    def sub(self, x, y) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if x.ndim == 0:
            x = np.full_like(y, x.item())
        if y.ndim == 0:
            y = np.full_like(x, y.item())
        result = x - y
        result = self.batch_norms['sub'](result)
        return self._maybe_dropout(result)
    
    def mul(self, x, y) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if x.ndim == 0:
            x = np.full_like(y, x.item())
        if y.ndim == 0:
            y = np.full_like(x, y.item())
        result = x * y
        result = self.batch_norms['mul'](result)
        return self._maybe_dropout(result)
    
    def protected_div(self, x, y) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if x.ndim == 0:
            x = np.full_like(y, x.item())
        if y.ndim == 0:
            y = np.full_like(x, y.item())
        result = np.divide(x, y, out=np.ones_like(x), where=y!=0)
        result = self.batch_norms['div'](result)
        return self._maybe_dropout(result)
    
    def sin(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 0:
            x = np.array([x])
        result = np.sin(x)
        result = self.batch_norms['sin'](result)
        return self._maybe_dropout(result)
    
    def cos(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 0:
            x = np.array([x])
        result = np.cos(x)
        result = self.batch_norms['cos'](result)
        return self._maybe_dropout(result)
    
    def neg(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 0:
            x = np.array([x])
        result = -x
        result = self.batch_norms['neg'](result)
        return self._maybe_dropout(result)

class ContrastiveGP:
    """Optimized Genetic Programming for learning contrastive representations."""
    
    def __init__(self, config: GPConfig):
        self.config = config
        self.ops = GPOperations(dropout_prob=config.dropout_prob)
        self.pset = self._init_primitives()
        self.toolbox = self._init_toolbox()
        self.val_data: List[DataPoint] = []
        
        # Pre-compile primitive operations
        self.primitive_lookup = {
            prim.name: prim for prim in self.pset.primitives[object]
        }

    def _init_toolbox(self) -> base.Toolbox:
        """Initialize DEAP toolbox with genetic operators."""
        # Create fitness and individual classes if they don't exist
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)
        
        tb = base.Toolbox()
        
        # Register tree generation operations
        tb.register("expr", gp.genHalfAndHalf, pset=self.pset, 
                min_=1, max_=self.config.max_tree_depth)
        tb.register("tree", tools.initIterate, gp.PrimitiveTree, tb.expr)
        tb.register("individual", tools.initRepeat, creator.Individual, 
                tb.tree, n=self.config.num_trees)
        tb.register("population", tools.initRepeat, list, tb.individual)
        
        # Register genetic operators
        tb.register("mate", self._crossover)
        tb.register("mutate", self._mutate)
        tb.register("select", tools.selTournament, 
                tournsize=self.config.tournament_size)
        
        return tb

    def _crossover(self, ind1: List[gp.PrimitiveTree], 
                ind2: List[gp.PrimitiveTree]) -> Tuple[List[gp.PrimitiveTree], 
                                                        List[gp.PrimitiveTree]]:
        """Perform crossover between individuals with depth limit."""
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = gp.cxOnePoint(ind1[i], ind2[i])
                # Enforce depth limit after crossover
                for idx, tree in enumerate([ind1[i], ind2[i]]):
                    if len(tree) > 2**self.config.max_tree_depth:
                        # Create new tree instead of modifying in place
                        new_tree = gp.PrimitiveTree(gp.genHalfAndHalf(
                            self.pset, 
                            min_=1,
                            max_=self.config.max_tree_depth))
                        if idx == 0:
                            ind1[i] = new_tree
                        else:
                            ind2[i] = new_tree
        return ind1, ind2

    def _mutate(self, individual: List[gp.PrimitiveTree]) -> Tuple[List[gp.PrimitiveTree]]:
        """Mutate an individual with depth limit."""
        for i in range(len(individual)):
            if random.random() < 0.2:
                individual[i], = gp.mutUniform(individual[i], 
                                            expr=self.toolbox.expr, 
                                            pset=self.pset)
                # Enforce depth limit after mutation
                if len(individual[i]) > 2**self.config.max_tree_depth:
                    # Create new tree instead of modifying in place
                    individual[i] = gp.PrimitiveTree(gp.genHalfAndHalf(
                        self.pset, 
                        min_=1,
                        max_=self.config.max_tree_depth))
        return individual,

    def print_accuracies(self, population: List[List[gp.PrimitiveTree]], 
                        train_data: List[DataPoint], val_data: List[DataPoint], 
                        generation: int) -> None:
        """Print training and validation accuracies and tree sizes."""
        best_ind = tools.selBest(population, 1)[0]
        
        train_preds, train_labels = self.get_predictions(best_ind, train_data)
        val_preds, val_labels = self.get_predictions(best_ind, val_data)
        
        train_acc = balanced_accuracy_score(train_labels, train_preds)
        val_acc = balanced_accuracy_score(val_labels, val_preds)
        
        avg_tree_size = sum(len(tree) for tree in best_ind) / len(best_ind)
        
        print(f"Generation {generation:3d} - "
            f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
            f"Avg Tree Size: {avg_tree_size:.1f}")

    def get_predictions(self, individual: List[gp.PrimitiveTree], 
                    data: List[DataPoint]) -> Tuple[List[int], List[float]]:
        """Get predictions and labels for a dataset."""
        self.ops.set_training(False)  # Disable dropout for prediction
        trees = [gp.compile(expr, self.pset) for expr in individual]
        predictions = []
        labels = []
        
        # Process data in batches
        for i in range(0, len(data), self.config.batch_size):
            batch = data[i:i + self.config.batch_size]
            _, batch_preds = self._evaluate_batch(trees, batch)
            predictions.extend(batch_preds)
            labels.extend(p.label for p in batch)
        
        return predictions, labels

    def random_constant(self) -> float:
        """Generate random constant between -1 and 1."""
        return random.uniform(-1, 1)

    def _init_primitives(self) -> gp.PrimitiveSet:
        """Initialize GP primitive operations."""
        pset = gp.PrimitiveSet("MAIN", self.config.n_features)  # Keep input arity as 1
        ops = GPOperations()
        
        primitives = [
            ('add', 2), ('sub', 2), ('mul', 2), ('protected_div', 2),
            ('sin', 1), ('cos', 1), ('neg', 1)
        ]
        
        for name, arity in primitives:
            pset.addPrimitive(getattr(ops, name), arity)
        
        # Add feature selection terminals
        for i in range(2080):  # Add terminals for each feature
            pset.addTerminal(i, name=f'f{i}')
        
        # Add constant terminals for more expressiveness
        for const in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            pset.addTerminal(const)
        
        pset.renameArguments(ARG0='x')
        return pset

    def _batch_get_outputs(self, trees: List[Callable], 
                      batch: np.ndarray) -> torch.Tensor:
        """Process a batch of inputs through trees efficiently."""
        batch_size = batch.shape[0]
        minibatch_size = min(32, batch_size)
        num_minibatches = (batch_size + minibatch_size - 1) // minibatch_size
        
        outputs = []
        for tree in trees:
            tree_outputs = []
            try:
                # Process in minibatches
                for i in range(num_minibatches):
                    start_idx = i * minibatch_size
                    end_idx = min((i + 1) * minibatch_size, batch_size)
                    minibatch = batch[start_idx:end_idx]
                    
                    # FIXED: Process each sample in minibatch
                    minibatch_outputs = []
                    for sample in minibatch:
                        # Pass each feature as a separate argument
                        out = tree(*[x for x in sample])
                        
                        # Handle feature selection
                        if isinstance(out, (int, np.integer)) and 0 <= out < len(sample):
                            out = sample[out]
                        
                        # Ensure output is a float
                        out = np.asarray(out, dtype=np.float32).item()
                        minibatch_outputs.append(out)
                    
                    # Stack minibatch outputs
                    out = np.array(minibatch_outputs, dtype=np.float32)
                    tree_outputs.append(out)
                
                # Concatenate minibatch results
                tree_output = np.concatenate(tree_outputs, axis=0)
                outputs.append(tree_output)
                
            except Exception as e:
                print(f"Error in tree evaluation: {str(e)}")
                outputs.append(np.zeros(batch_size, dtype=np.float32))
        
        outputs = np.stack(outputs).astype(np.float32)
        return torch.from_numpy(outputs).transpose(0, 1)

    def _evaluate_batch(self, trees: List[Callable], 
                   batch: List[DataPoint]) -> Tuple[float, List[int]]:
        """Evaluate a batch of data points with memory-efficient processing."""
        anchors = np.stack([p.anchor for p in batch])
        compares = np.stack([p.compare for p in batch])
        labels = torch.tensor([p.label for p in batch])
        
        out1 = self._batch_get_outputs(trees, anchors)
        out2 = self._batch_get_outputs(trees, compares)
        
        batch_size = len(batch)
        minibatch_size = min(32, batch_size)
        num_minibatches = (batch_size + minibatch_size - 1) // minibatch_size
        
        total_loss = 0.0
        all_predictions = []
        
        for i in range(num_minibatches):
            start_idx = i * minibatch_size
            end_idx = min((i + 1) * minibatch_size, batch_size)
            
            out1_mini = out1[start_idx:end_idx]
            out2_mini = out2[start_idx:end_idx]
            labels_mini = labels[start_idx:end_idx]
            
            distances = F.pairwise_distance(out1_mini, out2_mini)
            # BUGFIX: Reversed prediction logic - small distance means similar
            predictions = (distances < self.config.distance_threshold).int().tolist()
            all_predictions.extend(predictions)
            
            # Compute contrastive loss
            similar_loss = labels_mini * torch.pow(distances, 2)
            dissimilar_loss = (1 - labels_mini) * torch.pow(
                torch.clamp(self.config.margin - distances, min=0.0), 2)
            minibatch_loss = (similar_loss + dissimilar_loss).mean().item()
            total_loss += minibatch_loss * len(labels_mini)
        
        avg_loss = total_loss / batch_size
        return avg_loss, all_predictions

    def _compute_loss(self, output1: torch.Tensor, output2: torch.Tensor, 
                 label: float) -> float:
        """Compute improved contrastive loss between outputs."""
        try:
            # Normalize outputs to unit length
            output1 = F.normalize(output1, p=2, dim=1)
            output2 = F.normalize(output2, p=2, dim=1)
            
            distance = F.pairwise_distance(output1, output2)
            
            # Modified contrastive loss with better gradient properties
            if label == 1:  # Similar pairs
                loss = torch.pow(distance, 2)
            else:  # Dissimilar pairs
                loss = torch.pow(torch.clamp(self.config.margin - distance, min=0.0), 2)
            
            loss = loss.mean().item()
            
            if not np.isfinite(loss):
                return float('inf')
            return loss
            
        except Exception as e:
            print(f"Error in _compute_loss: {str(e)}")
            return float('inf')

    def evaluate(self, individual: List[gp.PrimitiveTree], 
            data: List[DataPoint]) -> Tuple[float]:
        """Evaluate individual on dataset."""
        try:
            self.ops.set_training(True)  # Enable dropout for training
            trees = [gp.compile(expr, self.pset) for expr in individual]
            total_loss = 0.0
            
            predictions, labels = self.get_predictions(individual, data)
            
            for point in data:
                out1 = self._batch_get_outputs(trees, point.anchor.reshape(1, -1))
                out2 = self._batch_get_outputs(trees, point.compare.reshape(1, -1))
                loss = self._compute_loss(out1, out2, point.label)
                if not np.isfinite(loss):
                    return (float('inf'),)
                total_loss += loss
            
            avg_loss = total_loss / len(data)
            accuracy = balanced_accuracy_score(labels, predictions)
            
            # Penalize extreme predictions
            pred_ratio = sum(predictions) / len(predictions)
            balance_penalty = abs(0.5 - pred_ratio)
            
            fitness = (
                0.6 * (1 - accuracy) +
                0.3 * avg_loss +
                0.1 * balance_penalty
            )
            
            if not np.isfinite(fitness):
                return (float('inf'),)
                
            return (float(fitness),)
            
        except Exception as e:
            print(f"Error in evaluate: {str(e)}")
            return (float('inf'),)

    def train(self, train_data: List[DataPoint], 
          val_data: List[DataPoint]) -> Tuple[List[List[gp.PrimitiveTree]], 
                                            tools.Logbook, 
                                            tools.HallOfFame]:
        """Train with robust parallel evaluation of population."""
        self.toolbox.register("evaluate", self.evaluate, data=train_data)
        self.val_data = val_data
        
        pop = self.toolbox.population(n=self.config.population_size)
        hof = tools.HallOfFame(self.config.elite_size)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values[0] 
                            if hasattr(ind.fitness, 'values') 
                            and len(ind.fitness.values) > 0 
                            else float('inf'))
        stats.register("avg", np.nanmean)
        stats.register("std", np.nanstd)
        stats.register("min", np.nanmin)
        stats.register("max", np.nanmax)
        
        def evaluate_population(population, is_offspring=False):
            """Helper function to evaluate population with error handling."""
            chunk_size = 5  # Process smaller chunks to avoid memory issues
            results = []
            
            for i in range(0, len(population), chunk_size):
                chunk = population[i:i + chunk_size]
                retry_count = 0
                max_retries = 3
                
                while retry_count < max_retries:
                    try:
                        with ProcessPoolExecutor(max_workers=min(chunk_size, self.config.num_workers)) as executor:
                            futures = []
                            for ind in chunk:
                                if not ind.fitness.valid:
                                    future = executor.submit(self.evaluate, ind, train_data)
                                    futures.append((ind, future))
                            
                            # Wait for all futures with timeout
                            for ind, future in futures:
                                try:
                                    result = future.result(timeout=60)  # 60 second timeout
                                    results.append((ind, result))
                                except Exception as e:
                                    print(f"Error evaluating individual: {str(e)}")
                                    results.append((ind, (float('inf'),)))
                        
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        retry_count += 1
                        print(f"Process pool error (attempt {retry_count}/{max_retries}): {str(e)}")
                        if retry_count == max_retries:
                            print("Max retries reached, falling back to sequential evaluation")
                            # Fall back to sequential evaluation for this chunk
                            for ind in chunk:
                                if not ind.fitness.valid:
                                    try:
                                        result = self.evaluate(ind, train_data)
                                        results.append((ind, result))
                                    except Exception as e:
                                        print(f"Sequential evaluation error: {str(e)}")
                                        results.append((ind, (float('inf'),)))
            
            # Update fitness values
            for ind, fitness in results:
                ind.fitness.values = fitness
        
        # Evaluate initial population
        print("\nEvaluating initial population...")
        evaluate_population(pop)
        
        # Continue with regular evolution...
        print("\nStarting evolution:")
        self.print_accuracies(pop, train_data, val_data, 0)
        
        for gen in range(1, self.config.generations + 1):
            offspring = algorithms.varAnd(
                pop, self.toolbox, 
                cxpb=self.config.crossover_prob, 
                mutpb=self.config.mutation_prob
            )
            
            # Evaluate offspring
            evaluate_population(offspring, is_offspring=True)
            
            pop = self.toolbox.select(offspring + pop, k=len(pop))
            hof.update(pop)
            
            valid_pop = [ind for ind in pop 
                        if hasattr(ind.fitness, 'values') 
                        and len(ind.fitness.values) > 0]
            
            if valid_pop:
                record = stats.compile(valid_pop)
                self.print_accuracies(pop, train_data, val_data, gen)
                print(f"Stats - avg: {record['avg']:.4f}, "
                    f"std: {record['std']:.4f}, "
                    f"min: {record['min']:.4f}, "
                    f"max: {record['max']:.4f}")
            else:
                print(f"Generation {gen}: No valid individuals found")
        
        return pop, None, hof

def prepare_data(loader: torch.utils.data.DataLoader) -> List[DataPoint]:
    """Convert data loader to list of DataPoints."""
    return [
        DataPoint(x1[i].numpy(), x2[i].numpy(), y[i].item())
        for x1, x2, y in loader
        for i in range(len(y))
    ]

def main():
    """Run the ContrastiveGP training pipeline."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Initialize configurations
    gp_config = GPConfig()
    
    # Prepare data
    print("\nPreparing datasets...")
    train_loader, val_loader = prepare_dataset(DataConfig())  # Assuming this is imported
    
    train_data = prepare_data(train_loader)
    val_data = prepare_data(val_loader)
    
    print(f"Prepared {len(train_data)} training samples and {len(val_data)} validation samples")
    
    # Initialize and train model
    print("\nInitializing ContrastiveGP model...")
    model = ContrastiveGP(gp_config)
    
    print("\nStarting training...")
    final_pop, _, hall_of_fame = model.train(train_data, val_data)
    
    # Print final results
    if hall_of_fame:
        best_individual = hall_of_fame[0]
        train_preds, train_labels = model.get_predictions(best_individual, train_data)
        val_preds, val_labels = model.get_predictions(best_individual, val_data)
        
        final_train_acc = balanced_accuracy_score(train_labels, train_preds)
        final_val_acc = balanced_accuracy_score(val_labels, val_preds)
        
        print("\nFinal Results:")
        print(f"Best Training Accuracy: {final_train_acc:.4f}")
        print(f"Best Validation Accuracy: {final_val_acc:.4f}")
        print(f"Best Fitness: {best_individual.fitness.values[0]:.4f}")

if __name__ == "__main__":
    main()