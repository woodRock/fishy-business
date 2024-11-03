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
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    num_trees: int = 20
    population_size: int = 100
    generations: int = 100
    elite_size: int = 10
    crossover_prob: float = 0.8
    mutation_prob: float = 0.3
    tournament_size: int = 7
    distance_threshold: float = 0.8
    margin: float = 1.0
    fitness_alpha: float = 0.8
    loss_alpha: float = 0.2
    balance_alpha: float = 0.001
    max_tree_depth: int = 6
    parsimony_coeff: float = 0.001
    batch_size: int = 128  # Added batch processing
    num_workers: int = 15  # Added parallel processing
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
    
    def __init__(self, dropout_prob: float = 0.2, momentum = 0.1, epsilon=1e-5):
        self.dropout_prob = dropout_prob
        self.training = True
        # Create batch norm layers for each operation
        self.batch_norms = {
            'add': BatchNorm(momentum=momentum, epsilon=epsilon),
            'sub': BatchNorm(momentum=momentum, epsilon=epsilon),
            'mul': BatchNorm(momentum=momentum, epsilon=epsilon),
            'div': BatchNorm(momentum=momentum, epsilon=epsilon),
            'sin': BatchNorm(momentum=momentum, epsilon=epsilon),
            'cos': BatchNorm(momentum=momentum, epsilon=epsilon),
            'neg': BatchNorm(momentum=momentum, epsilon=epsilon)
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
    
    # Class-level shared variables
    _shared_pset = None
    _shared_config = None
    _shared_ops = None  # Add shared ops instance
    
    def __init__(self, config: GPConfig):
        self.config = config
        # Create single shared ops instance
        if ContrastiveGP._shared_ops is None:
            ContrastiveGP._shared_ops = GPOperations(
                dropout_prob=config.dropout_prob,
                momentum=config.bn_momentum, 
                epsilon=config.bn_epsilon
            )
        self.ops = ContrastiveGP._shared_ops
        
        # Initialize the primitive set and store as class variable
        ContrastiveGP._shared_pset = self._init_primitives()
        ContrastiveGP._shared_config = config
        self.pset = ContrastiveGP._shared_pset
        self.toolbox = self._init_toolbox()
        self.val_data: List[DataPoint] = []

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

    def _init_primitives(self) -> gp.PrimitiveSet:
        """Initialize GP primitive operations using shared ops."""
        pset = gp.PrimitiveSet("MAIN", self.config.n_features)
        
        primitives = [
            ('add', 2), ('sub', 2), ('mul', 2), ('protected_div', 2),
            ('sin', 1), ('cos', 1), ('neg', 1)
        ]
        
        for name, arity in primitives:
            pset.addPrimitive(getattr(self.ops, name), arity)
        
        # Add feature selection terminals
        for i in range(2080):
            pset.addTerminal(i, name=f'f{i}')
        
        # Add constant terminals
        for const in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            pset.addTerminal(const)
        
        pset.renameArguments(ARG0='x')
        return pset

    def _evaluate_batch(self, trees: List[Callable], 
                   batch: List[DataPoint]) -> Tuple[float, List[int]]:
        """Evaluate a batch of data points with memory-efficient processing."""
        anchors = np.stack([p.anchor for p in batch])
        compares = np.stack([p.compare for p in batch])
        labels = torch.tensor([p.label for p in batch])
        
        out1 = self.batch_get_outputs_static(trees, anchors, self.ops)
        out2 = self.batch_get_outputs_static(trees, compares, self.ops)
        
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

    @staticmethod
    def evaluate_static(args: Tuple[List[gp.PrimitiveTree], List[DataPoint]]) -> Tuple[float]:
        """Static evaluation method using shared ops instance.
        
        Args:
            args: Tuple containing (individual, data)
            
        Returns:
            Tuple containing single fitness value
        """
        individual, data = args
        
        # Use shared ops instance instead of creating new one
        ops = ContrastiveGP._shared_ops
        ops.set_training(True)
        
        try:
            trees = [gp.compile(expr, ContrastiveGP._shared_pset) for expr in individual]
            total_loss = 0.0
            predictions = []
            labels = []
            
            # Process data in batches
            batch_size = ContrastiveGP._shared_config.batch_size
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                batch_anchors = np.stack([p.anchor for p in batch])
                batch_compares = np.stack([p.compare for p in batch])
                batch_labels = torch.tensor([p.label for p in batch])
                
                # Get outputs using shared ops instance
                out1 = ContrastiveGP.batch_get_outputs_static(trees, batch_anchors, ops)
                out2 = ContrastiveGP.batch_get_outputs_static(trees, batch_compares, ops)
                
                # Normalize outputs
                out1 = F.normalize(out1, p=2, dim=1)
                out2 = F.normalize(out2, p=2, dim=1)
                
                # Calculate distances and predictions
                distances = F.pairwise_distance(out1, out2)
                threshold = ContrastiveGP._shared_config.distance_threshold
                batch_preds = (distances < threshold).int().tolist()
                predictions.extend(batch_preds)
                labels.extend(p.label for p in batch)
                
                # Compute contrastive loss
                margin = ContrastiveGP._shared_config.margin
                similar_loss = batch_labels * torch.pow(distances, 2)
                dissimilar_loss = (1 - batch_labels) * torch.pow(
                    torch.clamp(margin - distances, min=0.0), 2)
                batch_loss = (similar_loss + dissimilar_loss).mean().item()
                total_loss += batch_loss * len(batch_labels)
            
            avg_loss = total_loss / len(data)
            accuracy = balanced_accuracy_score(labels, predictions)
            
            pred_ratio = sum(predictions) / len(predictions)
            balance_penalty = abs(0.5 - pred_ratio)

            print(f"accuracy: {accuracy}")
            print(f"avg_loss: {avg_loss}")
            print(f"balance_penalty: {balance_penalty}")
            print(f"parsimony: {sum(len(tree) for tree in individual)}")
            
            # Get coefficients from shared config
            fitness = (
                ContrastiveGP._shared_config.fitness_alpha * (1 - accuracy) #+
                # ContrastiveGP._shared_config.loss_alpha * avg_loss +
                # ContrastiveGP._shared_config.balance_alpha * balance_penalty +
                ContrastiveGP._shared_config.parsimony_coeff * sum(len(tree) for tree in individual)
            )
            
            return (float(fitness),)
            
        except Exception as e:
            print(f"Error in evaluate_static: {str(e)}")
            return (float('inf'),)
        finally:
            ops.set_training(False)
            
    @staticmethod
    def batch_get_outputs_static(trees: List[Callable], 
                               batch: np.ndarray,
                               ops: GPOperations) -> torch.Tensor:
        """Static method for batch processing outputs."""
        batch_size = batch.shape[0]
        outputs = []
        
        for tree in trees:
            tree_outputs = []
            for sample in batch:
                try:
                    out = tree(*[x for x in sample])
                    if isinstance(out, (int, np.integer)) and 0 <= out < len(sample):
                        out = sample[out]
                    out = np.asarray(out, dtype=np.float32).item()
                    tree_outputs.append(out)
                except Exception as e:
                    tree_outputs.append(0.0)
            
            outputs.append(np.array(tree_outputs, dtype=np.float32))
        
        outputs = np.stack(outputs).astype(np.float32)
        return torch.from_numpy(outputs).transpose(0, 1)

    def train(self, train_data: List[DataPoint], 
              val_data: List[DataPoint]) -> Tuple[List[List[gp.PrimitiveTree]], 
                                                 tools.Logbook, 
                                                 tools.HallOfFame]:
        """Train with multiprocessing Pool for parallel evaluation."""
        self.val_data = val_data
        
        # Create initial population
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
        
        with multiprocessing.Pool(processes=self.config.num_workers) as pool:
            # Evaluate initial population
            print("\nEvaluating initial population...")
            eval_args = [(ind, train_data) for ind in pop]
            fitnesses = pool.map(self.evaluate_static, eval_args)
            
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            
            print("\nStarting evolution:")
            self.print_accuracies(pop, train_data, val_data, 0)
            
            for gen in range(1, self.config.generations + 1):
                offspring = algorithms.varAnd(
                    pop, self.toolbox,
                    cxpb=self.config.crossover_prob,
                    mutpb=self.config.mutation_prob
                )
                
                # Evaluate offspring
                eval_args = [(ind, train_data) for ind in offspring]
                fitnesses = pool.map(self.evaluate_static, eval_args)
                
                for ind, fit in zip(offspring, fitnesses):
                    ind.fitness.values = fit
                
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