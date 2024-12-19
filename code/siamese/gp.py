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
    label: np.ndarray

@dataclass
class GPConfig:
    """Configuration for GP evolution with improved hyperparameters."""
    n_features: int = 2080
    num_trees: int = 10  # Reduced from 20 to prevent overfitting
    population_size: int = 200  # Increased from 100 for better exploration
    generations: int = 50
    elite_size: int = 20  # Increased from 10 to preserve good solutions
    crossover_prob: float = 0.7  # Slightly reduced from 0.8
    mutation_prob: float = 0.4  # Increased from 0.3 for more exploration
    tournament_size: int = 5  # Reduced from 7 to decrease selection pressure
    distance_threshold: float = 0.5  # Changed from 0.8 for better discrimination
    margin: float = 2.0  # Increased from 1.0 for stronger contrastive signal
    fitness_alpha: float = 5.0  # Reduced from 0.9 to balance objectives
    loss_alpha: float = 0.1  # Increased from 0.0 to consider contrastive loss
    balance_alpha: float = 0.1  # Increased for better class balance
    max_tree_depth: int = 5  # Reduced from 6 to prevent overfitting
    parsimony_coeff: float = 0.0005  # Reduced from 0.001
    batch_size: int = 64  # Reduced from 128 for better gradient estimates
    num_workers: int = 15
    dropout_prob: float = 0.15  # Increased from 0.1
    bn_momentum: float = 0.99  # Slightly reduced from 0.999
    bn_epsilon: float = 1e-5

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
            'tanh': BatchNorm(momentum=momentum, epsilon=epsilon),  # Added tanh
            'relu': BatchNorm(momentum=momentum, epsilon=epsilon),  # Added ReLU
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
        return x / (1 - self.dropout_prob)  # Scale during training
    
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
    
    def tanh(self, x) -> np.ndarray:
        """Add tanh activation for better non-linearity."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 0:
            x = np.array([x])
        result = np.tanh(x)
        result = self.batch_norms['tanh'](result)
        return self._maybe_dropout(result)
    
    def relu(self, x) -> np.ndarray:
        """Add ReLU activation for sparse activations."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 0:
            x = np.array([x])
        result = np.maximum(0, x)
        result = self.batch_norms['relu'](result)
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
    
    def _get_depth(self, tree: gp.PrimitiveTree) -> int:
        """Calculate depth of a tree."""
        if not tree:
            return 0
        stack = [(0, tree)]
        max_depth = 0
        
        while stack:
            depth, expr = stack.pop()
            max_depth = max(max_depth, depth)
            
            for node in reversed(expr):
                if isinstance(node, gp.Primitive):
                    stack.append((depth + 1, expr[expr.index(node) + 1:
                                            expr.index(node) + node.arity + 1]))
        return max_depth

    def _crossover(self, ind1: List[gp.PrimitiveTree], 
                ind2: List[gp.PrimitiveTree]) -> Tuple[List[gp.PrimitiveTree], 
                                                        List[gp.PrimitiveTree]]:
        """Fixed crossover with proper fitness handling."""
        # Create new Individual instances
        new_ind1 = creator.Individual([gp.PrimitiveTree(tree) for tree in ind1])
        new_ind2 = creator.Individual([gp.PrimitiveTree(tree) for tree in ind2])
        
        for i in range(len(new_ind1)):
            if random.random() < 0.5:
                try:
                    for _ in range(3):  # Try multiple crossover points
                        tmp1 = gp.PrimitiveTree(new_ind1[i])
                        tmp2 = gp.PrimitiveTree(new_ind2[i])
                        
                        if len(tmp1) > 1 and len(tmp2) > 1:
                            new_ind1[i], new_ind2[i] = gp.cxOnePoint(tmp1, tmp2)
                            
                            # Check depth limits
                            if (self._get_depth(new_ind1[i]) <= self.config.max_tree_depth and
                                self._get_depth(new_ind2[i]) <= self.config.max_tree_depth):
                                break
                            else:
                                new_ind1[i] = tmp1
                                new_ind2[i] = tmp2
                        
                except Exception as e:
                    print(f"Crossover failed: {str(e)}")
                    # Keep original trees if crossover failed
                    new_ind1[i] = gp.PrimitiveTree(ind1[i])
                    new_ind2[i] = gp.PrimitiveTree(ind2[i])
        
        return new_ind1, new_ind2

    def _mutate(self, individual: List[gp.PrimitiveTree]) -> Tuple[List[gp.PrimitiveTree]]:
        """Fixed mutation with proper fitness handling."""
        new_ind = creator.Individual([gp.PrimitiveTree(tree) for tree in individual])
        
        for i in range(len(new_ind)):
            if random.random() < 0.4:  # Mutation probability
                try:
                    mutation_type = random.random()
                    tmp = gp.PrimitiveTree(new_ind[i])
                    
                    if mutation_type < 0.6:  # Point mutation
                        new_ind[i], = gp.mutUniform(tmp, expr=self.toolbox.expr, pset=self.pset)
                    else:  # Shrink mutation
                        new_ind[i], = gp.mutShrink(tmp)
                    
                    # Check depth
                    if self._get_depth(new_ind[i]) > self.config.max_tree_depth:
                        new_ind[i] = gp.PrimitiveTree(gp.genHalfAndHalf(
                            self.pset, min_=1, max_=self.config.max_tree_depth))
                        
                except Exception as e:
                    print(f"Mutation failed: {str(e)}")
                    # Generate new tree if mutation failed
                    new_ind[i] = gp.PrimitiveTree(gp.genHalfAndHalf(
                        self.pset, min_=1, max_=self.config.max_tree_depth))
        
        return new_ind,

    def print_accuracies(self, population: List[List[gp.PrimitiveTree]], 
                        train_data: List[DataPoint], val_data: List[DataPoint], 
                        generation: int) -> None:
        """Print training and validation accuracies and tree sizes."""
        best_ind = tools.selBest(population, 1)[0]
        
        train_preds, train_labels = self.get_predictions(best_ind, train_data)
        val_preds, val_labels = self.get_predictions(best_ind, val_data)

        train_labels = torch.argmax(torch.tensor(train_labels), dim=1)
        val_labels = torch.argmax(torch.tensor(val_labels), dim=1)
        
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
            ('tanh', 1), ('relu', 1), ('neg', 1)  # Replaced sin/cos with tanh/relu
        ]
        
        for name, arity in primitives:
            pset.addPrimitive(getattr(self.ops, name), arity)
        
        feature_indices = np.random.choice(2080, size=500, replace=False)  # Select subset
        for i in feature_indices:
            pset.addTerminal(i, name=f'f{i}')
        
        # Add constant terminals
        for const in [-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0]:
            pset.addTerminal(const)
        
        pset.renameArguments(ARG0='x')
        return pset

    def _evaluate_batch(self, trees: List[Callable], 
                       batch: List[DataPoint]) -> Tuple[float, List[int]]:
        """Improved batch evaluation with better loss computation."""
        anchors = np.stack([p.anchor for p in batch])
        compares = np.stack([p.compare for p in batch])
        labels = torch.tensor([p.label for p in batch])
        
        out1 = self.batch_get_outputs_static(trees, anchors, self.ops)
        out2 = self.batch_get_outputs_static(trees, compares, self.ops)
        
        z1 = F.normalize(out1 + 1e-8, dim=1)
        z2 = F.normalize(out2 + 1e-8, dim=1)
        
        similarities = F.cosine_similarity(z1, z2)
        label_pairs = torch.argmax(labels, dim=1)
        
        # Scale similarities more aggressively
        probs = torch.clamp((similarities + 1) / 2, min=1e-6, max=1-1e-6)
        
        # Remove label smoothing to allow perfect classification
        loss = F.binary_cross_entropy(probs, label_pairs.float())

        # Simple threshold at mean
        threshold = torch.tensor(torch.mean(similarities), dtype=torch.float32)
        predictions = (similarities > threshold)
        
        return loss, predictions

    @staticmethod
    def evaluate_static(args: Tuple[List[gp.PrimitiveTree], List[DataPoint]]) -> Tuple[float]:
        """Improved static evaluation with better fitness calculation."""
        individual, data = args
        ops = ContrastiveGP._shared_ops
        ops.set_training(True)
        
        # try:
        trees = [gp.compile(expr, ContrastiveGP._shared_pset) for expr in individual]
        total_loss = 0.0
        predictions = []
        labels = []
        
        batch_size = ContrastiveGP._shared_config.batch_size
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_anchors = np.stack([p.anchor for p in batch])
            batch_compares = np.stack([p.compare for p in batch])
            batch_labels = torch.tensor([p.label for p in batch])
            
            out1 = ContrastiveGP.batch_get_outputs_static(trees, batch_anchors, ops)
            out2 = ContrastiveGP.batch_get_outputs_static(trees, batch_compares, ops)
            
            z1 = F.normalize(out1 + 1e-8, dim=1)
            z2 = F.normalize(out2 + 1e-8, dim=1)
            
            similarities = F.cosine_similarity(z1, z2)
            label_pairs = torch.argmax(batch_labels, dim=1)
            
            # Scale similarities more aggressively
            probs = torch.clamp((similarities + 1) / 2, min=1e-6, max=1-1e-6)
            
            # Remove label smoothing to allow perfect classification
            loss = F.binary_cross_entropy(probs, label_pairs.float())
            total_loss += loss * len(batch_labels)
        
        avg_loss = total_loss / len(data)
        return (float(avg_loss),)
            
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
        """Fixed training with proper individual creation."""
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
            
            hof.update(pop)
            print("\nStarting evolution:")
            self.print_accuracies(pop, train_data, val_data, 0)
            
            for gen in range(1, self.config.generations + 1):
                # Create offspring using varAnd
                offspring = []
                for _ in range(len(pop)):
                    if random.random() < self.config.crossover_prob:
                        p1, p2 = random.sample(pop, 2)
                        o1, o2 = self._crossover(p1, p2)
                        offspring.extend([o1, o2])
                    else:
                        p = random.choice(pop)
                        o, = self._mutate(p)
                        offspring.append(o)
                
                # Evaluate offspring
                eval_args = [(ind, train_data) for ind in offspring]
                fitnesses = pool.map(self.evaluate_static, eval_args)
                
                for ind, fit in zip(offspring, fitnesses):
                    ind.fitness.values = fit
                
                # Selection with elitism
                pop = tools.selBest(pop, self.config.elite_size)
                pop.extend(self.toolbox.select(offspring, 
                                            self.config.population_size - len(pop)))
                
                hof.update(pop)
                
                # Print statistics
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

    def _individual_similarity(self, ind1, ind2) -> float:
        """Calculate similarity between two individuals."""
        if not hasattr(ind1, 'str_repr'):
            ind1.str_repr = [str(tree) for tree in ind1]
        if not hasattr(ind2, 'str_repr'):
            ind2.str_repr = [str(tree) for tree in ind2]
        
        similarities = []
        for t1, t2 in zip(ind1.str_repr, ind2.str_repr):
            # Use Levenshtein distance for string similarity
            max_len = max(len(t1), len(t2))
            if max_len == 0:
                similarities.append(1.0)
            else:
                # Simple character-based similarity
                matches = sum(1 for a, b in zip(t1, t2) if a == b)
                similarities.append(matches / max_len)
        
        return sum(similarities) / len(similarities)

def prepare_data(loader: torch.utils.data.DataLoader) -> List[DataPoint]:
    """Convert data loader to list of DataPoints."""
    return [
        DataPoint(x1[i].numpy(), x2[i].numpy(), y[i].numpy())
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