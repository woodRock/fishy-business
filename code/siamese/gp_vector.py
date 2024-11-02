from dataclasses import dataclass
from typing import List, Tuple, NamedTuple, Dict, Any, Callable
import random

import numpy as np
import torch
import torch.nn.functional as F
from deap import algorithms, base, creator, gp, tools
from sklearn.metrics import balanced_accuracy_score

from util import prepare_dataset, DataConfig

class DataPoint(NamedTuple):
    """Single data point for contrastive learning."""
    anchor: np.ndarray
    compare: np.ndarray
    label: float

@dataclass
class GPConfig:
    """Configuration for GP evolution."""
    num_trees: int = 5
    population_size: int = 10
    generations: int = 50
    elite_size: int = 5
    crossover_prob: float = 0.5
    mutation_prob: float = 0.2
    tournament_size: int = 3
    distance_threshold: float = 0.5
    margin: float = 1.0
    fitness_alpha: float = 0.5

class GPOperations:
    """Primitive operations for GP trees."""
    
    @staticmethod
    def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x.astype(float) + y.astype(float)
    
    @staticmethod
    def sub(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x.astype(float) - y.astype(float)
    
    @staticmethod
    def mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x.astype(float) * y.astype(float)
    
    @staticmethod
    def protected_div(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.divide(x, y, out=np.ones_like(x, dtype=float), where=y!=0)
    
    @staticmethod
    def sin(x: np.ndarray) -> np.ndarray:
        return np.sin(x.astype(float))
    
    @staticmethod
    def cos(x: np.ndarray) -> np.ndarray:
        return np.cos(x.astype(float))
    
    @staticmethod
    def neg(x: np.ndarray) -> np.ndarray:
        return -x.astype(float)

class ContrastiveGP:
    """Genetic Programming for learning contrastive representations."""
    
    def __init__(self, config: GPConfig):
        self.config = config
        self.pset = self._init_primitives()
        self.toolbox = self._init_toolbox()
        self.val_data: List[DataPoint] = []  # Will be set during training

    def _init_primitives(self) -> gp.PrimitiveSet:
        """Initialize GP primitive operations."""
        pset = gp.PrimitiveSet("MAIN", 1)
        ops = GPOperations()
        
        primitives = [
            ('add', 2), ('sub', 2), ('mul', 2), ('protected_div', 2),
            ('sin', 1), ('cos', 1), ('neg', 1)
        ]
        
        for name, arity in primitives:
            pset.addPrimitive(getattr(ops, name), arity)
        
        pset.renameArguments(ARG0='x')
        return pset

    def _init_toolbox(self) -> base.Toolbox:
        """Initialize DEAP toolbox with genetic operators."""
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)
        
        tb = base.Toolbox()
        
        tb.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=6)
        tb.register("tree", tools.initIterate, gp.PrimitiveTree, tb.expr)
        tb.register("individual", tools.initRepeat, creator.Individual, 
                   tb.tree, n=self.config.num_trees)
        tb.register("population", tools.initRepeat, list, tb.individual)
        
        tb.register("mate", self._crossover)
        tb.register("mutate", self._mutate)
        tb.register("select", tools.selTournament, 
                   tournsize=self.config.tournament_size)
        tb.register("map", map)
        
        return tb
    
    def _get_outputs(self, trees: List[Callable], x: np.ndarray) -> torch.Tensor:
        """Apply trees to input and get tensor output."""
        try:
            # Handle potential NaN/inf in tree outputs
            outputs = []
            for tree in trees:
                out = tree(x)
                if not np.all(np.isfinite(out)):
                    out = np.zeros_like(out, dtype=np.float32)
                outputs.append(out)
            
            outputs = np.array(outputs, dtype=np.float32)
            outputs = torch.from_numpy(outputs)
            return outputs.view(1, -1)
            
        except Exception as e:
            print(f"Error in _get_outputs: {str(e)}")
            return torch.zeros((1, len(trees)), dtype=torch.float32)
    
    def get_predictions(self, individual: List[gp.PrimitiveTree], 
                       data: List[DataPoint]) -> Tuple[List[int], List[float]]:
        """Get predictions and labels for a dataset."""
        trees = [gp.compile(expr, self.pset) for expr in individual]
        predictions = []
        labels = []
        
        for point in data:
            out1 = self._get_outputs(trees, point.anchor)
            out2 = self._get_outputs(trees, point.compare)
            
            distance = F.pairwise_distance(out1, out2).mean().item()
            pred = 0 if distance < self.config.distance_threshold else 1
            
            predictions.append(pred)
            labels.append(point.label)
            
        return predictions, labels

    def _compute_loss(self, output1: torch.Tensor, output2: torch.Tensor, 
                     label: float) -> float:
        """Compute contrastive loss between outputs."""
        try:
            distance = F.pairwise_distance(output1, output2)
            similar_loss = label * torch.pow(distance, 2)
            dissimilar_loss = (1 - label) * torch.pow(
                torch.clamp(self.config.margin - distance, min=0.0), 2)
            loss = (similar_loss + dissimilar_loss).mean().item()
            
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
            trees = [gp.compile(expr, self.pset) for expr in individual]
            total_loss = 0.0
            
            predictions, labels = self.get_predictions(individual, data)
            
            for point in data:
                out1 = self._get_outputs(trees, point.anchor)
                out2 = self._get_outputs(trees, point.compare)
                loss = self._compute_loss(out1, out2, point.label)
                if not np.isfinite(loss):
                    return (float('inf'),)
                total_loss += loss
            
            avg_loss = total_loss / len(data)
            accuracy = balanced_accuracy_score(labels, predictions)
            fitness = (self.config.fitness_alpha * (1 - accuracy) + 
                      (1 - self.config.fitness_alpha) * avg_loss)
            
            # Handle any NaN or infinite values
            if not np.isfinite(fitness):
                return (float('inf'),)
                
            return (float(fitness),)
            
        except Exception as e:
            print(f"Error in evaluate: {str(e)}")
            return (float('inf'),)

    def _crossover(self, ind1: List[gp.PrimitiveTree], 
                  ind2: List[gp.PrimitiveTree]) -> Tuple[List[gp.PrimitiveTree], 
                                                       List[gp.PrimitiveTree]]:
        """Perform crossover between individuals."""
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = gp.cxOnePoint(ind1[i], ind2[i])
        return ind1, ind2

    def _mutate(self, individual: List[gp.PrimitiveTree]) -> Tuple[List[gp.PrimitiveTree]]:
        """Mutate an individual."""
        for i in range(len(individual)):
            if random.random() < 0.2:
                individual[i], = gp.mutUniform(individual[i], expr=self.toolbox.expr, 
                                             pset=self.pset)
        return individual,

    def print_accuracies(self, population: List[List[gp.PrimitiveTree]], 
                        train_data: List[DataPoint], val_data: List[DataPoint], 
                        generation: int) -> None:
        """Print training and validation accuracies for best individual."""
        best_ind = tools.selBest(population, 1)[0]
        
        # Get training accuracy
        train_preds, train_labels = self.get_predictions(best_ind, train_data)
        train_acc = balanced_accuracy_score(train_labels, train_preds)
        
        # Get validation accuracy
        val_preds, val_labels = self.get_predictions(best_ind, val_data)
        val_acc = balanced_accuracy_score(val_labels, val_preds)
        
        print(f"Generation {generation:3d} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    def train(self, train_data: List[DataPoint], 
             val_data: List[DataPoint]) -> Tuple[List[List[gp.PrimitiveTree]], 
                                               tools.Logbook, tools.HallOfFame]:
        """Train the GP model."""
        self.toolbox.register("evaluate", self.evaluate, data=train_data)
        self.val_data = val_data
        
        # Initialize evolution
        pop = self.toolbox.population(n=self.config.population_size)
        hof = tools.HallOfFame(self.config.elite_size)
        
        # Setup statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.nanmean)
        stats.register("std", np.nanstd)
        stats.register("min", np.nanmin)
        stats.register("max", np.nanmax)
        
        # Print initial accuracies
        print("\nStarting evolution:")
        self.print_accuracies(pop, train_data, val_data, 0)
        
        # Run evolution with generation callbacks
        for gen in range(1, self.config.generations + 1):
            # Create next generation
            offspring = algorithms.varAnd(
                pop, self.toolbox, 
                cxpb=self.config.crossover_prob, 
                mutpb=self.config.mutation_prob
            )
            
            # Evaluate offspring
            for ind in offspring:
                fit = self.evaluate(ind, train_data)
                ind.fitness.values = fit
            
            # Select next generation
            pop = self.toolbox.select(offspring + pop, k=len(pop))
            
            # Update hall of fame and record stats
            hof.update(pop)
            record = stats.compile(pop)
            
            # Print accuracies for this generation
            self.print_accuracies(pop, train_data, val_data, gen)
            print(f"Stats - avg: {record['avg']:.4f}, std: {record['std']:.4f}, "
                  f"min: {record['min']:.4f}, max: {record['max']:.4f}")
        
        return pop, None, hof

def prepare_data(loader: torch.utils.data.DataLoader) -> List[DataPoint]:
    """Convert data loader to list of DataPoints."""
    return [
        DataPoint(x1[i].numpy(), x2[i].numpy(), y[i].item())
        for x1, x2, y in loader
        for i in range(len(y))
    ]

def main() -> None:
    """Run the ContrastiveGP training."""    
    # Initialize model
    config = GPConfig()
    model = ContrastiveGP(config)
    
    # Prepare data
    config = DataConfig()
    print("\nStarting data preparation...")
    train_loader, val_loader = prepare_dataset(config)

    train_data = prepare_data(train_loader)
    val_data = prepare_data(val_loader)
    
    # Train model
    model.train(train_data, val_data)

if __name__ == "__main__":
    main()