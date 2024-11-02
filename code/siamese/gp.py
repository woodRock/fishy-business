"""
Multi-tree genetic programming for contrastive learning.

This module implements genetic programming for learning contrastive representations
of data using multiple trees. It evolves a population of GP trees that transform
input features into a space where similar instances are close and dissimilar ones
are far apart.

References: 
1. Bromley, J., et al. (1993). Signature verification using a "siamese" 
   time delay neural network. Advances in neural information processing systems, 6.
2. Koza, J. R. (1994). Genetic programming II: automatic discovery of 
   reusable programs.
"""

import logging
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Dict, Any, Union
import random
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deap import algorithms, base, creator, gp, tools
from sklearn.metrics import balanced_accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

# Type aliases
Individual = List[gp.PrimitiveTree]
Population = List[Individual]
Features = np.ndarray
Label = float
DataPoint = Tuple[Features, Features, Label]
Dataset = List[DataPoint]

@dataclass
class GPConfig:
    """Configuration parameters for GP run."""
    file_path: Path = Path("checkpoints/embedded-gp.pth")
    dataset: str = "instance-recognition"
    load_checkpoint: bool = False
    run_number: int = 0
    output_dir: Path = Path("logs/results")
    population_size: int = 100
    generations: int = 10
    mutation_rate: float = 0.2
    crossover_rate: float = 0.8
    elite_size: int = 5
    tree_depth: int = 6
    num_trees: int = 20
    tournament_size: int = 7
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'GPConfig':
        """Create config from command line arguments."""
        return cls(
            file_path=Path(args.file_path),
            dataset=args.dataset,
            load_checkpoint=args.load,
            run_number=args.run,
            output_dir=Path(args.output),
            population_size=args.population,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            elite_size=args.elitism,
            tree_depth=args.tree_depth,
            num_trees=args.num_trees
        )

class GPPrimitives:
    """Primitive operations for GP trees."""
    
    @staticmethod
    def protected_div(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Protected division that handles division by zero."""
        return np.divide(left, right, out=np.ones_like(left, dtype=float), where=right!=0)
    
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
    def neg(x: np.ndarray) -> np.ndarray:
        return -x.astype(float)
    
    @staticmethod
    def sin(x: np.ndarray) -> np.ndarray:
        return np.sin(x.astype(float))
    
    @staticmethod
    def cos(x: np.ndarray) -> np.ndarray:
        return np.cos(x.astype(float))
    
    @staticmethod
    def rand101(x: np.ndarray) -> np.ndarray:
        return np.random.uniform(-1, 1, size=x.shape)

class ContrastiveLoss:
    """Implementation of contrastive loss functions."""
    
    @staticmethod
    def cosine_contrastive(z1: torch.Tensor, z2: torch.Tensor, y: Union[float, torch.Tensor], 
                          temperature: float = 0.5) -> torch.Tensor:
        """Compute contrastive loss using cosine similarity."""
        similarity = F.cosine_similarity(z1, z2)
        loss = y * torch.pow(1 - similarity, 2) + \
               (1 - y) * torch.pow(torch.clamp(similarity - 0.1, min=0.0), 2)
        return loss.mean()

class ContrastiveGP:
    """Main class for contrastive genetic programming implementation."""
    
    def __init__(self, config: GPConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.pset = self._setup_primitives()
        self.toolbox = self._setup_toolbox()
        
    def _setup_primitives(self) -> gp.PrimitiveSet:
        """Initialize primitive set for GP."""
        pset = gp.PrimitiveSet("MAIN", 2080)
        primitives = GPPrimitives()
        
        primitive_ops = [
            ('add', 2), ('sub', 2), ('mul', 2), ('protected_div', 2),
            ('neg', 1), ('sin', 1), ('cos', 1), ('rand101', 1)
        ]
        
        for name, arity in primitive_ops:
            pset.addPrimitive(getattr(primitives, name), arity)
            
        pset.renameArguments(ARG0='x')
        return pset
    
    def _setup_toolbox(self) -> base.Toolbox:
        """Initialize DEAP toolbox with genetic operators."""
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        
        # Register genetic operators
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, 
                        min_=1, max_=self.config.tree_depth)
        toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                        toolbox.tree, n=self.config.num_trees)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Register genetic operations
        toolbox.register("mate", self._crossover)
        toolbox.register("mutate", self._mutate)
        toolbox.register("select", tools.selTournament, 
                        tournsize=self.config.tournament_size)
        
        return toolbox
    
    def _crossover(self, ind1: Individual, ind2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two individuals."""
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = gp.cxOnePoint(ind1[i], ind2[i])
        return ind1, ind2
    
    def _mutate(self, individual: Individual) -> Tuple[Individual]:
        """Mutate an individual."""
        for i in range(len(individual)):
            if random.random() < self.config.mutation_rate:
                individual[i], = gp.mutUniform(individual[i], expr=self.toolbox.expr,
                                             pset=self.pset)
        return individual,
    
    def evaluate(self, individual: Individual, data: Dataset) -> Tuple[float]:
        """Evaluate a single individual on the given dataset."""
        return self._evaluate_individual_with_pset((individual, data, self.pset))
    
    @staticmethod
    def _evaluate_individual_with_pset(args: Tuple[Individual, Dataset, gp.PrimitiveSet]) -> Tuple[float]:
        """Static evaluation method that includes primitive set in arguments."""
        individual, data, primitive_set = args
        trees = [gp.compile(expr, primitive_set) for expr in individual]
        total_loss = 0
        predictions = []
        labels = []
        
        for x1, x2, label in data:
            outputs1 = torch.tensor(np.array([tree(*x1) for tree in trees]), dtype=torch.float32)
            outputs2 = torch.tensor(np.array([tree(*x2) for tree in trees]), dtype=torch.float32)
            
            # Calculate contrastive loss
            similarity = F.cosine_similarity(outputs1.unsqueeze(0), outputs2.unsqueeze(0))
            loss = label * torch.pow(1 - similarity, 2) + \
                   (1 - label) * torch.pow(torch.clamp(similarity - 0.1, min=0.0), 2)
            total_loss += loss.mean().item()
            
            preds = (similarity > 0.5).float()
            predictions.append(preds)
            labels.append(label)
        
        avg_loss = total_loss / len(data)
        balanced_acc = balanced_accuracy_score(labels, predictions)
        fitness = 0.5 * (1 - balanced_acc) + 0.5 * avg_loss
        return (fitness,)
    
    def _parallel_evaluate(self, individuals: List[Individual], data: Dataset, 
                          pool: Pool) -> List[Tuple[float]]:
        """Evaluate individuals in parallel."""
        eval_args = [(ind, data, self.pset) for ind in individuals]
        return list(pool.map(self._evaluate_individual_with_pset, eval_args))
    
    def train(self, train_data: Dataset, val_data: Dataset) -> Tuple[Population, tools.Logbook, 
                                                                    tools.HallOfFame]:
        """Train the GP model."""
        pop = self.toolbox.population(n=self.config.population_size)
        hof = tools.HallOfFame(self.config.elite_size)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        with Pool() as pool:
            pop, log = self._run_evolution(pop, hof, stats, train_data, val_data, pool)
        
        return pop, log, hof
    
    def _run_evolution(self, population: Population, hof: tools.HallOfFame,
                      stats: tools.Statistics, train_data: Dataset,
                      val_data: Dataset, pool: Pool) -> Tuple[Population, tools.Logbook]:
        """Run the evolutionary process."""
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        
        # Initial evaluation
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self._parallel_evaluate(invalid_ind, train_data, pool)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        hof.update(population)
        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        
        # Evolution loop
        for gen in tqdm(range(1, self.config.generations + 1), desc="Training"):
            # Selection
            offspring = self.toolbox.select(population, 
                                          len(population) - self.config.elite_size)
            
            # Apply variation operators
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover
            for i in range(1, len(offspring), 2):
                if random.random() < self.config.crossover_rate:
                    offspring[i-1], offspring[i] = self.toolbox.mate(offspring[i-1], 
                                                                   offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values
            
            # Apply mutation
            for i in range(len(offspring)):
                if random.random() < self.config.mutation_rate:
                    offspring[i], = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values
            
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self._parallel_evaluate(invalid_ind, train_data, pool)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Add elite individuals back
            offspring.extend(hof.items)
            
            # Update population and hall of fame
            population[:] = offspring
            hof.update(population)
            
            # Log progress
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            
            # Evaluate and log performance
            best_fit = hof[0].fitness.values[0]
            train_acc = self.evaluate_accuracy(hof[0], train_data)
            val_acc = self.evaluate_accuracy(hof[0], val_data)
            
            self.logger.info(
                f"Gen {gen}: Best Fitness = {best_fit:.4f}, "
                f"Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}"
            )
        
        return population, logbook
    
    def _evolve_generation(self, population: Population, 
                          hof: tools.HallOfFame) -> Population:
        """Evolve population for one generation."""
        offspring = self.toolbox.select(population, 
                                      len(population) - self.config.elite_size)
        offspring = algorithms.varAnd(offspring, self.toolbox,
                                    self.config.crossover_rate,
                                    self.config.mutation_rate)
        
        # Evaluate offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Add elite individuals back
        offspring.extend(hof.items)
        hof.update(offspring)
        
        return offspring
    
    def evaluate_accuracy(self, individual: Individual, data: Dataset) -> float:
        """Evaluate balanced accuracy score of an individual."""
        trees = [gp.compile(expr, self.pset) for expr in individual]
        predictions = []
        labels = []
        
        for x1, x2, label in data:
            outputs1 = torch.tensor(np.array([tree(*x1) for tree in trees]), 
                                  dtype=torch.float32)
            outputs2 = torch.tensor(np.array([tree(*x2) for tree in trees]), 
                                  dtype=torch.float32)
            
            similarity = F.cosine_similarity(outputs1.unsqueeze(0), outputs2.unsqueeze(0))
            preds = (similarity > 0.5).float()
            predictions.append(preds)
            labels.append(label)
        
        return balanced_accuracy_score(labels, predictions)

class Visualization:
    """Handles visualization of GP trees and feature spaces."""
    
    @staticmethod
    def plot_features(features_x1: np.ndarray, features_x2: np.ndarray, 
                     labels: np.ndarray, output_path: Path) -> None:
        """Plot 2D visualization of learned feature space."""
        pca = PCA(n_components=2)
        features_x1_2d = pca.fit_transform(features_x1)
        features_x2_2d = pca.transform(features_x2)
        
        plt.figure(figsize=(12, 6))
        
        # Plot features with different colors and markers for each class
        for feature_set, marker, prefix in [
            (features_x1_2d, 'o', 'x1'),
            (features_x2_2d, 's', 'x2')
        ]:
            for label, color in [(0, 'blue'), (1, 'red')]:
                mask = labels == label
                plt.scatter(
                    feature_set[mask, 0], feature_set[mask, 1],
                    c=color, marker=marker,
                    label=f'{prefix} (class {label})',
                    alpha=0.5
                )
        
        plt.title("Feature Space Visualization")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / "constructed_features.pdf")
        plt.close()
    
    @staticmethod
    def visualize_tree(tree: gp.PrimitiveTree, output_path: Path, 
                      index: int) -> None:
        """Visualize a single GP tree using pygraphviz."""
        try:
            import pygraphviz as pgv
            
            nodes, edges, labels = gp.graph(tree)
            
            g = pgv.AGraph()
            g.add_nodes_from(nodes)
            g.add_edges_from(edges)
            g.layout(prog="dot")
            
            for i in nodes:
                n = g.get_node(i)
                n.attr["label"] = labels[i]
            
            g.draw(output_path / f"tree_{index}.pdf")
        except ImportError:
            print("Warning: pygraphviz not installed. Skipping tree visualization.")

def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        prog='Contrastive Genetic Programming',
        description='Multi-tree GP for contrastive learning.',
        epilog='Implemented using DEAP framework.'
    )
    
    parser.add_argument('-f', '--file-path', 
                       type=str, 
                       default="checkpoints/embedded-gp.pth",
                       help="Checkpoint filepath (default: checkpoints/embedded-gp.pth)")
    
    parser.add_argument('-d', '--dataset',
                       type=str,
                       default="instance-recognition",
                       help="Dataset name (default: instance-recognition)")
    
    parser.add_argument('-l', '--load',
                       action='store_true',
                       help="Load checkpoint from file")
    
    parser.add_argument('-r', '--run',
                       type=int,
                       default=0,
                       help="Run number for random seed (default: 0)")
    
    parser.add_argument('-o', '--output',
                       type=str,
                       default="logs/results",
                       help="Output logging filepath (default: logs/results)")
    
    parser.add_argument('-p', '--population',
                       type=int,
                       default=100,
                       help="Population size (default: 100)")
    
    parser.add_argument('-g', '--generations',
                       type=int,
                       default=10,
                       help="Number of generations (default: 10)")
    
    parser.add_argument('-mx', '--mutation-rate',
                       type=float,
                       default=0.2,
                       help="Mutation probability (default: 0.2)")
    
    parser.add_argument('-cx', '--crossover-rate',
                       type=float,
                       default=0.8,
                       help="Crossover probability (default: 0.8)")
    
    parser.add_argument('-e', '--elitism',
                       type=int,
                       default=5,
                       help="Number of elite individuals (default: 5)")
    
    parser.add_argument('-td', '--tree-depth',
                       type=int,
                       default=6,
                       help="Maximum tree depth (default: 6)")
    
    parser.add_argument('-nt', '--num-trees',
                       type=int,
                       default=20,
                       help="Number of trees per individual (default: 20)")
    
    return parser

def setup_logging(config: GPConfig) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger(__name__)
    
    # Create output directory if it doesn't exist
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_file = config.output_dir / f"run_{config.run_number}.log"
    handlers = [
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logger

def load_data(config: GPConfig) -> Tuple[Dataset, Dataset]:
    """Load and preprocess dataset."""
    try:
        from util import preprocess_dataset  # type: ignore
        train_loader, val_loader = preprocess_dataset(
            dataset=config.dataset,
            batch_size=64
        )
    except ImportError:
        raise ImportError("Could not import util.preprocess_dataset")
    
    def loader_to_list(loader: torch.utils.data.DataLoader) -> Dataset:
        """Convert DataLoader to list format for GP."""
        data_list = []
        for x1, x2, y in loader:
            for i in range(len(y)):
                data_list.append((
                    x1[i].numpy(),
                    x2[i].numpy(),
                    y[i].item()
                ))
        return data_list
    
    return loader_to_list(train_loader), loader_to_list(val_loader)

def main() -> None:
    """Main execution function."""
    # Parse arguments and setup
    parser = setup_argparse()
    args = parser.parse_args()
    config = GPConfig.from_args(args)
    logger = setup_logging(config)
    
    # Set random seeds for reproducibility
    random.seed(config.run_number)
    np.random.seed(config.run_number)
    torch.manual_seed(config.run_number)
    
    # Create output directories
    for directory in ['checkpoints', 'logs', 'figures']:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset: {config.dataset}")
    train_data, val_data = load_data(config)
    
    # Initialize and train GP model
    logger.info("Initializing GP model")
    gp_model = ContrastiveGP(config, logger)
    
    logger.info("Starting training")
    pop, log, hof = gp_model.train(train_data, val_data)
    
    # Evaluate final model
    best_individual = hof[0]
    final_val_accuracy = gp_model.evaluate_accuracy(best_individual, val_data)
    logger.info(f"Final validation accuracy: {final_val_accuracy:.4f}")
    
    # Visualizations
    logger.info("Generating visualizations")
    
    # Extract and visualize features
    features_x1, features_x2, labels = [], [], []
    for x1, x2, label in train_data:
        trees = [gp.compile(expr, gp_model.pset) for expr in best_individual]
        feat1 = np.array([tree(*x1) for tree in trees])
        feat2 = np.array([tree(*x2) for tree in trees])
        features_x1.append(feat1)
        features_x2.append(feat2)
        labels.append(label)
    
    features_x1 = np.array(features_x1)
    features_x2 = np.array(features_x2)
    labels = np.array(labels)
    
    viz = Visualization()
    viz.plot_features(features_x1, features_x2, labels, Path("figures"))
    
    # Visualize GP trees
    for i, tree in enumerate(best_individual):
        viz.visualize_tree(tree, Path("figures"), i)
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()