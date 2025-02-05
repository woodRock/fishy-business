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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import random
from sklearn.metrics import balanced_accuracy_score
from typing import List, Tuple, Optional
import copy

from util import prepare_dataset, DataConfig

@dataclass
class GPConfig:
    population_size: int = 200
    gene_length: int = 1028
    projection_dim: int = 256  # Dimension of final projection head output
    generations: int = 100
    mutation_rate: float = 0.3
    tournament_size: int = 8
    hall_of_fame_size: int = 10
    temperature: float = 0.07
    l2_regularization: float = 0.001
    dropout: float = 0.1

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=128, dropout=0.1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

        # Initialize weights properly
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)

        # Enable gradients
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class Individual:
    def __init__(self, gene_length: int, projection_dim: int):
        # Initialize genetic projection matrix
        self.projection = np.random.randn(2080, gene_length) / np.sqrt(gene_length)
        self.projection = F.normalize(torch.from_numpy(self.projection), dim=1).numpy()
        self.projection = self.projection.astype(np.float32)

        # Initialize SimCLR projection head
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.projection_head = ProjectionHead(gene_length, hidden_dim=512, output_dim=projection_dim)
        self.projection_head = self.projection_head.to(device)

        # Ensure parameters require gradients
        for param in self.projection_head.parameters():
            param.requires_grad = True

        self.optimizer = torch.optim.AdamW(self.projection_head.parameters(), lr=0.001)
        self.fitness = None

    def train_projection_head(self, z_i: torch.Tensor, z_j: torch.Tensor, labels: torch.Tensor, temperature: float):
        """Train the projection head for a few steps on the current batch"""
        self.projection_head.train()
        labels = labels.float() if not labels.dtype.is_floating_point else labels

        # Convert labels to target similarities: 1 for positive pairs, -1 for negative
        target_sims = 2 * labels - 1  # Maps 0->-1 and 1->1

        for _ in range(3):
            self.optimizer.zero_grad()

            # Forward pass with proper tensor conversion
            with torch.set_grad_enabled(True):
                # Convert inputs properly
                z_i_tensor = z_i.clone().detach().requires_grad_(True)
                z_j_tensor = z_j.clone().detach().requires_grad_(True)

                p_i = self.projection_head(z_i_tensor)
                p_j = self.projection_head(z_j_tensor)

                p_i = F.normalize(p_i, dim=1)
                p_j = F.normalize(p_j, dim=1)

                # Calculate cosine similarity
                sim = F.cosine_similarity(p_i, p_j)

                # Direct similarity optimization with MSE loss
                pos_weight = (1 - labels).sum() / (labels.sum() + 1e-6)
                weights = torch.where(labels == 1, pos_weight, 1.0)
                loss = (weights * (sim - target_sims) ** 2).mean()

            # Backward pass
            loss.backward()

            # Step optimizer
            self.optimizer.step()

    def project(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        projection_tensor = torch.from_numpy(self.projection).to(x.device)
        z = torch.matmul(x, projection_tensor)
        return z

    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        # First genetic projection
        z = self.project(x)

        # Then through SimCLR projection head
        self.projection_head.train(train)
        if train:
            p = self.projection_head(z)
        else:
            with torch.no_grad():
                p = self.projection_head(z)

        return p

    def mutate(self, mutation_rate: float = 0.3):
        # Mutate genetic projection
        num_groups = 20
        group_size = 2080 // num_groups
        for i in range(num_groups):
            if random.random() < mutation_rate:
                start_idx = i * group_size
                end_idx = start_idx + group_size
                noise = np.random.randn(group_size, self.projection.shape[1]) * 0.3
                self.projection[start_idx:end_idx] += noise

        self.projection = F.normalize(torch.from_numpy(self.projection), dim=1).numpy()
        self.projection = self.projection.astype(np.float32)

        # Mutate projection head with small noise
        with torch.no_grad():
            for param in self.projection_head.parameters():
                noise = torch.randn_like(param) * 0.1 * mutation_rate
                mask = torch.rand_like(param) < mutation_rate
                param.data += noise * mask

class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, features1, features2, labels):
        self.features1 = F.normalize(features1.float(), dim=1)
        self.features2 = F.normalize(features2.float(), dim=1)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.features1[idx], self.features2[idx], self.labels[idx])

def prepare_data(data_loader) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    data = []
    for batch in data_loader:
        sample1, sample2, labels = batch
        sample1 = F.normalize(sample1.float(), dim=1)
        sample2 = F.normalize(sample2.float(), dim=1)
        data.extend(list(zip(sample1, sample2, labels)))
    return data

class ContrastiveGP:
    def __init__(self, config: GPConfig):
        self.config = config
        self.population = [Individual(config.gene_length, config.projection_dim)
                         for _ in range(config.population_size)]
        self.hall_of_fame = []
        self.best_fitness = 0
        self.generations_without_improvement = 0

    def evaluate_individual(self, individual: Individual,
                          data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> float:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_labels = []

        for i in range(0, len(data), 32):
            batch = data[i:i + 32]
            samples1, samples2, labels = zip(*batch)

            with torch.no_grad():
                samples1 = torch.stack(samples1).float().to(device)
                samples2 = torch.stack(samples2).float().to(device)
                labels = torch.stack([label.to(device) for label in labels])
                if labels.dim() > 1:  # If labels are one-hot
                    labels = labels[:, 0]
                labels = labels.float()

                # Get intermediate representations for training
                z_i = individual.project(samples1)
                z_j = individual.project(samples2)

            # Train projection head with gradient tracking
            individual.train_projection_head(z_i, z_j, labels, self.config.temperature)

            with torch.no_grad():
                # Get final projections for evaluation
                p_i = individual.forward(samples1, train=False)
                p_j = individual.forward(samples2, train=False)

                sim = F.cosine_similarity(p_i, p_j)

                # Calculate loss (MSE against target similarities)
                target_sims = 2 * labels - 1
                pos_weight = (1 - labels).sum() / (labels.sum() + 1e-6)
                weights = torch.where(labels == 1, pos_weight, 1.0)
                loss = (weights * (sim - target_sims) ** 2).mean()

                # Use fixed decision boundary at 0
                predictions = (sim > 0).float()

                total_loss += loss.item()
                num_batches += 1

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / num_batches
        balanced_acc = balanced_accuracy_score(all_labels, all_predictions)

        # Add L2 regularization loss - Fixed to use individual's projection head
        l2_loss = sum(p.pow(2).sum() for p in individual.projection_head.parameters())
        l2_loss = self.config.l2_regularization * l2_loss
        avg_loss += l2_loss.item()

        # Higher weight on accuracy vs loss
        return 0.3 * (1.0 / (1.0 + avg_loss)) + 0.7 * balanced_acc

    def evaluate_population(self, data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        for individual in self.population:
            individual.fitness = self.evaluate_individual(individual, data)

    def select_parents(self) -> Tuple[Individual, Individual]:
        def tournament():
            candidates = np.random.choice(self.population, self.config.tournament_size)
            best_candidate = max(candidates, key=lambda x: x.fitness)
            return best_candidate

        parent1 = tournament()
        parent2 = tournament()
        return parent1, parent2

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        child = Individual(self.config.gene_length, self.config.projection_dim)

        # Block crossover for genetic projection
        num_blocks = 10
        block_size = 2080 // num_blocks
        for i in range(num_blocks):
            if random.random() < 0.5:
                start_idx = i * block_size
                end_idx = start_idx + block_size
                child.projection[start_idx:end_idx] = parent1.projection[start_idx:end_idx]
            else:
                start_idx = i * block_size
                end_idx = start_idx + block_size
                child.projection[start_idx:end_idx] = parent2.projection[start_idx:end_idx]

        child.projection = F.normalize(torch.from_numpy(child.projection), dim=1).numpy()
        child.projection = child.projection.astype(np.float32)

        # Interpolate projection head parameters
        alpha = random.random()
        with torch.no_grad():
            for c_param, p1_param, p2_param in zip(
                child.projection_head.parameters(),
                parent1.projection_head.parameters(),
                parent2.projection_head.parameters()
            ):
                c_param.data.copy_(alpha * p1_param.data + (1 - alpha) * p2_param.data)

        return child

    def update_hall_of_fame(self):
        combined = self.hall_of_fame + self.population
        combined.sort(key=lambda x: x.fitness, reverse=True)
        self.hall_of_fame = combined[:self.config.hall_of_fame_size]

    def get_predictions(self, individual: Individual,
                       data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[np.ndarray, np.ndarray]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictions = []
        true_labels = []

        with torch.no_grad():
            for i in range(0, len(data), 32):
                batch = data[i:i + 32]
                samples1, samples2, labels = zip(*batch)

                samples1 = torch.stack(samples1).float().to(device)
                samples2 = torch.stack(samples2).float().to(device)
                labels = torch.stack([label.to(device) for label in labels])
                if labels.dim() > 1:  # If labels are one-hot
                    labels = labels[:, 0]
                true_batch_labels = labels.cpu().numpy()

                p_i = individual.forward(samples1, train=False)
                p_j = individual.forward(samples2, train=False)

                sim = F.cosine_similarity(p_i, p_j)
                pred = (sim > 0).cpu().numpy()  # Fixed decision boundary at 0

                predictions.extend(pred)
                true_labels.extend(true_batch_labels)

        return np.array(predictions), np.array(true_labels)

    def train(self, train_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
              val_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        for gen in range(self.config.generations):
            self.evaluate_population(train_data)
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            self.update_hall_of_fame()

            best_individual = self.population[0]
            train_preds, train_labels = self.get_predictions(best_individual, train_data)
            val_preds, val_labels = self.get_predictions(best_individual, val_data)

            train_acc = balanced_accuracy_score(train_labels, train_preds)
            val_acc = balanced_accuracy_score(val_labels, val_preds)
            current_fitness = best_individual.fitness
            avg_fitness = np.mean([ind.fitness for ind in self.population])

            # Adaptive mutation rate based on improvement
            if current_fitness > self.best_fitness:
                self.best_fitness = current_fitness
                self.generations_without_improvement = 0
                self.config.mutation_rate = max(0.1, self.config.mutation_rate * 0.9)
            else:
                self.generations_without_improvement += 1
                if self.generations_without_improvement > 5:
                    self.config.mutation_rate = min(0.5, self.config.mutation_rate * 1.1)

            print(f"Generation {gen}:")
            print(f"  Best Fitness = {current_fitness:.4f}")
            print(f"  Avg Fitness = {avg_fitness:.4f}")
            print(f"  Train Balanced Accuracy = {train_acc:.4f}")
            print(f"  Val Balanced Accuracy = {val_acc:.4f}")
            print(f"  Mutation Rate = {self.config.mutation_rate:.4f}")

            # Create new population with increased diversity
            new_population = []
            elite_size = max(1, self.config.population_size // 20)
            new_population.extend([copy.deepcopy(ind) for ind in self.population[:elite_size]])

            # Add some random individuals for diversity
            num_random = max(1, self.config.population_size // 10)
            new_population.extend([
                Individual(self.config.gene_length, self.config.projection_dim)
                for _ in range(num_random)
            ])

            # Fill rest with crossover and mutation
            while len(new_population) < self.config.population_size:
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                child.mutate(self.config.mutation_rate)
                new_population.append(child)

            self.population = new_population

        return self.population, None, self.hall_of_fame

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    gp_config = GPConfig()

    print("\nPreparing datasets...")
    train_loader, val_loader = prepare_dataset(DataConfig())
    train_data = prepare_data(train_loader)
    val_data = prepare_data(val_loader)
    print(f"Prepared {len(train_data)} training samples and {len(val_data)} validation samples")

    print("\nInitializing ContrastiveGP model...")
    model = ContrastiveGP(gp_config)
    print("\nStarting training...")
    final_pop, _, hall_of_fame = model.train(train_data, val_data)

    if hall_of_fame:
        best_individual = hall_of_fame[0]
        train_preds, train_labels = model.get_predictions(best_individual, train_data)
        val_preds, val_labels = model.get_predictions(best_individual, val_data)
        final_train_acc = balanced_accuracy_score(train_labels, train_preds)
        final_val_acc = balanced_accuracy_score(val_labels, val_preds)
        print("\nFinal Results:")
        print(f"Best Training Accuracy: {final_train_acc:.4f}")
        print(f"Best Validation Accuracy: {final_val_acc:.4f}")
        print(f"Best Fitness: {best_individual.fitness:.4f}")

if __name__ == "__main__":
    main()