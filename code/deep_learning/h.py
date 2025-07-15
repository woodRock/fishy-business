"""
Enhancing Transformer Performance with Test-Time Reasoning Heuristics: A Study on Intermediate Step Optimization

A transformer with test-time reasoning heuristics for intermediate steps.
It uses beam search to generate and rank intermediate steps at inference time.
Heuristics are used to evaluate the quality of intermediate steps and guide reasoning

Results:

## Species

Vanilla Validation Accuracy: Mean = 0.969, Std = 0.037
Beam Search Validation Accuracy: Mean = 0.97772, Std = 0.03070
Genetic Algorithm Validation Accuracy: Mean = 0.98078, Std = 0.02862

## Part

Vanilla Validation Accuracy: Mean = 0.558, Std = 0.091
Beam Search Validation Accuracy: Mean = 0.67593, Std = 0.08374
Genetic Algorithm Validation Accuracy: Mean = 0.67222, Std = 0.07638

## Cross-species

Vanilla Validation Accuracy: Mean = 0.883, Std = 0.072
Beam Search Validation Accuracy: Mean = 0.88568, Std = 0.04620
Genetic Search Validation Accuracy: Mean = 0.88413, Std = 0.04853

## Oil

Vanilla Validation Accuracy: Mean = 0.422, Std = 0.074
Beam Search Validation Accuracy: Mean = 0.45404, Std = 0.06590
Genetic Algorithm Validation Accuracy: Mean = 0.45533, Std = 0.06860

References:
1.  Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.
    N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in
    Neural Information Processing Systems, 30.
2.  Snell, C., Lee, J., Xu, K., & Kumar, A. (2024). Scaling llm test-time
    compute optimally can be more effective than scaling model parameters.
    arXiv preprint arXiv:2408.03314.
3.  Furcy, D., & Koenig, S. (2005, July). Limited discrepancy beam search.
    In IJCAI (pp. 125-131).
4.  Freitag, M., & Al-Onaizan, Y. (2017). Beam search strategies for neural
    machine translation. arXiv preprint arXiv:1702.01806.
5.  Graves, A. (2012). Sequence transduction with recurrent neural networks.
    arXiv preprint arXiv:1211.3711.
6.  Sutskever, I. (2014). Sequence to Sequence Learning with Neural Net-
    works. arXiv preprint arXiv:1409.3215.

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Assuming `create_data_module` is available in `util.py`
from util import create_data_module


class ReasoningHeuristics:
    """Heuristics to evaluate transformer intermediate steps"""

    @staticmethod
    def feature_attribution(
        intermediate: torch.Tensor,
        zero_baseline: torch.Tensor,
        layer_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Measure feature importance by comparing to zero baseline.
        Higher scores mean features contribute more to the difference from baseline.

        Args:
            intermediate: Current intermediate representation
            zero_baseline: Zero tensor of same shape as layer_output
            layer_output: Output from current layer
        """
        # Ensure intermediate tensor requires gradients
        intermediate = intermediate.requires_grad_(True)
        intermediate.retain_grad()

        # Compute distance from zero baseline
        distance = F.mse_loss(layer_output, zero_baseline)

        # Backpropagate
        distance.backward(retain_graph=True)

        if intermediate.grad is None:
            raise RuntimeError("Gradients not available")

        # Feature importance = gradient * activation (comparing to zero baseline)
        importance = torch.abs(intermediate.grad * intermediate)
        return importance.mean(dim=-1)

    @staticmethod
    def representation_clarity(intermediate: torch.Tensor) -> torch.Tensor:
        """
        Measure how clearly separated different patterns are.
        Higher score means features are more distinctly organized.
        Uses cosine similarity between different feature dimensions.
        """
        # Normalize features
        normed = F.normalize(intermediate, dim=-1)

        # Compute pairwise similarities
        similarity = torch.matmul(normed, normed.transpose(-1, -2))

        # Clear separation = low off-diagonal similarity
        mask = ~torch.eye(similarity.shape[-1], dtype=bool, device=similarity.device)
        clarity = -torch.mean(torch.abs(similarity[mask]), dim=-1)
        return clarity

    @staticmethod
    def information_gain(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        """
        Measure how much new information is gained in this step.
        Higher score means step provides more novel information.
        """
        # Compute change in representation
        delta = current - previous

        # Information gain = magnitude of change relative to previous state
        gain = torch.norm(delta, dim=-1) / (torch.norm(previous, dim=-1) + 1e-6)
        return gain

    @staticmethod
    def decision_confidence(layer_output: torch.Tensor) -> torch.Tensor:
        """
        Measure how confident/certain the predictions are.
        Higher score means more decisive predictions.
        """
        probs = F.softmax(layer_output, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        confidence = 1.0 - (entropy / torch.log(torch.tensor(probs.size(-1))))
        return confidence

    @staticmethod
    def feature_interactions(intermediate: torch.Tensor) -> torch.Tensor:
        """
        Measure how features interact/influence each other.
        Higher score means more complex feature relationships.
        """
        # Compute feature correlation matrix
        corr = torch.corrcoef(intermediate.transpose(-1, -2))

        # Measure strength of off-diagonal correlations
        mask = ~torch.eye(corr.shape[-1], dtype=bool, device=corr.device)
        interaction = torch.mean(torch.abs(corr[mask]), dim=-1)
        return interaction

    @staticmethod
    def reasoning_progress(
        current: torch.Tensor, target: torch.Tensor, num_steps: int
    ) -> torch.Tensor:
        """
        Measure progress toward target representation.
        Higher score means closer to final target state.
        """
        # Compute distance to target
        distance = torch.norm(current - target, dim=-1)
        max_distance = torch.norm(target, dim=-1)

        # Progress = how much of the distance has been covered
        progress = 1.0 - (distance / (max_distance + 1e-6))
        return progress


class TransformerWithHeuristics(nn.Module):
    """Transformer with beam search for inference only"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        num_classes: int,
        beam_width: int = 3,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.heuristics = ReasoningHeuristics()
        self.beam_width = beam_width  # Number of paths to explore at each step

    def beam_search(self, x: torch.Tensor):
        """
        Perform beam search to generate and rank intermediate steps at inference time.
        Returns the best output and metrics.
        """
        # Enable gradients temporarily for feature attribution
        with torch.enable_grad():
            # Initial projection
            current = self.input_proj(x)
            previous = current

            # Track intermediate representations and their scores
            beam = [(current, 0.0, [])]  # (state, cumulative_score, metrics)

            # Process through transformer layers
            for layer_idx, layer in enumerate(self.layers):
                new_beam = []

                for state, cumulative_score, metrics in beam:
                    # Generate new states by applying the transformer layer
                    new_state = layer(state.unsqueeze(1)).squeeze(1)

                    # Compute heuristic scores for the new state
                    layer_output = self.classifier(new_state)

                    feature_attribution = self.heuristics.feature_attribution(
                        new_state, torch.zeros_like(layer_output), layer_output
                    ).mean()
                    clarity = self.heuristics.representation_clarity(new_state).mean()
                    info_gain = self.heuristics.information_gain(
                        new_state, state
                    ).mean()
                    confidence = self.heuristics.decision_confidence(
                        layer_output
                    ).mean()
                    interactions = self.heuristics.feature_interactions(
                        new_state
                    ).mean()
                    progress = self.heuristics.reasoning_progress(
                        new_state, state, len(self.layers)
                    ).mean()

                    # Combine heuristic scores into a single quality score
                    quality_score = (
                        feature_attribution
                        + clarity
                        + info_gain
                        + confidence
                        + interactions
                        + progress
                    ) / 6  # Average of all heuristics

                    # Update cumulative score
                    new_cumulative_score = cumulative_score + quality_score.item()

                    # Update metrics
                    new_metrics = metrics + [
                        {
                            "layer": layer_idx,
                            "feature_attribution": feature_attribution.item(),
                            "clarity": clarity.item(),
                            "info_gain": info_gain.item(),
                            "confidence": confidence.item(),
                            "interactions": interactions.item(),
                            "progress": progress.item(),
                            "quality_score": quality_score.item(),
                        }
                    ]

                    # Add to new beam
                    new_beam.append((new_state, new_cumulative_score, new_metrics))

                # Rank the new beam by cumulative score and keep top-k
                new_beam.sort(key=lambda x: x[1], reverse=True)
                beam = new_beam[: self.beam_width]

            # Select the best state and metrics
            best_state, best_score, best_metrics = beam[0]

            # Final classification
            output = self.classifier(best_state)
            return (
                output,
                best_metrics,
                best_score / len(self.layers),
            )  # Normalize by number of layers

    def genetic_search(self, x: torch.Tensor):
        """
        Perform genetic algorithm search during transformer inference.
        Returns output logits, metrics for each layer, and average quality score.
        """
        # Initialize parameters
        population_size = 20
        num_generations = 10
        mutation_rate = 0.1
        elite_size = 2

        def mutate(state):
            """Apply random mutations to the state"""
            mutation_mask = torch.rand_like(state, device=state.device) < mutation_rate
            mutations = torch.randn_like(state, device=state.device) * 0.1
            return torch.where(mutation_mask, state + mutations, state)

        def crossover(parent1, parent2):
            """Perform crossover between two parent states"""
            mask = torch.rand_like(parent1, device=parent1.device) > 0.5
            child1 = torch.where(mask, parent1, parent2)
            child2 = torch.where(mask, parent2, parent1)
            return child1, child2

        def compute_heuristics(state, previous_state, logits, layer_idx):
            """Compute heuristics without requiring gradients"""
            with torch.no_grad():
                clarity = self.heuristics.representation_clarity(state).mean()
                info_gain = self.heuristics.information_gain(
                    state, previous_state
                ).mean()
                confidence = self.heuristics.decision_confidence(logits).mean()
                interactions = self.heuristics.feature_interactions(state).mean()
                progress = self.heuristics.reasoning_progress(
                    state, previous_state, len(self.layers)
                ).mean()

                return clarity, info_gain, confidence, interactions, progress

        def evaluate_fitness(state, previous_state, layer_idx):
            """Calculate fitness score using heuristics"""
            try:
                # Create computation graph
                with torch.enable_grad():
                    state = state.detach().requires_grad_()

                    # Forward pass
                    intermediate = layer(state.unsqueeze(1)).squeeze(1)
                    logits = self.classifier(intermediate)

                    # Compute pseudo-labels
                    pseudo_labels = torch.argmax(logits.detach(), dim=1)

                    # Compute cross entropy loss
                    loss = F.cross_entropy(logits, pseudo_labels)

                    # Compute gradients
                    loss.backward()

                    if state.grad is None:
                        return -float("inf"), None

                    # Feature attribution
                    feature_attr = torch.abs(state.grad * state).mean()

                    # Compute other heuristics
                    clarity, info_gain, confidence, interactions, progress = (
                        compute_heuristics(state, previous_state, logits, layer_idx)
                    )

                    # Combine heuristics
                    fitness = (
                        feature_attr
                        + clarity
                        + info_gain
                        + confidence
                        + interactions
                        + progress
                    ) / 6

                    metrics = {
                        "layer": layer_idx,
                        "feature_attribution": feature_attr.item(),
                        "clarity": clarity.item(),
                        "info_gain": info_gain.item(),
                        "confidence": confidence.item(),
                        "interactions": interactions.item(),
                        "progress": progress.item(),
                        "quality_score": fitness.item(),
                    }

                    return fitness.item(), metrics

            except Exception as e:
                print(f"Error in fitness evaluation: {str(e)}")
                return -float("inf"), None

            finally:
                # Clean up gradients
                if state.grad is not None:
                    state.grad.zero_()

        def tournament_select(population, fitness_scores):
            """Select parent using tournament selection"""
            idx = np.random.choice(len(population), 3, replace=False)
            tournament_fitness = [fitness_scores[i] for i in idx]
            winner_idx = idx[np.argmax(tournament_fitness)]
            return population[winner_idx]

        # Move input to correct device and initial projection
        device = next(self.parameters()).device
        x = x.to(device)
        current = self.input_proj(x)
        all_metrics = []

        # Process through transformer layers
        for layer_idx, layer in enumerate(self.layers):
            # Initialize population with mutations of current state
            population = [mutate(current.clone()) for _ in range(population_size)]
            best_fitness = float("-inf")
            best_state = None
            best_state_metrics = None

            # Evolution loop
            for generation in range(num_generations):
                fitness_scores = []
                generation_metrics = []

                # Evaluate fitness for each individual
                for state in population:
                    # Calculate fitness and metrics
                    fitness, metrics = evaluate_fitness(state, current, layer_idx)

                    if metrics is not None:
                        fitness_scores.append(fitness)
                        generation_metrics.append(metrics)

                        # Track best individual
                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_state = state.clone().detach()
                            best_state_metrics = metrics

                if not fitness_scores:
                    continue

                # Sort population by fitness
                sorted_indices = np.argsort(fitness_scores)[::-1]
                population = [population[i].clone().detach() for i in sorted_indices]
                fitness_scores = [fitness_scores[i] for i in sorted_indices]

                # Keep elite individuals
                new_population = population[:elite_size]

                # Create offspring
                while len(new_population) < population_size:
                    # Select parents using tournament selection
                    parent1 = tournament_select(population, fitness_scores)
                    parent2 = tournament_select(population, fitness_scores)

                    # Create and mutate offspring
                    child1, child2 = crossover(parent1, parent2)
                    child1, child2 = mutate(child1), mutate(child2)

                    new_population.extend([child1, child2])

                # Update population
                population = new_population[:population_size]

            # Process layer output
            if best_state is not None:
                # Use best found solution
                with torch.no_grad():
                    current = layer(best_state.unsqueeze(1)).squeeze(1)
                all_metrics.append(best_state_metrics)
            else:
                # Fallback to standard forward pass
                with torch.no_grad():
                    current = layer(current.unsqueeze(1)).squeeze(1)
                # Add placeholder metrics
                all_metrics.append(
                    {
                        "layer": layer_idx,
                        "feature_attribution": 0.0,
                        "clarity": 0.0,
                        "info_gain": 0.0,
                        "confidence": 0.0,
                        "interactions": 0.0,
                        "progress": 0.0,
                        "quality_score": 0.0,
                    }
                )

        # Final classification
        with torch.no_grad():
            output = self.classifier(current)
            avg_quality = sum(m["quality_score"] for m in all_metrics) / len(
                self.layers
            )

        return output, all_metrics, avg_quality

    def forward(self, x: torch.Tensor, label: torch.Tensor = None):
        """
        Forward pass:
        - During training: Standard forward pass.
        - During inference: Beam search.
        """
        if self.training:
            # Standard forward pass during training
            current = self.input_proj(x)
            for layer in self.layers:
                current = layer(current.unsqueeze(1)).squeeze(1)
            return self.classifier(current)
        else:
            # Beam search during inference
            return self.beam_search(x)
            # return self.genetic_search(x)


def load_data(dataset: str = "species") -> (TensorDataset, list):
    """
    Load the specified dataset and scale the features using StandardScaler.

    Args:
        dataset (str): The name of the dataset to load. One of ["species", "part", "oil", "cross-species"].

    Returns:
        scaled_dataset (TensorDataset): The scaled dataset.
        targets (list): The target labels.
    """
    data_module = create_data_module(
        dataset_name=dataset,
        batch_size=32,
    )

    data_loader, _ = data_module.setup()
    dataset = data_loader.dataset
    features = torch.stack([sample[0] for sample in dataset])
    labels = torch.stack([sample[1] for sample in dataset])

    features_np = features.numpy()
    scaler = StandardScaler()
    scaled_features_np = scaler.fit_transform(features_np)
    scaled_features = torch.tensor(scaled_features_np, dtype=torch.float32)

    scaled_dataset = TensorDataset(scaled_features, labels)
    targets = [sample[1].argmax(dim=0) for sample in dataset]

    return scaled_dataset, targets


def train_model(
    model: TransformerWithHeuristics,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> nn.Module:
    """Train the transformer model with quality score in loss function"""

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5)

    # Move model to device
    model.to(device)

    best_val_accuracy = -float("inf")
    best_model = None

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(x, y)

            # Compute classification loss
            classification_loss = criterion(logits, torch.argmax(y, dim=1))

            # Combine classification loss and quality score
            loss = classification_loss

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        # Print average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                # Forward pass
                logits, metrics, quality_score = model(x)  # Beam search is used here
                classification_loss = criterion(logits, torch.argmax(y, dim=1))

                # Accumulate validation loss
                val_loss += classification_loss.item()

                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                total += y.size(0)
                correct += (predicted == torch.argmax(y, dim=1)).sum().item()

        # Print validation results
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        print(
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict()

    if best_model is not None:
        # Early stopping. Load the best model
        model.load_state_dict(best_model)

    return model, best_val_accuracy


def analyze_reasoning(
    model: TransformerWithHeuristics, x: torch.Tensor, y: torch.Tensor
):
    """Analyze reasoning quality of transformer layers"""

    # Move input tensors to the same device as the model
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)

    # Get predictions and metrics
    _, metrics, _ = model(x, y)

    print("\nReasoning Analysis by Layer:")
    print("-" * 50)

    for layer_metrics in metrics:
        layer = layer_metrics["layer"]
        print(f"\nLayer {layer}:")
        print(f"  Feature Attribution: {layer_metrics['feature_attribution']:.3f}")
        print(f"  Representation Clarity: {layer_metrics['clarity']:.3f}")
        print(f"  Information Gain: {layer_metrics['info_gain']:.3f}")
        print(f"  Decision Confidence: {layer_metrics['confidence']:.3f}")
        print(f"  Feature Interactions: {layer_metrics['interactions']:.3f}")
        print(f"  Reasoning Progress: {layer_metrics['progress']:.3f}")
        print(f"  Quality Score: {layer_metrics['quality_score']:.3f}")


def main():
    # Set dataset and output dimensions
    dataset = "species"
    n_classes = {"species": 2, "part": 7, "oil": 7, "cross-species": 3}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    scaled_dataset, targets = load_data(dataset=dataset)

    # Initialize lists to store results
    all_pretrain_results = []  # Store results from all 30 runs

    # Perform 30 independent runs
    num_runs = 30
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")

        # Set a different random seed for each run
        torch.manual_seed(run)
        np.random.seed(run)

        # Initialize lists to store results for this run
        pretrain_results = []

        # Perform Stratified Cross-Validation
        n_splits = (
            3 if dataset == "part" else 5
        )  # Not enough classes for 5-fold cross-validation on the "part" dataset.
        kfold = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=run
        )  # Use run as the random seed

        for fold, (train_idx, val_idx) in enumerate(
            kfold.split(scaled_dataset, targets)
        ):
            print(f"\nFold {fold + 1}/{n_splits}")

            # Create train/val datasets for this fold
            train_dataset = Subset(scaled_dataset, train_idx)
            val_dataset = Subset(scaled_dataset, val_idx)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            # Initialize model
            model = TransformerWithHeuristics(
                input_dim=2080,
                hidden_dim=256,
                num_layers=3,
                num_heads=4,
                num_classes=n_classes[dataset],
            )

            # Train the model
            model, best_val_accuracy = train_model(
                model,
                train_loader,
                val_loader,
                num_epochs=50,
                learning_rate=0.001,
                device=device,
            )

            # Report the best validation accuracy for this fold
            pretrain_results.append(best_val_accuracy)
            print(f"Pretrain Validation Accuracy: {best_val_accuracy:.3f}")

        # Store results for this run
        all_pretrain_results.append(pretrain_results)
        print(f"\nRun {run + 1} Results:")
        print(
            f"Pretrain Validation Accuracy: Mean = {np.mean(pretrain_results):.3f}, Std = {np.std(pretrain_results):.3f}"
        )

    # Compute overall mean and standard deviation across all runs
    all_pretrain_results = np.array(all_pretrain_results)
    overall_mean = np.mean(all_pretrain_results)
    overall_std = np.std(all_pretrain_results)

    print("\nFinal Results Across All Runs:")
    print(
        f"Pretrain Validation Accuracy: Mean = {overall_mean:.3f}, Std = {overall_std:.3f}"
    )


if __name__ == "__main__":
    main()
