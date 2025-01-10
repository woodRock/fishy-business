import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import math
import numpy as np
from typing import Tuple, Optional, List
from collections import deque

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int) -> None:
        super().__init__()
        assert input_dim % num_heads == 0
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        self.qkv = nn.Linear(input_dim, 3 * input_dim)
        self.fc_out = nn.Linear(input_dim, input_dim)
        
        self.scale = self.head_dim**-0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(batch_size, -1, self.input_dim)
        x = self.fc_out(x)
        return x

class SimplifiedPRM(nn.Module):
    """Simplified Process Reward Model with stable architecture."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TestTimeTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        num_iterations: int = 3,
        beam_width: int = 4,
        min_confidence_threshold: float = 0.5,
        max_confidence_threshold: float = 0.8,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.min_confidence_threshold = min_confidence_threshold
        self.max_confidence_threshold = max_confidence_threshold
        self.temperature = temperature
        self.num_iterations = num_iterations
        self.beam_width = beam_width
        
        # Main transformer components
        self.attention_layers = nn.ModuleList(
            [MultiHeadAttention(input_dim, num_heads) for _ in range(num_layers)]
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(input_dim, output_dim)
        
        # Simplified PRM
        self.process_reward_model = SimplifiedPRM(output_dim, hidden_dim)
        
        # Simple refinement network
        self.refinement_network = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.register_buffer('best_candidates', torch.zeros(3, output_dim))
        self.register_buffer('best_scores', torch.zeros(3))

    def forward(self, x: torch.Tensor, use_test_time_compute: bool = False, use_beam_search: bool = True) -> torch.Tensor:
        """Forward pass through the model."""
        if not hasattr(self, 'device'):
            self.device = next(self.parameters()).device
        x = x.to(self.device)
        
        initial_output = self._standard_forward(x)

        if not use_test_time_compute:
            return initial_output

        # Compute confidence scores for each sample in the batch
        confidence = self.process_reward_model(initial_output).squeeze(1)  # Shape: [batch_size]

        # Apply batch-wise thresholds
        above_max_confidence = confidence > self.max_confidence_threshold
        above_min_confidence = confidence > self.min_confidence_threshold

        # Handle each case based on the confidence thresholds
        results = []
        for i in range(x.shape[0]):  # Iterate over each sample in the batch
            if above_max_confidence[i]:  # High confidence case
                results.append(initial_output[i])
            # elif above_min_confidence[i]:  # Medium confidence case
            else:
                if use_beam_search: # 
                    results.append(
                        self.beam_search_forward(
                            x[i].unsqueeze(0), 
                            initial_output[i].unsqueeze(0), 
                            max(2, self.num_iterations // 2), 
                            max(3, self.beam_width // 2)
                        ).squeeze(0)
                    )
                # else: # Low confidence case
                else:  
                    results.append(
                        self.genetic_forward(
                            x[i].unsqueeze(0), 
                            initial_output[i].unsqueeze(0), 
                            5, # generations 
                            100, # population 
                            10,  # elite size
                        ).squeeze(0)
                    )
        
        return torch.stack(results)  # Combine batch-wise predictions into a tensor
        
    def update_best_candidates(self, candidates: torch.Tensor, scores: torch.Tensor) -> None:
        """Update best candidates maintaining diversity for each sample in the batch."""
        batch_size = candidates.shape[0]

        for i in range(batch_size):
            single_candidate = candidates[i].unsqueeze(0)  # Shape: [1, output_dim]
            single_score = scores[i].item()  # Extract the scalar score

            # Check if this candidate improves the score for the current sample
            if single_score > self.best_scores[i]:
                self.best_candidates[i] = single_candidate
                self.best_scores[i] = single_score
    
    def get_ensemble_prediction(self, batch_size: int) -> torch.Tensor:
        """Get predictions for the entire batch."""
        if self.best_scores.sum() == 0:  # If no valid candidates exist
            return self.best_candidates[:batch_size]  # Return top candidates for the batch size
        
        weights = F.softmax(self.best_scores, dim=0)
        weighted_candidates = self.best_candidates * weights.unsqueeze(1)
        
        # Return weighted candidates for the batch size
        return weighted_candidates[:batch_size]


    def calculate_metrics(self, outputs: torch.Tensor, targets: torch.Tensor, task_type: str = 'classification'):
        """Calculate task-specific metrics with soft/continuous values."""
        batch_size = outputs.shape[0]
        
        if task_type == 'classification':
            # Get probability distributions
            probs = F.softmax(outputs, dim=-1)
            true_probs = F.softmax(targets, dim=-1)
            
            # Calculate soft accuracy using probability overlap
            accuracy = (probs * true_probs).sum(dim=-1)
            
            # Calculate soft F1 score using continuous predictions
            true_positives = (probs * true_probs).sum(dim=-1)
            false_positives = (probs * (1 - true_probs)).sum(dim=-1)
            false_negatives = ((1 - probs) * true_probs).sum(dim=-1)
            
            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            # Calculate cross-entropy as additional metric
            ce_loss = -torch.sum(true_probs * torch.log_softmax(outputs, dim=-1), dim=-1)
            ce_score = 1 - torch.clamp(ce_loss / math.log(outputs.shape[-1]), 0, 1)
            
            return {
                'accuracy': accuracy,
                'f1_score': f1_scores,
                'ce_score': ce_score,
                'combined_score': (accuracy + f1_scores + ce_score) / 3
            }
        else:  # regression
            # Calculate relative error scores
            relative_error = torch.abs(outputs - targets) / (torch.abs(targets) + 1e-8)
            mse = torch.exp(-((outputs - targets) ** 2).mean(dim=-1))
            mae = torch.exp(-torch.abs(outputs - targets).mean(dim=-1))
            
            # Calculate R-squared with smoothing
            target_mean = targets.mean(dim=-1, keepdim=True)
            total_variance = ((targets - target_mean) ** 2).sum(dim=-1)
            explained_variance = ((targets - outputs) ** 2).sum(dim=-1)
            r2 = 1 - (explained_variance / (total_variance + 1e-8))
            r2_normalized = (torch.tanh(r2) + 1) / 2
            
            return {
                'mse_score': mse,
                'mae_score': mae,
                'r2_score': r2_normalized,
                'combined_score': (mse + mae + r2_normalized) / 3
            }

    def train_prm(self, 
                train_loader: DataLoader,
                val_loader: Optional[DataLoader] = None,
                task_type: str = 'classification',
                metric_weights: Optional[dict] = None,
                num_epochs: int = 20,
                patience: int = 5,
                min_delta: float = 1e-4) -> None:
        """Train the Process Reward Model with continuous task-specific metrics."""
        if not hasattr(self, 'device'):
            self.device = next(self.parameters()).device
            
        if metric_weights is None:
            if task_type == 'classification':
                metric_weights = {
                    'accuracy': 0.4,
                    'f1_score': 0.4,
                    'ce_score': 0.2
                }
            else:
                metric_weights = {
                    'mse_score': 0.4,
                    'mae_score': 0.4,
                    'r2_score': 0.2
                }
        
        optimizer = optim.Adam(self.process_reward_model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        best_val_score = float('-inf')
        patience_counter = 0
        best_model_state = None
        
        scaler = None
        metrics_history = []
        
        for epoch in range(num_epochs):
            # Training phase
            self.process_reward_model.train()
            train_loss = 0.0
            epoch_metrics = []
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Generate model outputs
                with torch.no_grad():
                    outputs = self._standard_forward(batch_x)
                
                # Calculate task-specific metrics
                metrics = self.calculate_metrics(outputs, batch_y, task_type)
                epoch_metrics.append({k: v.detach().cpu().numpy().mean() for k, v in metrics.items()})
                
                # Initialize scaler if needed
                if scaler is None:
                    scaler = {k: StandardScaler() for k in metrics.keys()}
                    
                # Compute weighted quality score
                quality_score = torch.zeros(batch_x.shape[0], device=self.device)
                for k, v in metrics.items():
                    if k in metric_weights:
                        if len(metrics_history) > 0:
                            v = (v - v.mean()) / (v.std() + 1e-8)
                        quality_score += metric_weights[k] * v
                
                # Normalize quality score to [0, 1]
                quality_score = torch.sigmoid(quality_score).view(-1, 1)
                
                # Get PRM predictions
                predictions = self.process_reward_model(outputs)
                
                # Compute loss
                loss = criterion(predictions, quality_score)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.process_reward_model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Store metrics history
            metrics_history.append({k: np.mean([m[k] for m in epoch_metrics]) 
                                for k in epoch_metrics[0].keys()})
            
            # Validation phase
            if val_loader is not None:
                self.process_reward_model.eval()
                val_metrics = []
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self._standard_forward(batch_x)
                        metrics = self.calculate_metrics(outputs, batch_y, task_type)
                        
                        batch_score = sum(metric_weights[k] * v.mean() 
                                        for k, v in metrics.items() 
                                        if k in metric_weights)
                        val_metrics.append(batch_score.item())
                
                val_score = np.mean(val_metrics)
                
                print(f'Epoch {epoch+1}/{num_epochs}:')
                print(f'Training Loss: {train_loss/len(train_loader):.4f}')
                print(f'Validation Score: {val_score:.4f}')
                print('Metrics:', {k: f'{v:.4f}' for k, v in metrics_history[-1].items()})
                
                if val_score > best_val_score + min_delta:
                    best_val_score = val_score
                    patience_counter = 0
                    best_model_state = self.process_reward_model.state_dict()
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    if best_model_state is not None:
                        self.process_reward_model.load_state_dict(best_model_state)
                    break
            else:
                print(f'Epoch {epoch+1}/{num_epochs}:')
                print(f'Training Loss: {train_loss/len(train_loader):.4f}')
    
    def _standard_forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        for attention in self.attention_layers:
            residual = x
            x = self.layer_norm1(x)
            x = residual + self.dropout(attention(x))
        
        residual = x
        x = self.layer_norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        
        x = x.mean(dim=1)
        x = self.fc_out(x)
        return x
    
    # def genetic_forward(self, x: torch.Tensor, initial_outputs: torch.Tensor,
    #                     num_generations: int, population_size: int, elite_size: int) -> torch.Tensor:
    #     """Enhanced genetic programming with adaptive mutation, crossover, and diversity preservation.
        
    #     Key improvements:
    #     1. Adaptive mutation rate based on population diversity
    #     2. Tournament selection for better parent selection
    #     3. Crossover operations between good solutions
    #     4. Diversity preservation through niching
    #     5. Local search refinement for elite solutions
    #     6. Dynamic population management
    #     """
    #     batch_size = x.shape[0]
    #     device = x.device
    #     output_dim = initial_outputs.shape[-1]
        
    #     # Initialize population with controlled diversity
    #     population = [
    #         initial_outputs.clone() + torch.randn_like(initial_outputs) * (0.01 * (i / population_size))
    #         for i in range(population_size)
    #     ]
        
    #     # Calculate initial fitness scores
    #     scores = torch.stack([self.process_reward_model(candidate).squeeze(1) for candidate in population])
        
    #     # Track best solution and population statistics
    #     best_solution = population[0].clone()
    #     best_score = scores[0].clone()
    #     stagnation_counter = 0
        
    #     # Calculate initial population diversity
    #     def calculate_diversity(pop):
    #         pop_tensor = torch.stack(pop)
    #         centroid = pop_tensor.mean(dim=0)
    #         distances = torch.norm(pop_tensor - centroid, dim=-1)
    #         return distances.mean().item()
        
    #     def tournament_selection(pop, scores, tournament_size=3):
    #         indices = torch.randperm(len(pop))[:tournament_size]
    #         tournament_scores = scores[indices]
    #         winner_idx = indices[tournament_scores.argmax()]
    #         return pop[winner_idx]
        
    #     def adaptive_crossover(parent1, parent2, diversity):
    #         # Adaptive crossover rate based on population diversity
    #         crossover_rate = 0.7 * (1 - math.exp(-diversity))
    #         mask = torch.rand_like(parent1) < crossover_rate
    #         child = torch.where(mask, parent1, parent2)
    #         return child
        
    #     def adaptive_mutation(individual, diversity, generation):
    #         # Adaptive mutation rate and strength
    #         base_rate = 0.1 * math.exp(-diversity)
    #         mutation_rate = base_rate * (1 - generation / num_generations)
    #         mutation_strength = 0.01 * (1 + diversity)
            
    #         # Apply mutation with varying strengths
    #         mask = torch.rand_like(individual) < mutation_rate
    #         mutation = torch.randn_like(individual) * mutation_strength
    #         return individual + mask * mutation
        
    #     for generation in range(num_generations):
    #         diversity = calculate_diversity(population)
    #         new_population = []
            
    #         # Elitism - preserve and enhance best solutions
    #         sorted_indices = torch.argsort(scores, descending=True)
    #         for i in range(elite_size):
    #             elite = population[sorted_indices[i]].clone()
    #             # Local search refinement for elite solutions
    #             refined = self.refinement_network(torch.cat([initial_outputs, elite], dim=-1))
    #             elite = 0.9 * refined + 0.1 * elite
    #             new_population.append(elite)
            
    #         # Generate new individuals through selection, crossover, and mutation
    #         while len(new_population) < population_size:
    #             # Tournament selection
    #             parent1 = tournament_selection(population, scores)
    #             parent2 = tournament_selection(population, scores)
                
    #             # Adaptive crossover
    #             child = adaptive_crossover(parent1, parent2, diversity)
                
    #             # Adaptive mutation
    #             child = adaptive_mutation(child, diversity, generation)
                
    #             # Refinement with probability
    #             if torch.rand(1).item() < 0.3:
    #                 child = self.refinement_network(torch.cat([initial_outputs, child], dim=-1))
    #                 child = 0.8 * child + 0.2 * parent1
                
    #             new_population.append(child)
            
    #         # Update population and calculate new scores
    #         population = new_population
    #         scores = torch.stack([self.process_reward_model(candidate).squeeze(1) for candidate in population])
            
    #         # Update best solution
    #         current_best_idx = scores.argmax()
    #         if scores[current_best_idx] > best_score:
    #             best_solution = population[current_best_idx].clone()
    #             best_score = scores[current_best_idx].clone()
    #             stagnation_counter = 0
    #         else:
    #             stagnation_counter += 1
            
    #         # Dynamic population management
    #         if stagnation_counter > 5:
    #             # Introduce new diversity while preserving elite solutions
    #             num_new = population_size // 4
    #             for i in range(elite_size, elite_size + num_new):
    #                 population[i] = initial_outputs.clone() + torch.randn_like(initial_outputs) * 0.05
    #             stagnation_counter = 0
        
    #     return best_solution

    def genetic_forward(self, x: torch.Tensor, initial_outputs: torch.Tensor,
                        num_generations: int = 5, population_size: int = 20, elite_size: int = 4) -> torch.Tensor:
        """Genetic algorithm optimized for very fast convergence in just 5 generations.
        Uses aggressive mutation scales and heavy refinement to quickly improve solutions."""
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize population with more aggressive mutations
        population = [
            initial_outputs.clone() + torch.randn_like(initial_outputs) * (0.05 * (1 + i/5))
            for i in range(population_size)
        ]
        
        # Add the initial output to ensure we don't regress
        population[0] = initial_outputs.clone()
        
        # Initial scoring
        scores = [self.process_reward_model(candidate).squeeze(1) for candidate in population]
        best_score = max(scores)
        best_solution = population[scores.index(max(scores))].clone()
        
        # Very quick evolution
        for generation in range(num_generations):
            # Sort and select elite
            sorted_pairs = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
            elite_population = [ind for ind, score in sorted_pairs[:elite_size]]
            
            # Start new population with elites
            new_population = elite_population.copy()
            
            # Aggressive mutation and refinement for remaining population
            while len(new_population) < population_size:
                # Base mutation from best solution with aggressive scale
                base = elite_population[0].clone()
                mutation_scale = 0.05 * (1 - generation/num_generations)  # Decreasing scale
                mutation = torch.randn_like(initial_outputs) * mutation_scale
                
                # Create mutated version
                mutated = base + mutation
                
                # Heavy refinement
                refined = self.refinement_network(torch.cat([initial_outputs, mutated], dim=-1))
                refined = 0.7 * refined + 0.3 * mutated  # Stronger refinement influence
                
                new_population.append(refined)
            
            population = new_population
            scores = [self.process_reward_model(candidate).squeeze(1) for candidate in population]
            
            # Update best solution if we found better
            current_best = max(scores)
            if current_best > best_score:
                best_score = current_best
                best_solution = population[scores.index(current_best)].clone()
        
        return best_solution

    # def beam_search_forward(self, x: torch.Tensor, initial_outputs: torch.Tensor,
    #                         num_iterations: int, beam_width: int) -> torch.Tensor:
    #     """Beam search that preserves batch-wise predictions."""
    #     batch_size = x.shape[0]
    #     device = x.device

    #     # Initialize beams for each sample in the batch
    #     beams = [(initial_outputs, self.process_reward_model(initial_outputs).squeeze(1))]  # List of (candidate, score) tuples

    #     # Create tensor to hold batch-wise candidates
    #     candidates = initial_outputs.clone()  # Shape: [batch_size, output_dim]

    #     for iteration in range(num_iterations):
    #         new_candidates = []
    #         for i in range(batch_size):
    #             sample_candidates = []
    #             for _ in range(beam_width):
    #                 perturbation = candidates[i] + torch.randn_like(candidates[i]) * 0.01 * self.temperature
    #                 refined = self.refinement_network(torch.cat([candidates[i], perturbation], dim=-1))
    #                 refined = 0.9 * refined + 0.1 * perturbation
    #                 score = self.process_reward_model(refined).item()
    #                 sample_candidates.append((refined, score))
    #             sample_candidates = sorted(sample_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    #             new_candidates.append(sample_candidates[0][0])  # Select top candidate per batch sample

    #         candidates = torch.stack(new_candidates)  # Maintain batch structure

    #     return candidates  # Return batch-aligned predictions

    def beam_search_forward(self, x: torch.Tensor, initial_outputs: torch.Tensor,
                            num_iterations: int, beam_width: int) -> torch.Tensor:
        """Enhanced beam search with adaptive width, diversity preservation, and smart exploration.
        
        Key improvements:
        1. Adaptive beam width based on confidence scores
        2. Diversity preservation through penalty terms
        3. Temperature-based exploration schedule
        4. Momentum-based updates
        5. Local refinement for promising candidates
        6. Dynamic temperature adjustment
        """
        batch_size = x.shape[0]
        device = x.device
        output_dim = initial_outputs.shape[-1]
        
        # Initialize momentum buffers
        momentum = torch.zeros_like(initial_outputs)
        best_score_history = []
        
        # Initialize beam states for each sample
        beam_states = {
            'candidates': initial_outputs.unsqueeze(1).repeat(1, beam_width, 1),  # [batch_size, beam_width, output_dim]
            'scores': torch.zeros(batch_size, beam_width, device=device),  # [batch_size, beam_width]
            'diversity_history': [],  # Track diversity metrics
            'temperature': torch.ones(batch_size, device=device) * self.temperature  # Adaptive temperature
        }
        
        # Calculate initial scores
        with torch.no_grad():
            initial_scores = self.process_reward_model(initial_outputs).squeeze(-1)
            beam_states['scores'][:, 0] = initial_scores
        
        def calculate_diversity_penalty(candidates, current_candidate):
            """Calculate diversity penalty based on cosine similarity.
            
            Args:
                candidates: Shape [beam_width, output_dim]
                current_candidate: Shape [output_dim]
            Returns:
                Scalar diversity penalty
            """
            # Reshape tensors to proper dimensions for cosine similarity
            current_candidate = current_candidate.unsqueeze(0)  # [1, output_dim]
            
            # Calculate pairwise cosine similarity
            similarities = F.cosine_similarity(
                current_candidate.unsqueeze(0),  # [1, 1, output_dim]
                candidates.unsqueeze(1),         # [beam_width, 1, output_dim]
                dim=2
            )
            
            # Average similarity across all candidates
            return similarities.mean()
        
        def adaptive_beam_width(scores, iteration):
            """Adjust beam width based on score distribution."""
            score_std = scores.std(dim=-1, keepdim=True)
            conf_factor = torch.sigmoid(score_std * 10)  # Scale factor based on score spread
            new_width = torch.ceil(beam_width * (1 + conf_factor)).long()
            return torch.clamp(new_width, min=beam_width // 2, max=beam_width * 2)
        
        def update_temperature(temp, score_history, iteration):
            """Dynamic temperature adjustment based on improvement rate."""
            if len(score_history) > 1:
                improvement = (score_history[-1] - score_history[-2]) / score_history[-2]
                if improvement < 0.01:  # Small improvement
                    temp = temp * 1.1  # Increase exploration
                else:
                    temp = temp * 0.9  # Decrease exploration
            return torch.clamp(temp, min=0.1, max=2.0)
        
        for iteration in range(num_iterations):
            all_candidates = []
            all_scores = []
            
            # Adaptive exploration scale based on iteration progress
            exploration_scale = self.temperature * (1 - iteration / num_iterations) ** 0.5
            
            for batch_idx in range(batch_size):
                current_candidates = beam_states['candidates'][batch_idx]
                current_scores = beam_states['scores'][batch_idx]
                
                # Generate proposals using momentum and adaptive noise
                momentum_scale = 0.9 ** iteration  # Decay momentum influence
                noise_scale = exploration_scale * (1 + current_scores.std().item())
                
                # Generate new candidates with momentum
                new_candidates = []
                new_scores = []
                
                # Adaptive beam width for this batch sample
                current_beam_width = adaptive_beam_width(current_scores.unsqueeze(0), iteration)[0].item()
                
                for _ in range(int(current_beam_width)):
                    # Generate perturbation with momentum
                    perturbation = (
                        torch.randn_like(current_candidates[0]) * noise_scale +
                        momentum[batch_idx] * momentum_scale
                    )
                    
                    # Create new candidate
                    candidate = current_candidates[0] + perturbation
                    
                    # Apply refinement network
                    refined = self.refinement_network(
                        torch.cat([initial_outputs[batch_idx], candidate], dim=-1)
                    )
                    
                    # Interpolate between refined and original with adaptive weight
                    alpha = 0.8 * (1 - iteration / num_iterations)
                    candidate = alpha * refined + (1 - alpha) * candidate
                    
                    # Calculate score with diversity penalty
                    base_score = self.process_reward_model(candidate.unsqueeze(0)).squeeze()
                    if len(new_candidates) > 0:  # Only calculate diversity penalty if we have previous candidates
                        diversity_penalty = calculate_diversity_penalty(torch.stack(new_candidates), candidate)
                        score = base_score - 0.1 * diversity_penalty
                    else:
                        score = base_score
                    
                    new_candidates.append(candidate)
                    new_scores.append(score)
                
                # Combine current and new candidates
                combined_candidates = torch.cat([current_candidates, torch.stack(new_candidates)])
                combined_scores = torch.cat([current_scores, torch.stack(new_scores)])
                
                # Select top candidates while maintaining diversity
                _, indices = combined_scores.sort(descending=True)
                selected_candidates = combined_candidates[indices[:beam_width]]
                selected_scores = combined_scores[indices[:beam_width]]
                
                # Update momentum using best candidate
                momentum[batch_idx] = 0.9 * momentum[batch_idx] + 0.1 * (
                    selected_candidates[0] - current_candidates[0]
                )
                
                all_candidates.append(selected_candidates)
                all_scores.append(selected_scores)
            
            # Update beam states
            beam_states['candidates'] = torch.stack(all_candidates)
            beam_states['scores'] = torch.stack(all_scores)
            
            # Update temperature
            best_score_history.append(beam_states['scores'].max().item())
            beam_states['temperature'] = update_temperature(
                beam_states['temperature'],
                best_score_history,
                iteration
            )
            
            # Early stopping if we've converged
            if len(best_score_history) > 2 and \
            abs(best_score_history[-1] - best_score_history[-2]) < 1e-4:
                break
        
        # Return best candidates for each batch sample
        best_indices = beam_states['scores'].argmax(dim=1)
        best_candidates = torch.stack([
            beam_states['candidates'][i, idx] 
            for i, idx in enumerate(best_indices)
        ])
        
        return best_candidates