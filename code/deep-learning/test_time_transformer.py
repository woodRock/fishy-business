import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler
import math
import numpy as np
from typing import Tuple, Optional

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
        min_confidence_threshold: float = 0.6,
        max_confidence_threshold: float = 0.85,
        temperature: float = 0.8,
        prm_learning_rate: float = 1e-4
    ) -> None:
        super().__init__()

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
        
        # Enhanced Process Reward Model with self-attention
        class PRMBlock(nn.Module):
            def __init__(self, dim, dropout):
                super().__init__()
                self.layer1 = nn.Linear(dim, dim)
                self.layer2 = nn.Linear(dim, dim)
                self.norm1 = nn.LayerNorm(dim)
                self.norm2 = nn.LayerNorm(dim)
                self.activation = nn.GELU()
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                # First residual block
                residual = x
                x = self.norm1(x)
                x = self.layer1(x)
                x = self.activation(x)
                x = self.dropout(x)
                x = residual + x

                # Second residual block
                residual = x
                x = self.norm2(x)
                x = self.layer2(x)
                x = self.activation(x)
                x = self.dropout(x)
                x = residual + x
                return x

        class EnhancedPRM(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_heads, dropout):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, hidden_dim * 2)
                self.dropout = nn.Dropout(dropout)
                self.activation = nn.GELU()
                
                # Residual blocks
                self.blocks = nn.ModuleList([
                    PRMBlock(hidden_dim * 2, dropout) for _ in range(3)
                ])
                
                # Self-attention
                self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads, dropout=dropout, batch_first=True)
                self.norm = nn.LayerNorm(hidden_dim * 2)
                
                # Output layers
                self.out_proj1 = nn.Linear(hidden_dim * 2, hidden_dim)
                self.out_proj2 = nn.Linear(hidden_dim, hidden_dim // 2)
                self.out_proj3 = nn.Linear(hidden_dim // 2, 1)
                
            def forward(self, x):
                # Input projection
                x = self.input_proj(x)
                x = self.activation(x)
                x = self.dropout(x)
                
                # Residual blocks
                for block in self.blocks:
                    x = block(x)
                
                # Add sequence dimension for attention
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                
                # Self-attention
                attn_out, _ = self.attention(x, x, x)
                x = self.norm(x + attn_out)
                
                # Remove sequence dimension
                x = x.squeeze(1)
                
                # Output projection
                x = self.out_proj1(x)
                x = self.activation(x)
                x = self.dropout(x)
                x = self.out_proj2(x)
                x = self.activation(x)
                x = self.out_proj3(x)
                return torch.sigmoid(x)

        # Initialize the enhanced PRM
        self.process_reward_model = EnhancedPRM(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Refinement network
        self.refinement_network = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def to(self, device):
        """Ensures all model components are moved to the specified device."""
        super().to(device)
        self.device = device
        return self

    def calculate_metrics(self, outputs: torch.Tensor, targets: torch.Tensor, task_type: str = 'classification'):
        """Calculate task-specific metrics with soft/continuous values.
        
        Args:
            outputs: Model predictions (batch_size x num_classes)
            targets: Ground truth labels (batch_size x num_classes)
            task_type: Type of task ('classification' or 'regression')
        
        Returns:
            Dictionary containing calculated metrics with proper batch dimensions
        """
        batch_size = outputs.shape[0]
        
        if task_type == 'classification':
            # Get probability distributions
            probs = F.softmax(outputs, dim=-1)
            true_probs = F.softmax(targets, dim=-1)
            
            # Calculate soft accuracy using probability overlap
            accuracy = (probs * true_probs).sum(dim=-1)
            
            # Calculate soft F1 score using continuous predictions
            # Compute precision and recall using soft assignments
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
        else:
            # Calculate relative error scores
            relative_error = torch.abs(outputs - targets) / (torch.abs(targets) + 1e-8)
            mse = torch.exp(-((outputs - targets) ** 2).mean(dim=-1))  # Exponential decay of MSE
            mae = torch.exp(-torch.abs(outputs - targets).mean(dim=-1))  # Exponential decay of MAE
            
            # Calculate R-squared with smoothing
            target_mean = targets.mean(dim=-1, keepdim=True)
            total_variance = ((targets - target_mean) ** 2).sum(dim=-1)
            explained_variance = ((targets - outputs) ** 2).sum(dim=-1)
            r2 = 1 - (explained_variance / (total_variance + 1e-8))
            
            # Normalize R2 score to [0, 1]
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
        if metric_weights is None:
            if task_type == 'classification':
                metric_weights = {
                    'accuracy': 0.3,
                    'f1_score': 0.3,
                    'ce_score': 0.4
                }
            else:
                metric_weights = {
                    'mse_score': 0.3,
                    'mae_score': 0.3,
                    'r2_score': 0.4
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
                
                # Initialize scaler based on first batch metrics if needed
                if scaler is None:
                    scaler = {k: StandardScaler() for k in metrics.keys()}
                    
                # Compute weighted quality score
                quality_score = torch.zeros(batch_x.shape[0], device=self.device)
                for k, v in metrics.items():
                    if k in metric_weights:
                        # Apply dynamic scaling using exponential moving average
                        if len(metrics_history) > 0:
                            v = (v - v.mean()) / (v.std() + 1e-8)
                        quality_score += metric_weights[k] * v
                
                # Normalize quality score to [0, 1]
                quality_score = torch.sigmoid(quality_score).view(-1, 1)
                
                # Get PRM predictions
                predictions = self.process_reward_model(outputs)
                
                # Compute and backpropagate loss with added regularization
                main_loss = criterion(predictions, quality_score)
                consistency_loss = torch.var(predictions) * 0.1  # Encourage diverse predictions
                loss = main_loss + consistency_loss
                
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
                        
                        # Compute validation score
                        batch_score = torch.zeros(batch_x.shape[0], device=self.device)
                        for k, v in metrics.items():
                            if k in metric_weights:
                                batch_score += metric_weights[k] * v
                        
                        val_metrics.append(batch_score.mean().item())
                
                val_score = sum(val_metrics) / len(val_metrics)
                
                # Print detailed metrics
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

        # Load the best model.
        best_model_state = self.process_reward_model.state_dict()

    def forward(self, x: torch.Tensor, use_test_time_compute: bool = False, use_beam_search: bool = True) -> torch.Tensor:
        # Move input to correct device
        x = x.to(self.device)
        initial_output = self._standard_forward(x)
        
        if not use_test_time_compute:
            return initial_output
            
        with torch.no_grad():
            confidence = self.process_reward_model(initial_output).mean().item()
            
        if confidence > self.max_confidence_threshold:
            return initial_output
            
        if confidence > self.min_confidence_threshold:
            actual_iterations = max(2, self.num_iterations // 2)
            actual_beam_width = max(3, self.beam_width // 2)
        else:
            actual_iterations = self.num_iterations
            actual_beam_width = self.beam_width
            
        return self._test_time_forward(x, initial_output, actual_iterations, actual_beam_width, use_beam_search)

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
    
    def genetic_test_time_forward(self, x: torch.Tensor, initial_outputs: torch.Tensor, 
                                num_generations: int, population_size: int) -> torch.Tensor:
        """Forward pass using genetic programming principles."""
        batch_size = x.shape[0]
        device = x.device  # Get the device from input tensor
        
        # Initialize population with mutations of initial output
        population = []
        scores = []
        
        # Create initial population
        population.append(initial_outputs)  # Keep original solution
        scores.append(self.process_reward_model(initial_outputs).mean().item())
        
        # Add mutations of initial solution
        for _ in range(population_size - 1):
            mutation = initial_outputs + torch.randn_like(initial_outputs) * 0.15  # Fine-tuned variation
            population.append(mutation)
            scores.append(self.process_reward_model(mutation).mean().item())
        
        # Track best solution
        best_solution = initial_outputs
        best_score = scores[0]
        
        # Evolution loop
        for generation in range(num_generations):
            new_population = []
            new_scores = []
            
            # Elitism: Keep best solution
            elite_idx = torch.tensor(scores, device=device).argmax()
            new_population.append(population[elite_idx])
            new_scores.append(scores[elite_idx])
            
            # Create new solutions through genetic operations
            while len(new_population) < population_size:
                # Selection: Tournament selection
                tournament_size = 3
                tournament_idx = torch.randperm(len(population), device=device)[:tournament_size]
                parent1_idx = tournament_idx[torch.tensor([scores[i] for i in tournament_idx], device=device).argmax()]
                tournament_idx = torch.randperm(len(population), device=device)[:tournament_size]
                parent2_idx = tournament_idx[torch.tensor([scores[i] for i in tournament_idx], device=device).argmax()]
                
                parent1 = population[parent1_idx.item()]
                parent2 = population[parent2_idx.item()]
                
                # Crossover
                if torch.rand(1, device=device) < 0.8:  # 80% crossover rate
                    alpha = torch.rand(1, device=device)
                    child = alpha * parent1 + (1 - alpha) * parent2
                else:
                    child = parent1.clone()
                
                # Mutation
                if torch.rand(1, device=device) < 0.3:  # Fine-tuned mutation rate
                    mutation_strength = (1 - scores[parent1_idx.item()]) * self.temperature * 0.12
                    child = child + torch.randn_like(child) * mutation_strength
                
                # Refinement step through the network
                refined_input = torch.cat([initial_outputs, child], dim=-1)
                child = self.refinement_network(refined_input)
                
                # Evaluate child
                child_score = self.process_reward_model(child).mean().item()
                
                # Add to new population
                new_population.append(child)
                new_scores.append(child_score)
                
                # Update best solution
                if child_score > best_score:
                    best_score = child_score
                    best_solution = child
            
            # Update population
            population = new_population
            scores = new_scores
            
            # Early stopping if no improvement in last few generations
            if generation > 5 and best_score <= max(scores[-5:]):
                break
        
        return best_solution

    def _test_time_forward(self, x: torch.Tensor, initial_outputs: torch.Tensor, 
                        num_iterations: int, beam_width: int,
                        beam_search: bool) -> torch.Tensor:
        """Forward pass with genetic programming-based test-time compute."""
        if beam_search:
            return self.beam_search_test_time_forward(
                x, 
                initial_outputs,
                num_iterations,
                beam_width,
            )
        else:
            return self.genetic_test_time_forward(
                x,
                initial_outputs,
                num_generations=num_iterations,
                population_size=beam_width * 2
            )

    def beam_search_test_time_forward(self, x: torch.Tensor, initial_outputs: torch.Tensor, 
                                    num_iterations: int, beam_width: int) -> torch.Tensor:
        batch_size = x.shape[0]
        device = x.device  # Determine device

        # Initialize beam with the initial outputs
        best_output = initial_outputs
        best_score = self.process_reward_model(initial_outputs).mean().item()
        beam_candidates = [(initial_outputs, best_score) for _ in range(beam_width)]

        for _ in range(num_iterations):
            new_candidates = []

            for candidate, score in beam_candidates:
                # Refinement of candidate
                refinement_input = torch.cat([initial_outputs, candidate], dim=-1)
                refined = self.refinement_network(refinement_input)

                reward_score = self.process_reward_model(refined)
                current_score = reward_score.mean().item()

                # Update the best result if the score improves
                if current_score > best_score:
                    best_score = current_score
                    best_output = refined

                # Generate perturbed candidates
                for _ in range(beam_width):
                    perturbation_scale = (1 - current_score) * self.temperature * 0.1
                    perturbed = refined + torch.randn_like(refined) * perturbation_scale
                    perturbed_score = self.process_reward_model(perturbed).mean().item()
                    new_candidates.append((perturbed, perturbed_score))

            # Keep the top-scoring candidates for the next iteration
            beam_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        return best_output
