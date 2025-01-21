import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import random

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
        
        # Store attention pattern for exploration
        self.last_attention = attn
        
        x = (attn @ v).transpose(1, 2).reshape(batch_size, -1, self.input_dim)
        x = self.fc_out(x)
        return x

class TestTimeTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 2080,
        hidden_dim: int = 512,
        output_dim: int = 7,
        num_classes: int = None,  # Number of output classes
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        num_mc_rollouts: int = 16,
        num_iterations: int = 5,
        temperature: float = 0.8,
        min_confidence: float = 0.2,
        max_confidence: float = 0.95
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_mc_rollouts = num_mc_rollouts
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        
        # Spectral feature embedding
        self.spectral_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Transformer layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers * 2)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        
        # Process Reward Model components
        self.prm_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def _process_step(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Process input through a single transformer layer."""
        # Apply attention
        norm1 = self.layer_norms[layer_idx * 2](x)
        attn = self.attention_layers[layer_idx](norm1)
        x = x + self.dropout(attn)
        
        # Apply feed-forward
        norm2 = self.layer_norms[layer_idx * 2 + 1](x)
        ff = self.feed_forward(norm2)
        x = x + self.dropout(ff)
        
        return x

    def forward_pass(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Complete forward pass with stored intermediate states."""
        # Embed spectral features
        x = self.spectral_embedding(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        features = x
        intermediate_states = []
        attention_patterns = []
        
        # Process through transformer layers
        for i in range(len(self.attention_layers)):
            features = self._process_step(features, i)
            
            # Store intermediate state
            intermediate_states.append(features)
            
            # Store attention pattern
            if hasattr(self.attention_layers[i], 'last_attention'):
                attention_patterns.append(
                    self.attention_layers[i].last_attention
                )
        
        # Final output
        features = features.mean(dim=1)
        logits = self.fc_out(features)
        
        return {
            'logits': logits,
            'intermediate_states': intermediate_states,
            'attention_patterns': attention_patterns,
            'final_features': features
        }

    def estimate_step_confidence(
        self,
        states: List[torch.Tensor],
        logits: torch.Tensor
    ) -> torch.Tensor:
        """Estimate confidence using intermediate states and logits."""
        # Encode states
        encoded_states = [self.prm_encoder(s.mean(dim=1)) for s in states]
        state_features = torch.stack(encoded_states).mean(dim=0)
        
        # Get prediction info
        probs = F.softmax(logits, dim=-1)
        max_probs, pred_classes = probs.max(dim=-1)
        
        # Get sorted probabilities for margin
        sorted_probs, _ = probs.sort(dim=-1, descending=True)
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        
        # Factor in prediction margin
        margin_factor = margin.unsqueeze(-1)
        confidence = self.confidence_net(state_features) * (0.5 + 0.5 * margin_factor)
        
        # Clamp confidence
        confidence = torch.clamp(confidence, self.min_confidence, self.max_confidence)
        confidence = torch.min(confidence, max_probs.unsqueeze(-1))
        
        return confidence

    def generate_rollout(
        self,
        x: torch.Tensor,
        noise_scale: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """Generate rollout by adding noise to intermediate states."""
        # Initial forward pass
        with torch.no_grad():
            outputs = self.forward_pass(x)
            states = outputs['intermediate_states']
            
            # Add noise to states
            noisy_states = []
            for state in states:
                noise = torch.randn_like(state) * noise_scale
                noisy_states.append(state + noise)
            
            # Forward pass with noisy states
            features = noisy_states[-1].mean(dim=1)
            logits = self.fc_out(features)
            
            return {
                'states': noisy_states,
                'logits': logits,
                'features': features
            }

    def train_confidence_net(self, model, data_loader, num_epochs=100, device='cuda'):
        # Example usage
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.confidence_net.parameters(), lr=1e-4)
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)

                # Generate rollouts
                logits, states = model.monte_carlo_search(x, return_states=True)

                # Compute confidence
                confidence_scores = model.estimate_step_confidence(states, logits)

                # Define target confidence (e.g., based on correctness of logits).
                probs = F.softmax(logits, dim=-1)
                max_probs, pred_classes = probs.max(dim=-1)

                # Convert y to class indices if it's one-hot encoded.
                if y.dim() == 2 and y.size(-1) > 1:
                    y = y.argmax(dim=-1)

                target_confidence = (pred_classes == y).float().unsqueeze(-1) * max_probs.unsqueeze(-1)

                # Compute loss
                loss = criterion(confidence_scores, target_confidence)
                epoch_loss += loss.item()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    def monte_carlo_search(
        self, 
        x: torch.Tensor,
        return_states: bool = False,
        temperature: float = 0.8
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Perform Monte Carlo search over transformer computations."""
        best_confidence = -float('inf')
        best_output = None
        best_states = None
        
        with torch.no_grad():
            for _ in range(self.num_mc_rollouts):
                # Start fresh each rollout
                features = x
                curr_states = []
                
                # First pass through spectral embedding with noise
                features = self.spectral_embedding(features)
                if features.dim() == 2:
                    features = features.unsqueeze(1)
                
                # Add temperature-scaled noise
                noise = torch.randn_like(features) * temperature
                features = features + noise
                
                # Process through transformer layers with random perturbations
                for layer_idx in range(len(self.attention_layers)):
                    # Add random noise at each layer
                    layer_noise = torch.randn_like(features) * temperature
                    features = features + layer_noise
                    
                    # Apply attention with random scaling
                    norm1 = self.layer_norms[layer_idx * 2](features)
                    attn = self.attention_layers[layer_idx](norm1)
                    attn = attn * (1.0 + torch.randn_like(attn) * temperature * 0.1)
                    features = features + self.dropout(attn)
                    
                    # Apply feed-forward with noise
                    norm2 = self.layer_norms[layer_idx * 2 + 1](features)
                    ff = self.feed_forward(norm2)
                    ff = ff * (1.0 + torch.randn_like(ff) * temperature * 0.1)
                    features = features + self.dropout(ff)
                    
                    # Store state
                    curr_states.append(features)
                
                # Generate logits from final features
                final_features = features.mean(dim=1)
                logits = self.fc_out(final_features)
                
                # Add temperature to logits
                logits = logits / temperature
                
                # Estimate confidence
                confidence = self.estimate_step_confidence(
                    curr_states,
                    logits
                )
                
                # Update best solution
                confidence_val = confidence.mean().item()
                if confidence_val > best_confidence:
                    best_confidence = confidence_val
                    best_output = logits
                    best_states = curr_states
        
        if return_states:
            return best_output, best_states
        return best_output

    def beam_search(
        self,
        x: torch.Tensor,
        beam_width: int = 4,
        num_iterations: int = 10
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Perform beam search over intermediate states."""
        with torch.no_grad():
            # Initial forward pass
            outputs = self.forward_pass(x)
            curr_states = outputs['intermediate_states']
            curr_logits = outputs['logits']
            
            # Initialize beam with first states
            beam = [{
                'states': curr_states,
                'logits': curr_logits,
                'score': self.estimate_step_confidence(curr_states, curr_logits).mean().item()
            }]
            
            for _ in range(num_iterations):
                candidates = []
                
                # Expand each beam
                for candidate in beam:
                    # Generate variations
                    for _ in range(beam_width):
                        # Add noise to states and recompute through transformer layers
                        features = x
                        noisy_states = []
                        
                        # First pass through spectral embedding
                        features = self.spectral_embedding(features)
                        if features.dim() == 2:
                            features = features.unsqueeze(1)
                        
                        # Process through each transformer layer with noise
                        for layer_idx in range(len(self.attention_layers)):
                            # Add noise to current state
                            noise = torch.randn_like(features) * 0.1
                            features = features + noise
                            
                            # Process through layer
                            features = self._process_step(features, layer_idx)
                            noisy_states.append(features)
                        
                        # Generate logits from final features
                        final_features = features.mean(dim=1)
                        logits = self.fc_out(final_features)
                        
                        # Score candidate
                        score = self.estimate_step_confidence(noisy_states, logits).mean().item()
                        
                        candidates.append({
                            'states': noisy_states,
                            'logits': logits,
                            'score': score
                        })
                
                # Select top candidates
                candidates.sort(key=lambda x: x['score'], reverse=True)
                beam = candidates[:beam_width]
            
            # Return best candidate
            best = max(beam, key=lambda x: x['score'])
            return best['logits'], best['states']

    def genetic_search(
        self,
        x: torch.Tensor,
        population_size: int = 20,
        num_generations: int = 10,
        mutation_rate: float = 0.3,  # Increased mutation rate
        noise_scale: float = 0.2     # Increased noise scale
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Perform genetic algorithm search over transformer states with increased variation."""
        with torch.no_grad():
            # Initial population creation with more diverse starting points
            population = []
            
            # Helper function for running transformer with noise
            def run_noisy_transformer(features, noise_scale):
                features = self.spectral_embedding(features)
                if features.dim() == 2:
                    features = features.unsqueeze(1)
                
                states = []
                # Add substantial initial noise
                features = features + torch.randn_like(features) * noise_scale * 2
                
                for layer_idx in range(len(self.attention_layers)):
                    # Layer-specific noise scale
                    layer_noise = noise_scale * (1 + layer_idx * 0.2)
                    
                    # Add pre-layer noise
                    features = features + torch.randn_like(features) * layer_noise
                    
                    # Process through attention with noise
                    norm1 = self.layer_norms[layer_idx * 2](features)
                    attn = self.attention_layers[layer_idx](norm1)
                    attn = attn * (1.0 + torch.randn_like(attn) * layer_noise)
                    features = features + self.dropout(attn)
                    
                    # Process through feed-forward with noise
                    norm2 = self.layer_norms[layer_idx * 2 + 1](features)
                    ff = self.feed_forward(norm2)
                    ff = ff * (1.0 + torch.randn_like(ff) * layer_noise)
                    features = features + self.dropout(ff)
                    
                    states.append(features)
                
                final_features = features.mean(dim=1)
                logits = self.fc_out(final_features)
                
                return states, logits, features

            # Initialize population with diverse individuals
            for _ in range(population_size):
                # Generate individual with random noise scale
                ind_noise = noise_scale * (0.5 + 2 * random.random())
                states, logits, features = run_noisy_transformer(x, ind_noise)
                
                # Score individual
                score = self.estimate_step_confidence(states, logits).mean().item()
                
                population.append({
                    'states': states,
                    'logits': logits,
                    'features': features,
                    'score': score
                })
            
            for gen in range(num_generations):
                new_population = []
                
                # Dynamic mutation rate that increases if population stagnates
                dynamic_mutation = mutation_rate * (1 + gen / (num_generations * 2))
                
                # Sort by fitness
                population.sort(key=lambda x: x['score'], reverse=True)
                
                # Keep top performers with some noise added
                elite_size = max(1, population_size // 8)
                for elite in population[:elite_size]:
                    states, logits, features = run_noisy_transformer(x, noise_scale * 0.1)
                    score = self.estimate_step_confidence(states, logits).mean().item()
                    new_population.append({
                        'states': states,
                        'logits': logits,
                        'features': features,
                        'score': score
                    })
                
                # Generate offspring
                while len(new_population) < population_size:
                    # Tournament selection
                    tournament_size = 3
                    parents = []
                    for _ in range(2):
                        tournament = random.sample(population, tournament_size)
                        tournament.sort(key=lambda x: x['score'], reverse=True)
                        parents.append(tournament[0])
                    
                    parent1, parent2 = parents[0], parents[1]
                    
                    # Start with fresh features
                    features = x
                    features = self.spectral_embedding(features)
                    if features.dim() == 2:
                        features = features.unsqueeze(1)
                    
                    # Add initial variation
                    if random.random() < dynamic_mutation:
                        features = features + torch.randn_like(features) * noise_scale * 2
                    
                    child_states = []
                    
                    # Process through layers with crossover and mutation
                    for layer_idx in range(len(self.attention_layers)):
                        # More aggressive crossover
                        if random.random() < 0.5:
                            alpha = random.random() * 2 - 0.5  # Allow extrapolation
                            features = alpha * parent1['states'][layer_idx] + (1 - alpha) * parent2['states'][layer_idx]
                        
                        # Layer-specific mutation
                        if random.random() < dynamic_mutation:
                            layer_noise = noise_scale * (1 + layer_idx * 0.2)
                            features = features + torch.randn_like(features) * layer_noise
                        
                        # Process through layer with potential noise
                        norm1 = self.layer_norms[layer_idx * 2](features)
                        attn = self.attention_layers[layer_idx](norm1)
                        if random.random() < dynamic_mutation:
                            attn = attn * (1.0 + torch.randn_like(attn) * noise_scale)
                        features = features + self.dropout(attn)
                        
                        norm2 = self.layer_norms[layer_idx * 2 + 1](features)
                        ff = self.feed_forward(norm2)
                        if random.random() < dynamic_mutation:
                            ff = ff * (1.0 + torch.randn_like(ff) * noise_scale)
                        features = features + self.dropout(ff)
                        
                        child_states.append(features)
                    
                    # Generate logits
                    final_features = features.mean(dim=1)
                    logits = self.fc_out(final_features)
                    
                    # Score child
                    score = self.estimate_step_confidence(child_states, logits).mean().item()
                    
                    new_population.append({
                        'states': child_states,
                        'logits': logits,
                        'features': features,
                        'score': score
                    })
                
                population = new_population
            
            # Return best individual
            best = max(population, key=lambda x: x['score'])
            return best['logits'], best['states']  
      
    def forward(
        self,
        x: torch.Tensor,
        use_mc_search: bool = False,
        use_beam_search: bool = False,
        use_genetic_search: bool = False,
        return_confidence: bool = False,
        return_states: bool = False,
        return_intermediary: bool = False,
        use_test_time_compute: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass with optional search methods and test-time compute."""
        # Store batch size
        batch_size = x.size(0)
        
        # Standard forward pass
        initial_outputs = self.forward_pass(x)
        initial_logits = initial_outputs['logits']
        
        # Get initial predictions
        initial_preds = torch.argmax(initial_logits, dim=-1)
        print("\nInitial predictions:", initial_preds.cpu().numpy())
        
        # During training or when test-time compute is disabled
        if not use_test_time_compute:
            if return_intermediary:
                return {
                    "logits": initial_logits,
                    "intermediary_steps": initial_outputs['intermediate_states']
                }
            if return_confidence:
                confidence = self.estimate_step_confidence(
                    initial_outputs['intermediate_states'],
                    initial_logits
                )
                return initial_logits, confidence
            return initial_logits
        
        # Test-time compute with search methods
        results = []
        uncertainties = []
        all_states = []
        
        # Track prediction changes
        prediction_changes = []
        
        # Process each example in batch
        for i in range(batch_size):
            x_i = x[i:i+1]  # Keep batch dimension
            initial_logits_i = initial_logits[i:i+1]
            initial_pred_i = initial_preds[i].item()
            
            # Get appropriate search method
            if use_beam_search:
                output_i, states_i = self.beam_search(x_i)
                method = "Beam Search"
            elif use_genetic_search:
                output_i, states_i = self.genetic_search(x_i)
                method = "Genetic Search"
            elif use_mc_search:
                output_i, states_i = self.monte_carlo_search(x_i, return_states=True)
                method = "Monte Carlo"
            else:
                outputs_i = self.forward_pass(x_i)
                output_i = outputs_i['logits']
                states_i = outputs_i['intermediate_states']
                method = "Standard"
            
            # Get confidence scores
            confidence_i = self.estimate_step_confidence(states_i, output_i)

            # Ensure the confidence is higher than the initial prediction.
            if confidence_i.mean().item() < self.estimate_step_confidence(initial_outputs['intermediate_states'], initial_logits_i).mean().item():
                # Return the original prediction.
                output_i = initial_logits_i
                states_i = initial_outputs['intermediate_states']
            
            # Check if prediction changed
            new_pred = torch.argmax(output_i, dim=-1).item()
            if new_pred != initial_pred_i:
                print(f"\nPrediction changed for example {i}:")
                print(f"  Method: {method}")
                print(f"  Initial prediction: {initial_pred_i}")
                print(f"  New prediction: {new_pred}")
                print(f"  Initial confidence: {self.estimate_step_confidence(initial_outputs['intermediate_states'], initial_logits_i).mean().item():.3f}")
                print(f"  New confidence: {confidence_i.mean().item():.3f}")
                prediction_changes.append({
                    'example': i,
                    'method': method,
                    'initial_pred': initial_pred_i,
                    'new_pred': new_pred,
                    'confidence_change': confidence_i.mean().item() - self.estimate_step_confidence(initial_outputs['intermediate_states'], initial_logits_i).mean().item()
                })
            
            # Store results
            results.append(output_i)
            uncertainties.append(confidence_i)
            all_states.append(states_i)
        
        # Stack results back into batches
        output = torch.cat(results, dim=0)
        
        # Print summary of changes
        if prediction_changes:
            print(f"\nTotal prediction changes: {len(prediction_changes)}")
            for change in prediction_changes:
                print(f"Example {change['example']}: {change['method']} changed prediction from {change['initial_pred']} to {change['new_pred']} (confidence delta: {change['confidence_change']:.3f})")
        else:
            print("\nNo predictions were changed by the search methods.")
            
            # Print logit differences to debug
            final_preds = torch.argmax(output, dim=-1)
            print("\nLogit analysis:")
            for i in range(batch_size):
                initial_probs = F.softmax(initial_logits[i], dim=-1)
                final_probs = F.softmax(results[i], dim=-1)
                print(f"\nExample {i}:")
                print(f"Initial probs: {initial_probs.cpu().numpy().round(3)}")
                print(f"Final probs: {final_probs.cpu().numpy().round(3)}")
                print(f"Prob diff: {(final_probs - initial_probs).abs().max().item():.3f}")
        
        # Ensure we maintained batch size
        assert output.size(0) == batch_size, \
            f"Expected batch size {batch_size}, got {output.size(0)}"
        
        # Prepare return values
        if return_confidence or return_states or return_intermediary:
            return_dict = {}
            if return_confidence:
                return_dict['confidence'] = torch.cat(uncertainties, dim=0)
            if return_states:
                try:
                    return_dict['states'] = [torch.cat([s[i] for s in all_states], dim=0) 
                                           for i in range(len(all_states[0]))]
                except:
                    return_dict['states'] = all_states
            if return_intermediary:
                return_dict['intermediary_steps'] = all_states
            return output, return_dict
            
        return output