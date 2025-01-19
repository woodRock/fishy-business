import random
import math
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from typing import Tuple, Optional, List, Dict, Union

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
        
        # Store attention pattern for chain-of-thought exploration
        self.last_attention = attn
        
        x = (attn @ v).transpose(1, 2).reshape(batch_size, -1, self.input_dim)
        x = self.fc_out(x)
        return x

class ImprovedPRM(nn.Module):
    """Process Reward Model with improved calibration and uncertainty estimation."""
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        
        self.num_classes = num_classes
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        # Feature extraction network operating on logits
        self.feature_net = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),  # Takes logits as input
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Confidence prediction networks
        self.confidence_net = nn.ModuleDict({
            'aleatoric': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1)
            ),
            'epistemic': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1)
            )
        })
        
        # Class-specific calibration
        self.class_calibration = nn.Parameter(torch.zeros(num_classes, 2))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='linear')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        nn.init.ones_(self.class_calibration[:, 0])
        nn.init.zeros_(self.class_calibration[:, 1])
    
    def forward(
        self,
        logits: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Extract features from logits
        features = self.feature_net(logits)
        
        # Get predicted classes and probabilities
        probs = torch.softmax(logits / self.temperature, dim=-1)
        pred_classes = probs.argmax(dim=-1)
        
        # Get uncertainty estimates
        aleatoric_conf = torch.sigmoid(self.confidence_net['aleatoric'](features))
        epistemic_conf = torch.sigmoid(self.confidence_net['epistemic'](features))
        
        # Class-specific calibration
        scales = self.class_calibration[pred_classes, 0].unsqueeze(-1)
        biases = self.class_calibration[pred_classes, 1].unsqueeze(-1)
        
        # Combine confidences with class calibration
        base_confidence = (aleatoric_conf * epistemic_conf)
        calibrated_confidence = torch.sigmoid(scales * base_confidence + biases)
        
        if return_uncertainty:
            uncertainty_info = {
                'aleatoric': aleatoric_conf,
                'epistemic': epistemic_conf,
                'total': 1 - calibrated_confidence,
                'temperature': self.temperature,
                'class_scales': scales,
                'class_biases': biases
            }
            return calibrated_confidence, uncertainty_info
        
        return calibrated_confidence

    def loss_function(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        confidence: torch.Tensor,
        uncertainty_info: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Get predictions
        probs = torch.softmax(logits / self.temperature, dim=-1)
        preds = probs.argmax(dim=-1)
        correct = (preds == targets.argmax(dim=-1))
        
        # Basic confidence loss (focal loss for better calibration)
        gamma = 2.0  # focal loss parameter
        pt = correct.float()
        focal_weights = (1 - pt) ** gamma
        confidence_loss = F.binary_cross_entropy(
            confidence.squeeze(),
            pt,
            weight=focal_weights
        )
        
        # Uncertainty regularization
        aleatoric_penalty = torch.mean(
            torch.abs(uncertainty_info['aleatoric'] - pt)
        )
        epistemic_penalty = torch.mean(
            torch.abs(uncertainty_info['epistemic'] - pt)
        )
        
        # Temperature regularization
        temp_reg = 0.01 * torch.abs(self.temperature - 1.5)
        
        # Class calibration regularization
        calibration_reg = 0.01 * (
            torch.norm(self.class_calibration[:, 0] - 1.0) +
            torch.norm(self.class_calibration[:, 1])
        )
        
        # Total loss
        total_loss = (
            confidence_loss +
            0.1 * aleatoric_penalty +
            0.1 * epistemic_penalty +
            temp_reg +
            calibration_reg
        )
        
        return total_loss, {
            'confidence_loss': confidence_loss.item(),
            'aleatoric_penalty': aleatoric_penalty.item(),
            'epistemic_penalty': epistemic_penalty.item(),
            'temp_reg': temp_reg.item(),
            'calibration_reg': calibration_reg.item(),
            'total_loss': total_loss.item()
        }
    
class TestTimeTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        num_iterations: int = 5,
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
        
        # Store attention layers in ModuleList for access during genetic programming
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
        self.process_reward_model = ImprovedPRM(
            input_dim=output_dim,  # PRM input should match model output dimension
            hidden_dim=hidden_dim,
            num_classes=output_dim)

        # New: adaptive thresholds based on calibration
        self.register_buffer('threshold_history', torch.zeros(100))
        self.threshold_idx = 0

    def _standard_forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Store attention patterns and features for chain-of-thought exploration
        self.attention_patterns = []
        features = x
        
        # Store intermediate states
        self.intermediate_features = []
        
        for attention in self.attention_layers:
            residual = features
            features = self.layer_norm1(features)
            attention_output = attention(features)
            # Store attention pattern and features
            if hasattr(attention, 'last_attention'):
                self.attention_patterns.append(attention.last_attention)
            self.intermediate_features.append(features)
            features = residual + self.dropout(attention_output)
        
        residual = features
        features = self.layer_norm2(features)
        features = residual + self.dropout(self.feed_forward(features))
        
        features = features.mean(dim=1)
        output = self.fc_out(features)
        return output

    def forward(
        self,
        x: torch.Tensor,
        use_test_time_compute: bool = False,
        use_beam_search: bool = False,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Enhanced forward pass with uncertainty estimation"""
        # Standard forward pass
        initial_output = self._standard_forward(x)
        
        if not use_test_time_compute:
            if return_uncertainty:
                confidence, uncertainty_info = self.process_reward_model(
                    initial_output,
                    return_uncertainty=True
                )
                return initial_output, uncertainty_info
            return initial_output
        
        # Get confidence and uncertainty estimates
        confidence, uncertainty_info = self.process_reward_model(initial_output, return_uncertainty=True)
        
        # Debug: Print shapes
        print("\nUncertainty info shapes:")
        for k, v in uncertainty_info.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}")
            else:
                print(f"{k}: {type(v)}")
                
        print(f"\nInitial confidence shape: {confidence.shape}")
        
        # Ensure confidence is properly shaped
        confidence = confidence.squeeze()
        if confidence.dim() == 0:
            confidence = confidence.unsqueeze(0)
        
        # Get uncertainties with proper shapes
        total_uncertainty = uncertainty_info['total'].squeeze()
        if total_uncertainty.dim() == 0:
            total_uncertainty = total_uncertainty.unsqueeze(0)
        
        aleatoric = uncertainty_info['aleatoric'].squeeze()
        if aleatoric.dim() == 0:
            aleatoric = aleatoric.unsqueeze(0)
        
        epistemic = uncertainty_info['epistemic'].squeeze()
        if epistemic.dim() == 0:
            epistemic = epistemic.unsqueeze(0)
            
        aleatoric_uncertainty = 1 - aleatoric
        epistemic_uncertainty = 1 - epistemic
        
        print(f"\nProcessed shapes:")
        print(f"Confidence: {confidence.shape}")
        print(f"Total uncertainty: {total_uncertainty.shape}")
        print(f"Aleatoric uncertainty: {aleatoric_uncertainty.shape}")
        print(f"Epistemic uncertainty: {epistemic_uncertainty.shape}")
        
        # Dynamic threshold adjustment
        dynamic_min_threshold = self.min_confidence_threshold * (
            1 + 0.5 * epistemic_uncertainty
        )
        dynamic_max_threshold = self.max_confidence_threshold * (
            1 + 0.2 * aleatoric_uncertainty
        )
        
        results = []
        uncertainties = []
        
        for i in range(x.shape[0]):
            # Create sample-specific uncertainty info
            sample_uncertainty = {}
            for k, v in uncertainty_info.items():
                if isinstance(v, torch.Tensor):
                    # Handle different tensor shapes
                    v_squeezed = v.squeeze()
                    if v_squeezed.dim() == 0:  # Scalar tensor
                        sample_uncertainty[k] = v_squeezed.unsqueeze(0)
                    elif v_squeezed.dim() == 1:  # 1D tensor
                        if len(v_squeezed) == 1:
                            sample_uncertainty[k] = v_squeezed
                        else:
                            sample_uncertainty[k] = v_squeezed[i].unsqueeze(0)
                    else:  # Multi-dimensional tensor
                        sample_uncertainty[k] = v_squeezed[i:i+1]
                else:
                    sample_uncertainty[k] = v
            
            if confidence[i] > dynamic_max_threshold[i]:
                # High confidence: use initial prediction
                results.append(initial_output[i])
                uncertainties.append(sample_uncertainty)
            
            elif confidence[i] > dynamic_min_threshold[i]:
                # Medium confidence: use beam search
                if use_beam_search:
                    beam_output, beam_uncertainty = self.beam_search_forward(
                        x[i].unsqueeze(0),
                        initial_output[i].unsqueeze(0),
                        uncertainty_info=sample_uncertainty
                    )
                else: 
                    beam_output, beam_uncertainty = self.genetic_forward(
                        x[i].unsqueeze(0),
                        initial_output[i].unsqueeze(0),
                        uncertainty_info=sample_uncertainty
                    )
                results.append(beam_output.squeeze(0))
                uncertainties.append(beam_uncertainty)
            
            else:
                # Low confidence: try both methods
                beam_output, beam_uncertainty = self.beam_search_forward(
                    x[i].unsqueeze(0),
                    initial_output[i].unsqueeze(0),
                    uncertainty_info=sample_uncertainty
                )
                
                genetic_output, genetic_uncertainty = self.genetic_forward(
                    x[i].unsqueeze(0),
                    initial_output[i].unsqueeze(0),
                    uncertainty_info=sample_uncertainty
                )
                
                # Choose based on calibrated confidence
                beam_conf = self.process_reward_model(beam_output)[0]
                genetic_conf = self.process_reward_model(genetic_output)[0]
                
                if beam_conf > genetic_conf:
                    results.append(beam_output.squeeze(0))
                    uncertainties.append(beam_uncertainty)
                else:
                    results.append(genetic_output.squeeze(0))
                    uncertainties.append(genetic_uncertainty)
        
        output = torch.stack(results)
        
        if return_uncertainty:
            # Aggregate uncertainties
            combined_uncertainty = {}
            for k in uncertainties[0].keys():
                if isinstance(uncertainties[0][k], torch.Tensor):
                    try:
                        combined_uncertainty[k] = torch.cat([u[k] for u in uncertainties])
                    except:
                        # Fallback for tensors that can't be concatenated
                        combined_uncertainty[k] = uncertainties[0][k]
                else:
                    combined_uncertainty[k] = uncertainties[0][k]
            return output, combined_uncertainty
        
        return output

    def train_prm(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        patience: int = 10,
        learning_rate: float = 1e-4,
        calibration_frequency: int = 5,
        min_improvement: float = 1e-4
    ) -> Dict[str, List[float]]:
        """
        Train Process Reward Model with improved calibration and uncertainty estimation.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            num_epochs: Maximum number of training epochs
            patience: Early stopping patience
            learning_rate: Learning rate for optimizer
            calibration_frequency: Epochs between calibration updates
            min_improvement: Minimum validation improvement for early stopping
        
        Returns:
            Dictionary containing training history
        """
        # Initialize optimizers with different learning rates for different components
        optimizer = torch.optim.AdamW([
            {'params': self.process_reward_model.feature_net.parameters(), 'lr': learning_rate},
            {'params': self.process_reward_model.confidence_net.parameters(), 'lr': learning_rate},
            {'params': [self.process_reward_model.temperature], 'lr': learning_rate * 0.1},
            {'params': [self.process_reward_model.class_calibration], 'lr': learning_rate * 0.01}
        ])
        
        # Learning rate scheduler with warmup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[learning_rate, learning_rate, learning_rate * 0.1, learning_rate * 0.01],
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,  # 10% warmup
            div_factor=25,
            final_div_factor=1000
        )
        
        # Initialize training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'ece': [],
            'class_ece': defaultdict(list),
            'temperature': [],
            'class_calibration': []
        }
        
        # Early stopping setup
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # Training loop
        for epoch in range(num_epochs):
            self.process_reward_model.train()
            epoch_losses = defaultdict(list)
            
            # Training phase
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # Forward pass with uncertainty
                with torch.no_grad():
                    outputs = self._standard_forward(batch_x)
                
                confidence, uncertainty_info = self.process_reward_model(
                    outputs,
                    return_uncertainty=True
                )
                
                # Compute loss with uncertainty components
                loss, loss_components = self.process_reward_model.loss_function(
                    logits=outputs,
                    targets=batch_y,
                    confidence=confidence,
                    uncertainty_info=uncertainty_info
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.process_reward_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Record losses
                for k, v in loss_components.items():
                    epoch_losses[k].append(v)
                
                # Log progress
                if (batch_idx + 1) % 50 == 0:
                    print(f"\nEpoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}")
                    print(f"Temperature: {self.process_reward_model.temperature.item():.3f}")
                    for k, v in epoch_losses.items():
                        print(f"{k}: {np.mean(v):.4f}")
            
            # Record training metrics
            history['train_loss'].append(np.mean(epoch_losses['total_loss']))
            history['temperature'].append(self.process_reward_model.temperature.item())
            history['class_calibration'].append(
                self.process_reward_model.class_calibration.detach().cpu().numpy()
            )
            
            # Validation phase
            if val_loader is not None:
                self.process_reward_model.eval()
                val_losses = defaultdict(list)
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.float().to(self.device)
                        batch_y = batch_y.float().to(self.device)
                        
                        outputs = self._standard_forward(batch_x)
                        confidence, uncertainty_info = self.process_reward_model(
                            outputs,
                            return_uncertainty=True
                        )
                        
                        loss, loss_components = self.process_reward_model.loss_function(
                            logits=outputs,
                            targets=batch_y,
                            confidence=confidence,
                            uncertainty_info=uncertainty_info
                        )
                        
                        for k, v in loss_components.items():
                            val_losses[k].append(v)
                
                val_loss = np.mean(val_losses['total_loss'])
                history['val_loss'].append(val_loss)
                
                # Calibration update
                if (epoch + 1) % calibration_frequency == 0:
                    calibration_metrics = self.calibrate(val_loader)
                    history['ece'].append(calibration_metrics['ece'])
                    
                    for class_idx, ece in calibration_metrics['class_ece'].items():
                        history['class_ece'][class_idx].append(ece)
                    
                    print("\nCalibration Metrics:")
                    print(f"ECE: {calibration_metrics['ece']:.4f}")
                    print("Class-wise ECE:", {k: f"{v:.4f}" for k, v in calibration_metrics['class_ece'].items()})
                
                # Early stopping check
                if val_loss < best_val_loss - min_improvement:
                    best_val_loss = val_loss
                    best_model_state = {
                        k: v.cpu() for k, v in self.process_reward_model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Train Loss: {history['train_loss'][-1]:.4f}")
            if val_loader is not None:
                print(f"Validation Loss: {history['val_loss'][-1]:.4f}")
            print(f"Temperature: {history['temperature'][-1]:.3f}")
        
        # Restore best model if early stopping occurred
        if best_model_state is not None:
            self.process_reward_model.load_state_dict(best_model_state)
        
        return history

    def calibrate(
        self,
        val_loader: DataLoader,
        num_bins: int = 10
    ) -> Dict[str, float]:
        """Calibrate model using validation data."""
        self.eval()
        self.process_reward_model.eval()
        
        confidences = []
        accuracies = []
        class_predictions = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidence': []})
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # Get model predictions
                logits = self._standard_forward(batch_x)
                confidence, uncertainty_info = self.process_reward_model(logits, return_uncertainty=True)
                
                # Get predictions and true labels
                predictions = logits.argmax(dim=-1)
                true_labels = batch_y.argmax(dim=-1)
                correct = (predictions == true_labels)
                
                # Update statistics
                confidences.extend(confidence.squeeze().cpu().numpy())
                accuracies.extend(correct.cpu().numpy())
                
                # Update class-specific statistics
                for i, (pred, conf, corr) in enumerate(zip(predictions, confidence, correct)):
                    pred_class = pred.item()
                    class_predictions[pred_class]['total'] += 1
                    class_predictions[pred_class]['correct'] += int(corr)
                    class_predictions[pred_class]['confidence'].append(conf.item())
            
            # Calculate calibration metrics
            confidences = np.array(confidences)
            accuracies = np.array(accuracies)
            
            # Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, num_bins + 1)
            bin_accuracies = []
            bin_confidences = []
            bin_counts = []
            
            for i in range(num_bins):
                mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
                if np.any(mask):
                    bin_accuracies.append(np.mean(accuracies[mask]))
                    bin_confidences.append(np.mean(confidences[mask]))
                    bin_counts.append(np.sum(mask))
            
            bin_accuracies = np.array(bin_accuracies)
            bin_confidences = np.array(bin_confidences)
            bin_counts = np.array(bin_counts)
            
            # Calculate ECE
            ece = np.sum(
                np.abs(bin_accuracies - bin_confidences) * (bin_counts / len(confidences))
            )
            
            # Calculate class-specific calibration
            class_ece = {}
            for class_idx, stats in class_predictions.items():
                if stats['total'] > 0:
                    class_acc = stats['correct'] / stats['total']
                    class_conf = np.mean(stats['confidence'])
                    class_ece[class_idx] = abs(class_acc - class_conf)
            
            # Calculate final metrics
            calibration_metrics = {
                'ece': float(ece),
                'mean_confidence': float(np.mean(confidences)),
                'mean_accuracy': float(np.mean(accuracies)),
                'class_ece': class_ece
            }
            
            # Update confidence thresholds based on ECE
            if ece > 0.1:  # High calibration error
                self.min_confidence_threshold *= 1.1
                self.max_confidence_threshold *= 1.1
            elif ece < 0.05:  # Low calibration error
                self.min_confidence_threshold *= 0.9
                self.max_confidence_threshold *= 0.9
            
            # Store threshold history
            self.threshold_history[self.threshold_idx] = self.min_confidence_threshold
            self.threshold_idx = (self.threshold_idx + 1) % self.threshold_history.size(0)
            
            return calibration_metrics

    def beam_search_forward(
        self,
        x: torch.Tensor,
        initial_outputs: torch.Tensor,
        uncertainty_info: Dict[str, torch.Tensor],
        num_iterations: Optional[int] = None,
        beam_width: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Enhanced beam search with uncertainty guidance"""
        num_iterations = num_iterations or self.num_iterations
        beam_width = beam_width or self.beam_width
        
        # Initialize beam states with uncertainty weighting
        aleatoric_uncertainty = 1 - uncertainty_info['aleatoric']
        epistemic_uncertainty = 1 - uncertainty_info['epistemic']
        
        # Adjust exploration based on uncertainty types
        exploration_scale = torch.sqrt(
            0.5 * aleatoric_uncertainty + 0.5 * epistemic_uncertainty
        )
        
        batch_size = x.shape[0]
        device = x.device
        output_dim = initial_outputs.shape[-1]
        
        # Initialize beam states
        beam_states = {
            'candidates': initial_outputs.unsqueeze(1).repeat(1, beam_width, 1),
            'scores': torch.zeros(batch_size, beam_width, device=device),
            'temperature': torch.ones(batch_size, device=device) * self.temperature
        }
        
        # Initialize momentum buffers
        momentum = torch.zeros_like(initial_outputs)
        best_score_history = []
        best_uncertainty = None
        
        # Calculate initial scores
        with torch.no_grad():
            initial_score, initial_uncertainty = self.process_reward_model(initial_outputs, return_uncertainty=True)
            beam_states['scores'][:, 0] = initial_score.squeeze()
            best_score = initial_score.clone()
            best_candidates = initial_outputs.clone()
            best_uncertainty = initial_uncertainty
            
            # Add initial score to history
            best_score_history.append(best_score.item())
            
            for iteration in range(num_iterations):
                # Generate candidates
                curr_exploration_scale = exploration_scale * (1 - iteration/num_iterations)
                
                candidates_list = []
                scores_list = []
                uncertainties_list = []
                
                for b in range(beam_width):
                    # Add noise and momentum
                    noise = torch.randn_like(best_candidates) * curr_exploration_scale
                    momentum = 0.9 * momentum + 0.1 * noise
                    candidate = best_candidates + momentum
                    
                    # Get score and uncertainty for this candidate
                    score, uncertainty = self.process_reward_model(candidate, return_uncertainty=True)
                    
                    candidates_list.append(candidate)
                    scores_list.append(score.squeeze())
                    uncertainties_list.append(uncertainty)
                
                # Stack candidates and scores
                candidates = torch.cat(candidates_list, dim=0)
                scores = torch.stack(scores_list)
                
                # Select top candidates
                top_score, top_idx = scores.max(dim=0)
                top_candidate = candidates_list[top_idx]
                
                # Update best if improved
                if top_score > best_score:
                    best_score = top_score.clone()
                    best_candidates = top_candidate.clone()
                    best_uncertainty = uncertainties_list[top_idx]
                    print(f"Iteration {iteration + 1}: New best score = {best_score.item():.4f}")
                
                # Update history
                best_score_history.append(best_score.item())
                
                # Dynamic temperature adjustment
                if len(best_score_history) >= 2:
                    improvement = (best_score_history[-1] - best_score_history[-2])
                    if improvement < 1e-4:
                        beam_states['temperature'] *= 1.1
                    else:
                        beam_states['temperature'] *= 0.9
                    beam_states['temperature'] = torch.clamp(beam_states['temperature'], 0.1, 2.0)
        
        return best_candidates, best_uncertainty

    def genetic_forward(
        self,
        x: torch.Tensor,
        initial_outputs: torch.Tensor,
        uncertainty_info: Dict[str, torch.Tensor],
        num_generations: int = 10,
        population_size: int = 100,
        elite_size: int = 10,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Genetic algorithm-based forward pass with uncertainty guidance and improved
        preservation of initial solutions.
        """
        # Get uncertainties and adjust parameters
        aleatoric_uncertainty = 1 - uncertainty_info['aleatoric']
        epistemic_uncertainty = 1 - uncertainty_info['epistemic']
        
        # Reduce mutation rate when uncertainty is high
        mutation_rate = 0.1 * (1 - epistemic_uncertainty.item())
        
        # Adjust population size based on aleatoric uncertainty
        effective_population_size = max(
            elite_size + 1,
            int(population_size * (1 - aleatoric_uncertainty.item()))
        )
        
        # Initialize containers with proper shapes
        batch_size = x.shape[0]
        device = x.device
        
        # Pre-allocate tensors
        population = torch.empty(
            effective_population_size, 
            batch_size, 
            self.output_dim, 
            device=device
        ).contiguous()
        
        scores = torch.empty(effective_population_size, device=device)
        uncertainties = []
        
        # Track metrics for debugging
        solution_changes = []
        score_distribution = []
        acceptance_rates = []
        
        with torch.no_grad():
            # Fill first half of population with initial outputs
            num_copies = effective_population_size // 2
            for i in range(num_copies):
                if i == 0:
                    # Keep one exact copy
                    population[i] = initial_outputs.clone()
                else:
                    # Apply progressively larger mutations
                    population[i] = initial_outputs.clone()
                    noise_scale = 0.01 * i/num_copies
                    noise = torch.randn_like(initial_outputs) * noise_scale
                    population[i] += noise
                
                # Evaluate copy
                score, curr_uncertainty = self.process_reward_model(
                    population[i],
                    return_uncertainty=True
                )
                scores[i] = score.squeeze()
                uncertainties.append(curr_uncertainty)
                
                # Track changes
                solution_changes.append(
                    torch.norm(population[i] - initial_outputs).item()
                )
            
            # Initialize best solution tracking
            best_idx = scores[:num_copies].argmax()
            best_score = scores[best_idx].item()
            best_individual = population[best_idx].clone()
            best_uncertainty = uncertainties[best_idx]
            
            # Generate remaining population through attention perturbation
            init_features = x.unsqueeze(1)
            layer_features = []
            layer_attentions = []
            features = init_features
            
            # Get baseline attention patterns
            for attention_layer in self.attention_layers:
                qkv = attention_layer.qkv(features)
                qkv = qkv.reshape(
                    batch_size, -1, 3,
                    attention_layer.num_heads,
                    attention_layer.head_dim
                )
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                attn = (q @ k.transpose(-2, -1)) * attention_layer.scale
                attn = attn.softmax(dim=-1)
                layer_attentions.append(attn)
                
                out = (attn @ v).transpose(1, 2).reshape(
                    batch_size, -1,
                    attention_layer.input_dim
                )
                layer_features.append(out)
                features = out
            
            # Fill remaining population with attention-based solutions
            for i in range(num_copies, effective_population_size):
                features = init_features
                
                # Process through attention layers with perturbation
                for j, (attn, attention_layer) in enumerate(zip(layer_attentions, self.attention_layers)):
                    # Scale noise based on uncertainty
                    noise_scale = 0.1 * (1 - epistemic_uncertainty.item())
                    noise = torch.randn_like(attn) * noise_scale
                    curr_attn = (attn + noise).softmax(dim=-1)
                    
                    qkv = attention_layer.qkv(features)
                    qkv = qkv.reshape(
                        batch_size, -1, 3,
                        attention_layer.num_heads,
                        attention_layer.head_dim
                    )
                    qkv = qkv.permute(2, 0, 3, 1, 4)
                    _, _, v = qkv[0], qkv[1], qkv[2]
                    
                    out = (curr_attn @ v).transpose(1, 2).reshape(
                        batch_size, -1,
                        attention_layer.input_dim
                    )
                    features = out + layer_features[j]
                
                # Generate logits and evaluate
                logits = self.fc_out(features.mean(dim=1))
                score, curr_uncertainty = self.process_reward_model(
                    logits,
                    return_uncertainty=True
                )
                
                population[i] = logits
                scores[i] = score.squeeze()
                uncertainties.append(curr_uncertainty)
                
                # Track changes
                solution_changes.append(
                    torch.norm(logits - initial_outputs).item()
                )
            
            # Evolution loop
            for generation in range(num_generations):
                accepted_solutions = 0
                
                # Adaptive parameters based on generation progress
                generation_progress = generation / num_generations
                current_mutation_rate = mutation_rate * (1 - 0.8 * generation_progress)  # Decay mutation rate
                current_mutation_scale = 0.01 * (1 - 0.9 * generation_progress)  # Smaller mutations later
                
                # Dynamic population sizing to prevent stagnation
                dynamic_elite_size = min(
                    elite_size + int(generation_progress * elite_size), 
                    effective_population_size // 2
                )
                
                # Sort and apply diversity preservation
                sorted_indices = scores.argsort(descending=True)
                population = population[sorted_indices]
                scores = scores[sorted_indices]
                sorted_uncertainties = [uncertainties[idx.item()] for idx in sorted_indices]
                uncertainties = sorted_uncertainties
                
                # Record score distribution
                score_distribution.append({
                    'mean': scores.mean().item(),
                    'std': scores.std().item(),
                    'max': scores.max().item(),
                    'min': scores.min().item()
                })
                
                # Stronger elitism for longer runs
                new_population = torch.empty_like(population)
                new_scores = torch.empty_like(scores)
                new_uncertainties = []
                
                # Always preserve multiple versions of best solutions
                new_population[0] = best_individual
                new_scores[0] = best_score
                new_uncertainties.append(best_uncertainty)
                
                new_population[1] = initial_outputs  # Keep original solution
                score, curr_uncertainty = self.process_reward_model(initial_outputs, return_uncertainty=True)
                new_scores[1] = score.squeeze()
                new_uncertainties.append(curr_uncertainty)
                
                # Maintain diversity in elite pool
                elite_counter = 2
                seen_solutions = {
                    tuple(best_individual.flatten().tolist()),
                    tuple(initial_outputs.flatten().tolist())
                }
                
                for i in range(population_size):
                    if elite_counter >= dynamic_elite_size:
                        break
                        
                    solution_tuple = tuple(population[i].flatten().tolist())
                    if solution_tuple not in seen_solutions:
                        new_population[elite_counter] = population[i].clone()
                        new_scores[elite_counter] = scores[i].clone()
                        new_uncertainties.append(uncertainties[i])
                        seen_solutions.add(solution_tuple)
                        elite_counter += 1
                
                # Generate offspring with diversity-aware breeding
                offspring_idx = elite_counter
                while offspring_idx < effective_population_size:
                    # Diverse parent selection
                    parent1_idx = torch.randint(elite_counter, (1,)).item()
                    
                    # Select second parent that's different from first
                    for _ in range(5):  # Try up to 5 times to find different parent
                        parent2_idx = torch.randint(elite_counter, (1,)).item()
                        if not torch.allclose(new_population[parent1_idx], new_population[parent2_idx]):
                            break
                    
                    # Crossover with adaptive mixing
                    crossover_point = torch.randint(self.output_dim, (1,)).item()
                    child = new_population[parent1_idx].clone()
                    
                    # Interpolated crossover
                    mix_ratio = torch.rand(1).item()  # Random mixing ratio
                    child[..., crossover_point:] = (
                        mix_ratio * child[..., crossover_point:] + 
                        (1 - mix_ratio) * new_population[parent2_idx, ..., crossover_point:]
                    )
                    
                    # Adaptive mutation based on generation and parent scores
                    if torch.rand(1) < current_mutation_rate:
                        parent_score_diff = abs(new_scores[parent1_idx] - new_scores[parent2_idx])
                        mutation_strength = current_mutation_scale * (1 + parent_score_diff)
                        mutation_noise = torch.randn_like(child) * mutation_strength
                        child = child + mutation_noise
                    
                    # Evaluate child
                    score, curr_uncertainty = self.process_reward_model(
                        child,
                        return_uncertainty=True
                    )
                    
                    # Record if solution was accepted
                    if score > scores[offspring_idx - 1]:
                        accepted_solutions += 1
                    
                    # Only accept if significantly better than parents or adds diversity
                    parent_max_score = max(new_scores[parent1_idx], new_scores[parent2_idx])
                    child_tuple = tuple(child.flatten().tolist())
                    
                    if (score > parent_max_score * 1.001 or  # 0.1% improvement threshold
                        child_tuple not in seen_solutions):
                        new_population[offspring_idx] = child
                        new_scores[offspring_idx] = score.squeeze()
                        new_uncertainties.append(curr_uncertainty)
                        seen_solutions.add(child_tuple)
                        
                        # Update best if improved
                        if score > best_score:
                            best_score = score.item()
                            best_individual = child.clone()
                            best_uncertainty = curr_uncertainty
                            print(f"New best score at generation {generation}: {best_score:.4f}")
                    else:
                        # If child rejected, clone better parent
                        better_parent_idx = parent1_idx if new_scores[parent1_idx] > new_scores[parent2_idx] else parent2_idx
                        new_population[offspring_idx] = new_population[better_parent_idx].clone()
                        new_scores[offspring_idx] = new_scores[better_parent_idx].clone()
                        new_uncertainties.append(new_uncertainties[better_parent_idx])
                    
                    offspring_idx += 1
                
                # Update population
                population = new_population
                scores = new_scores
                uncertainties = new_uncertainties
                
                # Record acceptance rate
                acceptance_rates.append(
                    accepted_solutions / (effective_population_size - elite_counter)
                )
                
                # Early stopping if no improvement
                if generation > 2:
                    recent_scores = [d['max'] for d in score_distribution[-3:]]
                    if max(recent_scores) - min(recent_scores) < 1e-4:
                        break
            
            # Log statistics
            print(f"\nGenetic Algorithm Statistics:")
            print(f"Final best score: {best_score:.4f}")
            print(f"Mean solution change: {np.mean(solution_changes):.4f}")
            print(f"Mean acceptance rate: {np.mean(acceptance_rates):.4f}")
            print(f"Score distribution in final generation:")
            print(f"  Mean: {score_distribution[-1]['mean']:.4f}")
            print(f"  Std:  {score_distribution[-1]['std']:.4f}")
            print(f"  Max:  {score_distribution[-1]['max']:.4f}")
            print(f"  Min:  {score_distribution[-1]['min']:.4f}")
        
        return best_individual, best_uncertainty
    
    def evaluate_prm(self, val_loader: DataLoader) -> None:
        """Detailed evaluation of PRM's prediction quality."""
        self.process_reward_model.eval()
        with torch.no_grad():
            confidence_buckets = defaultdict(lambda: {'correct': 0, 'total': 0})
            class_confidences = defaultdict(list)
            
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # Get model predictions
                outputs = self._standard_forward(batch_x)
                confidence = self.process_reward_model(outputs).squeeze()
                
                # Get predictions and true labels
                predictions = outputs.argmax(dim=-1)
                true_labels = batch_y.argmax(dim=-1)
                correct = (predictions == true_labels)
                
                # Analyze confidence distribution
                for conf, pred, true, is_correct in zip(confidence, predictions, true_labels, correct):
                    conf_bucket = round(conf.item() * 10) / 10  # Round to nearest 0.1
                    confidence_buckets[conf_bucket]['total'] += 1
                    confidence_buckets[conf_bucket]['correct'] += int(is_correct)
                    class_confidences[true.item()].append(conf.item())
            
            # Print confidence calibration
            print("\nConfidence Calibration:")
            print("Confidence | Accuracy | Samples")
            print("-" * 35)
            for conf in sorted(confidence_buckets.keys()):
                bucket = confidence_buckets[conf]
                accuracy = bucket['correct'] / bucket['total'] if bucket['total'] > 0 else 0
                print(f"{conf:9.1f} | {accuracy:8.3f} | {bucket['total']:7d}")
            
            # Print per-class confidence statistics
            print("\nPer-class Confidence Statistics:")
            print("Class | Mean Conf | Std Dev | Min | Max | Samples")
            print("-" * 50)
            for class_idx in sorted(class_confidences.keys()):
                confs = class_confidences[class_idx]
                if confs:
                    mean_conf = np.mean(confs)
                    std_conf = np.std(confs)
                    min_conf = min(confs)
                    max_conf = max(confs)
                    print(f"{class_idx:5d} | {mean_conf:9.3f} | {std_conf:7.3f} | {min_conf:3.3f} | {max_conf:3.3f} | {len(confs):7d}")