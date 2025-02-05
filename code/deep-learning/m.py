import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import numpy as np
from typing import Tuple, List, Dict

from util import create_data_module

class ExpertLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ReasoningHeuristics:
    @staticmethod
    def feature_attribution(intermediate: torch.Tensor, layer_output: torch.Tensor) -> torch.Tensor:
        # Simplified feature attribution using magnitude of activations
        return torch.norm(intermediate, dim=-1)

    @staticmethod
    def representation_clarity(intermediate: torch.Tensor) -> torch.Tensor:
        # Reshape to handle sequence dimension
        batch_size, seq_len, hidden_dim = intermediate.shape
        flat = intermediate.view(-1, hidden_dim)
        
        # Compute clarity on flattened representation
        normed = F.normalize(flat, dim=-1)
        similarity = torch.matmul(normed, normed.transpose(-1, -2))
        mask = ~torch.eye(similarity.shape[-1], dtype=bool, device=similarity.device)
        clarity = -torch.mean(torch.abs(similarity[mask]))
        
        return clarity.expand(batch_size)

    @staticmethod
    def information_gain(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        delta = current - previous
        return torch.norm(delta, dim=-1) / (torch.norm(previous, dim=-1) + 1e-6)

    @staticmethod
    def decision_confidence(layer_output: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(layer_output, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return 1.0 - (entropy / torch.log(torch.tensor(probs.size(-1), device=probs.device)))

class MoETransformerWithReasoning(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 num_heads: int, num_classes: int, num_experts: int = 4, 
                 k: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.num_experts = num_experts
        self.k = k
        
        # MoE layers
        self.expert_layers = nn.ModuleList([
            nn.ModuleList([
                ExpertLayer(hidden_dim, hidden_dim * 4, dropout)
                for _ in range(num_experts)
            ]) for _ in range(num_layers)
        ])
        
        # Gates for expert routing
        self.gates = nn.ModuleList([
            nn.Linear(hidden_dim, num_experts) 
            for _ in range(num_layers)
        ])
        
        # Transformer attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) 
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.heuristics = ReasoningHeuristics()
        
    def forward_layer(self, x: torch.Tensor, prev_x: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, dict]:
        metrics = {}
        
        # Attention
        attention_out, _ = self.attention_layers[layer_idx](x, x, x)
        x = x + attention_out
        x = self.layer_norms[layer_idx](x)
        
        # MoE routing
        gate_logits = self.gates[layer_idx](x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Combine expert outputs
        combined_output = torch.zeros_like(x)
        for k in range(self.k):
            expert_idx = top_k_indices[..., k]
            prob = top_k_probs[..., k:k+1]
            for i, expert in enumerate(self.expert_layers[layer_idx]):
                mask = expert_idx == i
                if mask.any():
                    combined_output[mask] += prob[mask] * expert(x[mask])
        
        # Compute metrics
        if not self.training:
            logits = self.classifier(combined_output.mean(dim=1))
            metrics = {
                'layer': layer_idx,
                'feature_attribution': self.heuristics.feature_attribution(
                    combined_output, logits).mean().item(),
                'clarity': self.heuristics.representation_clarity(
                    combined_output).mean().item(),
                'confidence': self.heuristics.decision_confidence(
                    logits).mean().item(),
                'info_gain': self.heuristics.information_gain(
                    combined_output, prev_x).mean().item()
            }
                
        return combined_output, metrics
    
    def forward(self, x: torch.Tensor, beam_width: int = 3) -> Tuple[torch.Tensor, List[Dict], float]:
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_beam_search(x, beam_width)
            
    def forward_train(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict], float]:
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        
        prev_x = x
        layer_metrics = []
        quality_scores = []
        
        for i in range(len(self.attention_layers)):
            x, metrics = self.forward_layer(x, prev_x, i)
            if metrics:
                layer_metrics.append(metrics)
                quality_scores.append(sum(metrics.values()))
            prev_x = x
            
        x = x.mean(dim=1)
        logits = self.classifier(x)
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        return logits, layer_metrics, avg_quality
        
    def forward_beam_search(self, x: torch.Tensor, beam_width: int) -> Tuple[torch.Tensor, List[Dict], float]:
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        
        # Initialize beam with first state
        beam = [(x, 0.0, [])]  # (state, cumulative_score, metrics)
        
        for layer_idx in range(len(self.attention_layers)):
            candidates = []
            
            for state, cum_score, metrics_list in beam:
                # Generate expert combinations
                gate_logits = self.gates[layer_idx](state)
                probs = F.softmax(gate_logits, dim=-1)
                
                # Get top-k experts per position
                topk_probs, topk_idx = torch.topk(probs, self.k, dim=-1)
                
                # Try different expert combinations
                for k in range(self.k):
                    new_state, metrics = self.forward_layer(state, state, layer_idx)
                    quality_score = sum(metrics.values()) if metrics else 0
                    candidates.append((
                        new_state,
                        cum_score + quality_score,
                        metrics_list + [metrics] if metrics else metrics_list
                    ))
            
            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_width]
        
        # Select best final state
        best_state, best_score, best_metrics = beam[0]
        best_state = best_state.mean(dim=1)
        logits = self.classifier(best_state)
        
        return logits, best_metrics, best_score/len(self.attention_layers)

def load_data(dataset: str) -> Tuple[TensorDataset, List]:
    data_module = create_data_module(dataset_name=dataset, batch_size=32)
    data_loader, _ = data_module.setup()
    dataset = data_loader.dataset
    
    features = torch.stack([sample[0] for sample in dataset])
    labels = torch.stack([sample[1] for sample in dataset])
    
    scaler = StandardScaler()
    scaled_features = torch.tensor(
        scaler.fit_transform(features.numpy()), 
        dtype=torch.float32
    )
    
    scaled_dataset = TensorDataset(scaled_features, labels)
    targets = [sample[1].argmax(dim=0) for sample in dataset]
    
    return scaled_dataset, targets

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
          epochs: int, lr: float, device: torch.device) -> Tuple[nn.Module, float]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5)
    
    best_val_acc = -float('inf')
    best_model = None
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _, _ = model(x)
            loss = criterion(logits, torch.argmax(y, dim=1))
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, metrics, quality = model(x)
                _, predicted = torch.max(logits, 1)
                total += y.size(0)
                correct += (predicted == torch.argmax(y, dim=1)).sum().item()
        
        val_acc = 100 * correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
            
        scheduler.step(val_acc)
    
    if best_model:
        model.load_state_dict(best_model)
    return model, best_val_acc

def main(dataset: str = "part", num_runs: int = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = {"species": 2, "part": 7, "oil": 7, "cross-species": 3}
    
    results = []
    for run in range(num_runs):
        scaled_dataset, targets = load_data(dataset)
        n_splits = 3 if dataset == "part" else 5
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=run)
        
        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(scaled_dataset, targets)):
            print(f"Run {run+1}, Fold {fold+1}")
            train_dataset = Subset(scaled_dataset, train_idx)
            val_dataset = Subset(scaled_dataset, val_idx)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            model = MoETransformerWithReasoning(
                input_dim=2080,
                hidden_dim=256,
                num_layers=3,
                num_heads=4,
                num_classes=n_classes[dataset],
                num_experts=4,
                k=2
            )
            
            _, accuracy = train(
                model, train_loader, val_loader,
                epochs=100, lr=0.001, device=device
            )
            fold_results.append(accuracy)
            
        results.append(np.mean(fold_results))
        
    print(f"Final Results - {dataset}:")
    print(f"Mean Accuracy: {np.mean(results):.3f} ± {np.std(results):.3f}")
    return results

if __name__ == "__main__":
    main()