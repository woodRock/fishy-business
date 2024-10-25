import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from scipy.linalg import null_space, orth
from util import preprocess_dataset

class TangentManifoldVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes, num_neighbors=5, device="cpu"):
        super(TangentManifoldVAE, self).__init__()
        
        # Dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim * 2  # Double the hidden dimension
        self.latent_dim = latent_dim * 2  # Double the latent dimension
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # VAE parameters
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dim, self.latent_dim)
        
        # Initialize with smaller values
        nn.init.kaiming_normal_(self.fc_mu.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.fc_var.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.zeros_(self.fc_var.bias)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, input_dim)
        )
        
        # Classifier with attention
        self.classifier_attention = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim // 2, num_classes)
        )
        
        self.num_neighbors = num_neighbors
        self.device = device
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = torch.clamp(self.fc_var(h), min=-10.0, max=2.0)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            std = torch.clamp(std, min=1e-4, max=10.0)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def decode(self, z):
        return torch.tanh(self.decoder(z))
    
    def compute_tangent_vectors(self, x, neighbors):
        batch_size = x.size(0)
        dists = torch.cdist(x, neighbors)
        _, indices = torch.topk(dists, k=self.num_neighbors, dim=1, largest=False)
        
        tangent_vectors = []
        for i in range(batch_size):
            neighbor_points = neighbors[indices[i]]
            vectors = neighbor_points - x[i].unsqueeze(0)
            vectors = F.normalize(vectors, dim=1)
            q, r = torch.linalg.qr(vectors.T)
            tangent_vectors.append(q.T)
            
        return torch.stack(tangent_vectors)
    
    def compute_latent_gradients(self, x, z):
        """
        Compute gradients of latent variables with respect to input
        """
        batch_size = x.size(0)
        latent_dim = z.size(1)
        gradients = []
        
        # Make sure input requires grad
        x.requires_grad_(True)
        
        for b in range(batch_size):
            sample_grads = []
            for i in range(latent_dim):
                # Zero out previous gradients
                x.grad = None
                
                # Create a zero tensor with a 1 at position i
                mask = torch.zeros_like(z)
                mask[b, i] = 1.0
                
                # Backward pass
                z.backward(gradient=mask, retain_graph=True)
                
                # Get the gradient
                if x.grad is not None:
                    grad = x.grad[b].clone().detach()
                    grad = torch.clamp(grad, min=-1.0, max=1.0)
                    sample_grads.append(grad)
                else:
                    # If no gradient, append zero tensor
                    sample_grads.append(torch.zeros_like(x[b]))
            
            if sample_grads:
                gradients.append(torch.stack(sample_grads, dim=0))
        
        return torch.stack(gradients) if gradients else torch.zeros(batch_size, latent_dim, x.size(1), device=x.device)

    def forward(self, x, neighbors=None):
        """
        Modified forward pass with proper gradient handling
        """
        # Clone input but keep gradient tracking
        x_input = x.clone()
        x_input.requires_grad_(True)
        
        # Add small noise during training for regularization
        if self.training:
            x_input = x_input + torch.randn_like(x_input) * 0.01
        
        # Encode
        mu, log_var = self.encode(x_input)
        
        # Sample latent vector
        z = self.reparameterize(mu, log_var)
        
        # Apply attention for classification
        z_att = z.unsqueeze(1)
        z_att, _ = self.classifier_attention(z_att, z_att, z_att)
        z_att = z_att.squeeze(1)
        
        # Decode
        x_recon = self.decode(z)
        
        # Classification logits
        logits = self.classifier(z_att)
        
        # Compute regularization if neighbors are provided
        reg_loss = torch.tensor(0.0, device=x.device)
        if neighbors is not None and self.training:
            reg_loss = self.tangent_space_regularization(x_input, z, neighbors)
        
        # Compute losses with improved stability
        recon_loss = F.mse_loss(x_recon, x)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = torch.clamp(kl_loss, max=10.0)
        
        return logits, x_recon, recon_loss, kl_loss, reg_loss

    def compute_tangent_vectors(self, x, neighbors):
        """
        Improved tangent vector computation with better numerical stability
        """
        batch_size = x.size(0)
        dists = torch.cdist(x, neighbors)
        _, indices = torch.topk(dists, k=min(self.num_neighbors, neighbors.size(0)), dim=1, largest=False)
        
        tangent_vectors = []
        for i in range(batch_size):
            neighbor_points = neighbors[indices[i]]
            vectors = neighbor_points - x[i].unsqueeze(0)
            
            # Normalize vectors with epsilon for numerical stability
            vectors = F.normalize(vectors + 1e-8, dim=1)
            
            # Compute QR decomposition
            try:
                q, r = torch.linalg.qr(vectors.T)
                tangent_vectors.append(q.T)
            except RuntimeError:
                # Fallback to simpler orthogonalization if QR fails
                vectors_normalized = F.normalize(vectors, dim=1)
                tangent_vectors.append(vectors_normalized)
        
        return torch.stack(tangent_vectors)

    def tangent_space_regularization(self, x, z, neighbors):
        """
        Modified regularization with improved stability
        """
        tangent_vectors = self.compute_tangent_vectors(x, neighbors)
        gradients = self.compute_latent_gradients(x, z)
        
        # Add small epsilon to prevent numerical instability
        epsilon = 1e-8
        tangent_vectors = tangent_vectors + epsilon
        
        # Compute projections
        proj_gradients = torch.matmul(gradients, tangent_vectors.transpose(-2, -1))
        proj_gradients = torch.matmul(proj_gradients, tangent_vectors)
        
        # Compute regularization loss with stability
        reg_loss = torch.mean((gradients - proj_gradients).pow(2))
        return torch.clamp(reg_loss, min=0.0, max=10.0)

def train_step(model, optimizer, train_loader, neighbors, reg_weight=0.1, beta=0.01):
    model.train()
    total_loss = 0
    running_accuracy = []
    
    # Adaptive loss weights based on training progress
    cls_weight = min(1.0, len(running_accuracy) / 1000) if running_accuracy else 0.1
    
    for x, y in train_loader:
        optimizer.zero_grad()
        
        x, y = x.to(model.device), y.to(model.device)
        
        # Forward pass
        logits, x_recon, recon_loss, kl_loss, reg_loss = model(x, neighbors)
        
        # Focal loss for classification
        cls_loss = focal_loss(logits, y, gamma=2.0)
        
        # Dynamic loss weighting
        total_loss = (
            cls_weight * cls_loss +
            (1.0 - cls_weight) * recon_loss +
            beta * kl_loss +
            reg_weight * reg_loss
        )
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Skip step if gradients explode
        if torch.isfinite(grad_norm):
            optimizer.step()
        
        # Update running accuracy
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == y.argmax(dim=1)).float().mean()
            running_accuracy.append(acc.item())
            
    return total_loss.item(), cls_loss.item(), recon_loss.item(), kl_loss.item(), reg_loss.item()

def focal_loss(logits, targets, gamma=2.0):
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** gamma * ce_loss).mean()
    return focal_loss

def evaluate_step(model, val_loader):
    """
    Evaluation step with proper model mode
    """
    model.eval()
    total_cls_loss = 0.0
    total_recon_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(model.device), y.to(model.device)
            
            logits, x_recon, recon_loss, _, _ = model(x)
            cls_loss = F.cross_entropy(logits, y)
            
            pred = logits.argmax(dim=1)
            actual = y.argmax(dim=1)
            correct = (pred == actual).sum().item()
            
            total_cls_loss += cls_loss.item()
            total_recon_loss += recon_loss.item()
            total_correct += correct
            total_samples += x.size(0)
            
    return (total_cls_loss / len(val_loader),
            total_recon_loss / len(val_loader),
            total_correct / total_samples)

def generate_classifier_neighbors(point, label, n_neighbors=20, inner_radius=0.1, outer_radius=0.2):
    """
    Generate classifier neighbors for tangent space regularization
    """
    neighbors = []
    labels = []
    
    for i in range(n_neighbors):
        # Randomly sample radius
        radius = np.random.uniform(inner_radius, outer_radius)
        angle = np.random.uniform(0, 2 * np.pi)
        
        # Generate random point
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        # Perturb point
        perturbed = point + torch.tensor(x, device=point.device)
        neighbors.append(perturbed)
        labels.append(label)
        
    return torch.stack(neighbors), torch.stack(labels)


if __name__ == "__main__":
    # Hyperparameters
    input_dim = 1023
    hidden_dim = 256
    latent_dim = 32
    num_classes = 2
    batch_size = 32  # Reduced batch size
    epochs = 100
    beta = 0.001  # Further reduced beta for KL loss
    reg_weight = 0.01  # Reduced regularization weight
    
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TangentManifoldVAE(input_dim, hidden_dim, latent_dim, num_classes, device=device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    train_loader, val_loader = preprocess_dataset(
        dataset="species",
        is_data_augmentation=True,  # Enable data augmentation
        batch_size=batch_size,
        is_pre_train=False
    )
    
    # neighbors = torch.randn(batch_size * 5, input_dim).to(device)
    neighbors = [] 
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        point = x
        label = y
        n, l = generate_classifier_neighbors(point, label)
        print(f"Neigbour type: {type(n)}")
        neighbors.extend(n)
    neighbors = torch.cat(neighbors, dim=0)
    print(f"neighbors shape: {neighbors.shape}")
    
    best_val_acc = 0.0  
    patience = 0
    max_patience = 10
    
    for epoch in range(epochs):
        total_loss, cls_loss, recon_loss, kl_loss, reg_loss = train_step(
            model, optimizer, train_loader, neighbors, reg_weight, beta
        )
        
        train_cls_loss, train_recon_loss, train_acc = evaluate_step(model, train_loader)
        val_cls_loss, val_recon_loss, val_acc = evaluate_step(model, val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_cls_loss)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
        else:
            patience += 1
            
        if patience >= max_patience:
            print("Early stopping triggered")
            print(f"Best validation accuracy: {best_val_acc}")
            # break
            
        print(f"Epoch {epoch}")
        print(f"Total Loss: {total_loss:.4f}, Classification Loss: {cls_loss:.4f}")
        print(f"Reconstruction Loss: {recon_loss:.4f}, KL Loss: {kl_loss:.4f}")
        print(f"Regularization Loss: {reg_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}\n")