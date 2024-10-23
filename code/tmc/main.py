import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from util import preprocess_dataset

class TangentManifoldVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes, num_neighbors=5, device="cpu"):
        super(TangentManifoldVAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_neighbors = num_neighbors
        self.device = device
        
        # Encoder - Added batch normalization and dropout
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # VAE parameters with smaller initialization
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Initialize weights with smaller values
        nn.init.xavier_normal_(self.fc_mu.weight, gain=0.01)
        nn.init.xavier_normal_(self.fc_var.weight, gain=0.01)
        
        # Decoder - Added batch normalization and dropout
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Classifier head with batch norm
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def encode(self, x):
        hidden = self.encoder(x)
        # Clip values to prevent extreme variances
        mu = self.fc_mu(hidden)
        log_var = torch.clamp(self.fc_var(hidden), min=-4.0, max=4.0)
        return mu, log_var
        
    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            # Add small epsilon to prevent division by zero
            std = torch.clamp(std, min=1e-3)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
        
    def decode(self, z):
        return self.decoder(z)
    
    def compute_tangent_vectors(self, x, neighbors):
        batch_size = x.size(0)
        dists = torch.cdist(x, neighbors)
        _, indices = torch.topk(dists, k=self.num_neighbors, dim=1, largest=False)
        
        tangent_vectors = []
        for i in range(batch_size):
            neighbor_points = neighbors[indices[i]]
            vectors = neighbor_points - x[i].unsqueeze(0)
            # Add small epsilon to prevent numerical instability
            vectors = vectors + 1e-6
            q, r = torch.linalg.qr(vectors.T, mode='reduced')
            tangent_vectors.append(q.T)
            
        return torch.stack(tangent_vectors)
    
    def compute_latent_gradients(self, x, z):
        gradients = []
        batch_size = x.size(0)
        
        x.requires_grad_(True)
            
        for b in range(batch_size):
            sample_grads = []
            for i in range(z.size(1)):
                if x.grad is not None:
                    x.grad.zero_()
                    
                z[b, i].backward(retain_graph=True)
                
                if x.grad is not None:
                    # Clip gradients to prevent explosion
                    grad = torch.clamp(x.grad[b].clone(), min=-1.0, max=1.0)
                    sample_grads.append(grad)
                    
            gradients.append(torch.stack(sample_grads, dim=0))
            
        return torch.stack(gradients)
    
    def tangent_space_regularization(self, x, z, neighbors):
        tangent_vectors = self.compute_tangent_vectors(x, neighbors)
        gradients = self.compute_latent_gradients(x, z)
        
        # Add small epsilon to prevent numerical instability
        tangent_vectors = tangent_vectors + 1e-6
        
        proj_gradients = torch.matmul(gradients, tangent_vectors.transpose(-2, -1))
        proj_gradients = torch.matmul(proj_gradients, tangent_vectors)
        
        reg_loss = torch.mean((gradients - proj_gradients).pow(2))
        
        return torch.clamp(reg_loss, max=10.0)  # Prevent extreme regularization
    
    def compute_kl_loss(self, mu, log_var):
        kl_div = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = torch.clamp(kl_div.mean(), max=10.0)  # Prevent extreme KL loss
        return kl_loss
    
    def forward(self, x, neighbors=None):
        x = x.clone().detach().requires_grad_(True)
        
        # Encode
        mu, log_var = self.encode(x)
        
        # Sample latent vector
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x_recon = self.decode(z)
        
        # Classification logits
        logits = self.classifier(z)
        
        # Compute regularization if neighbors are provided
        reg_loss = torch.tensor(0.0, device=x.device)
        if neighbors is not None and self.training:
            reg_loss = self.tangent_space_regularization(x, z, neighbors)
            
        # Compute VAE losses
        recon_loss = F.mse_loss(x_recon, x)
        kl_loss = self.compute_kl_loss(mu, log_var)
            
        return logits, x_recon, recon_loss, kl_loss, reg_loss

def train_step(model, optimizer, train_loader, neighbors, reg_weight=0.01, beta=0.001):
    """
    Single training step with adjusted loss weights and gradient clipping
    """
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_reg_loss = 0
    num_batches = 0
    
    for x, y in train_loader:
        optimizer.zero_grad()
        
        x, y = x.to(model.device), y.to(model.device)
        
        # Forward pass
        logits, x_recon, recon_loss, kl_loss, reg_loss = model(x, neighbors)
        
        # Compute classification loss with label smoothing
        cls_loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        
        # Scale losses for better balance
        scaled_recon_loss = recon_loss
        scaled_kl_loss = beta * kl_loss
        scaled_reg_loss = reg_weight * reg_loss
        
        # Total loss
        batch_loss = cls_loss + scaled_recon_loss + scaled_kl_loss + scaled_reg_loss
        
        # Backward pass with gradient clipping
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += batch_loss.item()
        total_cls_loss += cls_loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_reg_loss += reg_loss.item()
        num_batches += 1
    
    # Return average losses
    return (total_loss / num_batches,
            total_cls_loss / num_batches,
            total_recon_loss / num_batches,
            total_kl_loss / num_batches,
            total_reg_loss / num_batches)

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
    
    neighbors = torch.randn(batch_size * 5, input_dim).to(device)
    
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
            break
            
        print(f"Epoch {epoch}")
        print(f"Total Loss: {total_loss:.4f}, Classification Loss: {cls_loss:.4f}")
        print(f"Reconstruction Loss: {recon_loss:.4f}, KL Loss: {kl_loss:.4f}")
        print(f"Regularization Loss: {reg_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}\n")