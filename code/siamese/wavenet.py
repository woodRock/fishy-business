import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from tqdm import tqdm

class CausalConv1d(nn.Module):
    """
    Causal 1D convolution with dilations
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation, **kwargs
        ))
        
    def forward(self, x):
        x = self.conv(x)
        if self.padding != 0:
            return x[:, :, :-self.padding]
        return x

class ResidualBlock(nn.Module):
    """
    Residual block with skip connections and gated activation
    """
    def __init__(self, channels, kernel_size, dilation, dropout=0.2):
        super(ResidualBlock, self).__init__()
        
        self.dilated_conv = CausalConv1d(
            channels, 2 * channels, kernel_size, 
            dilation=dilation
        )
        self.conv_1x1 = weight_norm(nn.Conv1d(channels, channels, 1))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        original = x
        
        # Dilated convolution
        x = self.dilated_conv(x)
        
        # Gated activation unit (similar to the original WaveNet paper)
        filter_x, gate_x = torch.chunk(x, 2, dim=1)
        x = torch.tanh(filter_x) * torch.sigmoid(gate_x)
        
        # 1x1 convolution
        x = self.conv_1x1(x)
        x = self.dropout(x)
        
        # Residual and skip connections
        residual = original + x
        
        return residual, x

class WaveNetEncoder(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.2):
        super(WaveNetEncoder, self).__init__()
        
        # Initial causal convolution
        self.causal_conv = CausalConv1d(1, 32, kernel_size=2)
        
        # Hyperparameters
        self.n_layers = 8
        self.n_blocks = 3
        self.channels = 32
        self.kernel_size = 2
        
        # Build the WaveNet blocks
        self.blocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            for i in range(self.n_layers):
                dilation = 2 ** i
                self.blocks.append(
                    ResidualBlock(
                        channels=self.channels,
                        kernel_size=self.kernel_size,
                        dilation=dilation,
                        dropout=dropout
                    )
                )
        
        # Final processing
        self.adaptive_pool = nn.AdaptiveMaxPool1d(4)
        self.final_conv = weight_norm(nn.Conv1d(self.channels, self.channels, 1))
        self.relu = nn.ReLU()
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.channels * 4, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # Add channel dimension if not present
        x = x.unsqueeze(1)
        
        # Initial causal convolution
        x = self.causal_conv(x)
        
        # WaveNet blocks
        skip_connections = []
        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        
        # Sum skip connections
        x = torch.stack(skip_connections).sum(dim=0)
        x = self.relu(x)
        
        # Final processing
        x = self.final_conv(x)
        x = self.relu(x)
        x = self.adaptive_pool(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        
        return x

class ContrastiveModel(nn.Module):
    def __init__(self, input_channels, d_model, num_classes):
        super().__init__()
        self.encoder = WaveNetEncoder(input_channels, d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x1, x2=None):
        z1 = self.encoder(x1)
        if x2 is not None:
            z2 = self.encoder(x2)
            return z1, z2
        return z1

    def classify(self, x):
        z = self.encoder(x)
        return self.classifier(z)

# Loss functions
def contrastive_loss(z1, z2, y, temperature=0.5):
    similarity = nn.functional.cosine_similarity(z1, z2)
    loss = y * torch.pow(1 - similarity, 2) + (1 - y) * torch.pow(torch.clamp(similarity - 0.1, min=0.0), 2)
    return loss.mean()

def triplet_loss(anchor, positive, negative, margin=1.0):
    distance_positive = (anchor - positive).pow(2).sum(1)
    distance_negative = (anchor - negative).pow(2).sum(1)
    losses = torch.relu(distance_positive - distance_negative + margin)
    return losses.mean()

def balanced_accuracy_loss(preds, labels):
    ba = balanced_accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    return 1 - ba

def combined_loss(z1, z2, y, preds, anchor, positive, negative, logits, 
                 alpha=0.3, beta=0.3, gamma=0.2, delta=0.2, temperature=0.5, margin=1.0):
    cl = contrastive_loss(z1, z2, y, temperature)
    tl = triplet_loss(anchor, positive, negative, margin)
    ce = nn.functional.cross_entropy(logits, y.long())
    bal = balanced_accuracy_loss(preds, y)
    return alpha * cl + beta * tl + gamma * ce + delta * bal

def train_epoch(model, dataloader, optimizer, device, alpha=0.3, beta=0.3, gamma=0.2, delta=0.2):
    model.train()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    for x1, x2, y in tqdm(dataloader, desc="Training"):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        
        z1, z2 = model(x1, x2)
        logits = model.classify(x1)
        
        y = y.squeeze()
        similarity = nn.functional.cosine_similarity(z1, z2)
        preds = (similarity > 0.5).float()
        
        if torch.all(y == 1) or torch.all(y == 0):
            loss = combined_loss(z1, z2, y, preds, z1, z1, z1, logits, alpha, 0, gamma, delta)
        else:
            positive_indices = torch.where(y == 1)[0]
            negative_indices = torch.where(y == 0)[0]
            
            num_triplets = min(len(positive_indices), len(negative_indices), len(z1))
            anchor_indices = torch.randperm(len(z1))[:num_triplets]
            
            anchor = z1[anchor_indices]
            positive = z2[positive_indices[torch.randperm(len(positive_indices))[:num_triplets]]]
            negative = z2[negative_indices[torch.randperm(len(negative_indices))[:num_triplets]]]
            
            loss = combined_loss(z1, z2, y, preds, anchor, positive, negative, logits, 
                               alpha, beta, gamma, delta)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    loss = total_loss / len(dataloader)
    return loss, accuracy, balanced_accuracy

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x1, x2, y in tqdm(dataloader, desc="Evaluating"):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            z1, z2 = model(x1, x2)
            similarity = nn.functional.cosine_similarity(z1, z2)
            preds = (similarity > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    return accuracy, balanced_accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from util import preprocess_dataset
    train_loader, val_loader = preprocess_dataset(dataset="instance-recognition", batch_size=64)

    input_channels = 1
    d_model = 128
    num_classes = 2

    model = ContrastiveModel(input_channels, d_model, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    num_epochs = 200
    best_val_accuracy = 0
    alpha, beta, gamma, delta = 0.8, 0.0, 0.2, 0.0
    patience = 20
    initial_patience = patience

    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_balanced_accuracy = train_epoch(
            model, train_loader, optimizer, device, alpha, beta, gamma, delta)
        val_accuracy, val_balanced_accuracy = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Train Balanced Accuracy: {train_balanced_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Balanced Accuracy: {val_balanced_accuracy:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_balanced_accuracy > best_val_accuracy:
            best_val_accuracy = val_balanced_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            patience = initial_patience
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping triggered")
        
        print("------------------------")

    print(f"Best Validation Balanced Accuracy: {best_val_accuracy:.4f}")

if __name__ == "__main__":
    main()