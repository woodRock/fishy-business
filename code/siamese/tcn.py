import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from tqdm import tqdm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # Remove the incorrect LayerNorm implementation
        # Instead, use normalization after the convolution
        
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Use 1D batch normalization instead of layer normalization
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.bn2 = nn.BatchNorm1d(n_outputs)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight.data)
        nn.init.kaiming_normal_(self.conv2.weight.data)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight.data)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                   stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size,
                                   dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNEncoder(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.3):
        super(TCNEncoder, self).__init__()
        
        # Modified architecture with better stability
        num_channels = [32, 64, 64]  # Reduced complexity
        kernel_size = 3
        
        # Remove the problematic input_norm layer and use BatchNorm1d instead
        self.input_norm = nn.BatchNorm1d(1)  # For normalizing the channel dimension
        
        self.tcn = TemporalConvNet(
            num_inputs=1,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        self.adaptive_pool = nn.AdaptiveMaxPool1d(4)
        self.flatten = nn.Flatten()
        
        # Added batch normalization and modified FC layers
        self.fc_layers = nn.Sequential(
            nn.Linear(num_channels[-1] * 4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input shape: [batch_size, sequence_length]
        x = x.unsqueeze(1)  # Add channel dimension: [batch_size, 1, sequence_length]
        x = self.input_norm(x)  # Normalize along the channel dimension
        x = self.tcn(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

class ContrastiveModel(nn.Module):
    def __init__(self, input_channels, d_model, num_classes):
        super().__init__()
        self.encoder = TCNEncoder(input_channels, d_model)
        
        # Added batch normalization before classifier
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Initialize classifier weights
        nn.init.kaiming_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x1, x2=None):
        z1 = self.encoder(x1)
        z1 = self.batch_norm(z1)
        
        if x2 is not None:
            z2 = self.encoder(x2)
            z2 = self.batch_norm(z2)
            return z1, z2
        return z1

    def classify(self, x):
        z = self.encoder(x)
        z = self.batch_norm(z)
        logits = self.classifier(z)
        # Add logit scaling to prevent extreme values
        return logits * 0.1  # Scale down logits

# Loss functions remain the same as before
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
        
        # Ensure y is a 1D tensor
        y = y.squeeze()
        
        similarity = nn.functional.cosine_similarity(z1, z2)
        preds = (similarity > 0.5).float()
        
        # Handle cases where all samples are positive or all are negative
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
            
            loss = combined_loss(z1, z2, y, preds, anchor, positive, negative, logits, alpha, beta, gamma, delta)
        
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