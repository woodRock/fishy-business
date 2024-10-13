import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from tqdm import tqdm

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=dropout, activation='gelu', norm_first=False),
            num_layers
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return self.dropout(x)

class ContrastiveModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes):
        super().__init__()
        self.encoder = TransformerEncoder(input_dim, d_model, nhead, num_layers, dim_feedforward)
        self.classifier = nn.Linear(d_model, num_classes)
        self.memory_bank = {'features': [], 'labels': []}

    def forward(self, x1, x2=None):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2) if x2 is not None else None
        return z1, z2

    def classify(self, x):
        z = self.encoder(x)
        return self.classifier(z)

    def update_memory(self, features, labels):
        self.memory_bank['features'].extend(features.cpu().numpy())
        self.memory_bank['labels'].extend(labels.cpu().numpy())

    def predict_from_memory(self, x):
        features_tensor = torch.tensor(self.memory_bank['features'], device=x.device)
        labels_tensor = torch.tensor(self.memory_bank['labels'], device=x.device)
        
        similarities = nn.functional.cosine_similarity(self.encoder(x).unsqueeze(1), features_tensor.unsqueeze(0), dim=2)
        _, max_indices = torch.max(similarities, dim=1)
        return labels_tensor[max_indices]

def contrastive_loss(z1, z2, y1, y2, temperature=0.5, alpha=0.0, beta=1.0):
    similarity = nn.functional.cosine_similarity(z1, z2)
    labels = torch.FloatTensor([int(torch.all(y1 == y2))])
    loss = labels * torch.pow(1 - similarity, 2) + (1 - labels) * torch.pow(torch.clamp(similarity - 0.1, min=0.0), 2)
    
    # Calculate balanced accuracy
    preds = (similarity > 0).float()
    balanced_accuracy = balanced_accuracy_score(y1.argmax(dim=1).cpu().numpy(), preds.cpu().numpy())
    
    # Combine contrastive loss and balanced accuracy (scaled appropriately)
    return alpha * loss.mean() + beta * (1 - balanced_accuracy) # Scale the balanced accuracy effect

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    all_preds = []
    all_labels = []
    total_loss = 0

    for x1, x2, y1, y2 in tqdm(dataloader, desc="Training"):
        x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)
        optimizer.zero_grad()
        z1, z2 = model(x1, x2)
        logits = model.classify(x1)

        preds = torch.argmax(logits, dim=1)
        loss = contrastive_loss(z1, z2, y1, y2)
        labels = torch.argmax(y1, dim=1)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        model.update_memory(z1.detach(), y1.detach())

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), accuracy, balanced_accuracy

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x1, x2, y1, y2 in tqdm(dataloader, desc="Evaluating"):
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)
            preds = model.predict_from_memory(x1)
            preds = torch.argmax(preds, dim=1)
            labels = torch.argmax(y1, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    return accuracy, balanced_accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from util import preprocess_dataset
    train_loader, val_loader = preprocess_dataset(dataset="instance-recognition", batch_size=64)

    input_dim = next(iter(train_loader))[0].shape[1]
    d_model = 512
    nhead = 4
    num_layers = 4
    dim_feedforward = 256
    num_classes = 24
    learning_rate = 1E-4

    model = ContrastiveModel(input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    num_epochs = 200
    best_val_accuracy = 0

    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_balanced_accuracy = train_epoch(model, train_loader, optimizer, device)
        val_accuracy, val_balanced_accuracy = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Train Balanced Accuracy: {train_balanced_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Balanced Accuracy: {val_balanced_accuracy:.4f}")
        
        if val_balanced_accuracy > best_val_accuracy:
            best_val_accuracy = val_balanced_accuracy
            torch.save(model.state_dict(), "best_model.pth")
        
        print("------------------------")

    print(f"Best Validation Balanced Accuracy: {best_val_accuracy:.4f}")

if __name__ == "__main__":
    main()
