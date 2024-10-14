import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import preprocess_dataset

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=2, num_heads=4, dim_feedforward=512, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        self.embedding = nn.Linear(input_dim, dim_feedforward)
        self.pos_encoder = nn.Linear(dim_feedforward, dim_feedforward)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=num_heads, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(dim_feedforward, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = x.unsqueeze(1)  # Add sequence length dimension
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x

def train(model, train_loader, val_loader, num_epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = batch
            inputs = inputs.to(device).float()  # Convert to float32
            labels = labels.to(device).float()  # Ensure labels are long (int64)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            _, actual = labels.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(actual).sum().item()
        
        train_accuracy = train_correct / train_total
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        print()

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs = inputs.to(device).float()  # Convert to float32
            labels = labels.to(device).float()  # Ensure labels are long (int64)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            _, actual = labels.max(1)
            total += labels.size(0)
            correct += predicted.eq(actual).sum().item()
    
    accuracy = correct / total
    return total_loss / len(data_loader), accuracy

def main():
    # Hyperparameters
    input_dim = 1023  # Adjust based on your data
    num_classes = 24  # Adjust based on your classification task
    num_epochs = 50
    learning_rate = 0.001
    batch_size = 64
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load your data (replace with your actual data loading code)
    train_loader, val_loader = preprocess_dataset(dataset="instance-recognition", batch_size=batch_size)
    
    # Initialize the model
    model = TransformerClassifier(input_dim, num_classes).to(device)
    
    # Train the model
    train(model, train_loader, val_loader, num_epochs, learning_rate, device)
    
    # Final evaluation
    criterion = nn.CrossEntropyLoss()
    val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
    print(f"Final Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    main()