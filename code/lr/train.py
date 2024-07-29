from torch import nn, optim
from lr import LogisticRegression


def train_model(
        model: LogisticRegression, 
        train_loader, 
        val_loader, 
        criterion: nn.Module, 
        optimizer: optim.Optimizer,
        epochs: int = 1_000,
        device: str = 'cuda'
    ) -> nn.Module:
    # Train the model
    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        val_loss = 0

        model.eval()
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            outputs = model(x)
            val_loss = criterion(outputs, y)
        
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    return model