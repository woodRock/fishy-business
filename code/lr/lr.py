import torch
from torch import nn
import torch.nn.functional as F


# Define logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim, device='cuda'):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.device = device
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
    # Function to make predictions
    def predict_proba(self, x):
        x = torch.FloatTensor(x)
        x = x.to(self.device)
        with torch.no_grad():
            x = self.forward(x)
            probs = F.softmax(x, dim=-1).cpu().numpy()
            return probs