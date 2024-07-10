import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_size=1023, num_classes=7, dropout=0.5):
        super(CNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout),

            # nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.GELU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            # nn.Dropout(p=dropout),
        )
        
        self.flatten = nn.Flatten()

        # Calculate the size of the flattened features after convolutions
        # self.flat_features = 256 * (input_size // 2 // 2 // 2 // 2)
        self.flat_features = 128 * (input_size // 2 // 2 // 2)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_features, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        # Add channel dimension
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Flatten the output
        x = self.flatten(x)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        return x

# class CNN(nn.Module):
#     def __init__(self, input_size=1023, num_classes=10, dropout=0.2):
#         super(CNN, self).__init__()

#         # Convolutional Layer 1
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
#         self.dropout1 = nn.Dropout(p=dropout)
        
#         # Convolutional Layer 2
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
#         self.dropout2 = nn.Dropout(p=dropout)


#         # Fully connected layer
#         self.fc1 = nn.Linear(64  * (input_size // 2 // 2), num_classes)  # Adjust input size after pooling layers

#     def forward(self, x):
#         # Input shape: (batch_size, input_size)
#         # Reshape input to (batch_size, 1, input_size) for conv1d
#         x = x.unsqueeze(1)
        
#         x = self.pool1(F.gelu(self.conv1(x)))  # Apply conv1, ReLU, and pool
#         x = self.dropout1(x) # Dropout 1
#         x = self.pool2(F.gelu(self.conv2(x)))  # Apply conv2, ReLU, and pool
#         x = self.dropout2(x) # Dropout 2

#         # Flatten the output for the fully connected layer
#         x = x.view(-1, 64 * (1023 // 2 // 2))  # Adjust based on input size and pooling layers
#         x = self.fc1(x)  # Fully connected layer
#         return x