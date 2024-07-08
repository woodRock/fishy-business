import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from util import preprocess_dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Iterable


class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int
        ) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, 
                                     Q: torch.Tensor, 
                                     K: torch.Tensor, 
                                     V: torch.Tensor, 
                                     mask: torch.Tensor = None
        ) -> torch.Tensor:
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, 
                    x: torch.Tensor
        ) -> torch.Tensor:
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, 
                      x: torch.Tensor
        ) -> torch.Tensor:
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor, 
                mask: torch.Tensor = None
        ) -> torch.Tensor:
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class LSTMModel(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 n_layers: int, 
                 dropout: float, 
                 num_heads: int = 4
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=num_heads, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, 
                x: torch.Tensor
        ) -> torch.Tensor:
        output, _ = self.lstm(x)
        attention_output, _ = self.attention(output, output, output)
        context_vector = attention_output[:, -1, :]
        x = self.dropout(context_vector)
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return self.fc2(x)


class ResidualLSTMModel(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 n_layers: int, 
                 dropout: float, 
                 num_heads: int
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_dim)
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(hidden_dim, hidden_dim, batch_first=True) for _ in range(n_layers)
        ])
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                x: torch.Tensor
    ) -> torch.Tensor:
        x = self.input_proj(x)
        for i, (lstm, layer_norm) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            residual = x
            x, _ = lstm(x)
            x = self.dropout(x)
            x = layer_norm(x + residual)  # Residual connection
        # Apply multi-head attention
        attn_output = self.attention(x, x, x)
        # Residual connection after attention
        x = layer_norm(x + attn_output)
        # Use the last time step as the final representation
        context_vector = x[:, -1, :]
        return self.fc(context_vector)
    
class StackedLSTMWithResidual(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super(StackedLSTMWithResidual, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        # LSTM layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(hidden_size, hidden_size, batch_first=True) 
            for _ in range(num_layers)
        ])
        # Layer normalization for each LSTM layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) 
            for _ in range(num_layers)
        ])
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        # Process through LSTM layers with residual connections
        for i in range(self.num_layers):
            residual = x
            lstm_out, _ = self.lstm_layers[i](x)
            x = self.layer_norms[i](lstm_out + residual)  # Residual connection
            x = self.dropout(x)
        # Use the last time step's output
        x = x[:, -1, :]
        # Output layer
        output = self.fc(x)
        return output
    
class FocalLoss(nn.Module):
    def __init__(self, 
                 alpha: int = 1, 
                 gamma: int = 2
    ) -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, 
                inputs: torch.Tensor, 
                targets: torch.Tensor
    ) -> float:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

if __name__ == "__main__":
    # Handle the command line arguments for the script.
    parser = argparse.ArgumentParser(
                    prog='Transformer',
                    description='A transformer for fish species classification.',
                    epilog='Implemented in pytorch and written in python.')
    parser.add_argument('-f', '--file-path', type=str, default="transformer_checkpoint",
                        help="Filepath to store the model checkpoints to. Defaults to transformer_checkpoint")
    parser.add_argument('-d', '--dataset', type=str, default="species",
                        help="The fish species or part dataset. Defaults to species")
    parser.add_argument('-r', '--run', type=int, default=0)
    parser.add_argument('-o', '--output', type=str, default=f"logs/results")

    # Preprocessing
    parser.add_argument('-da', '--data-augmentation',
                    action='store_true', default=False,
                    help="Flag to perform data augmentation. Defaults to False.")  
    # Pre-training
    # parser.add_argument('-msm', '--masked-spectra-modelling',
    #                 action='store_true', default=False,
    #                 help="Flag to perform masked spectra modelling. Defaults to False.")  
    # parser.add_argument('-nsp', '--next-spectra-prediction',
    #                 action='store_true', default=False,
    #                 help="Flag to perform next spectra prediction. Defaults to False.") 
    # Regularization
    parser.add_argument('-es', '--early-stopping', type=int, default=10,
                        help='Early stopping patience. To disable early stopping set to the number of epochs. Defaults to 5.')
    parser.add_argument('-do', '--dropout', type=float, default=0.2,
                        help="Probability of dropout. Defaults to 0.2")
    parser.add_argument('-ls', '--label-smoothing', type=float, default=0.1,
                        help="The alpha value for label smoothing. 1 is a uniform distribution, 0 is no label smoothing. Defaults to 0.1")
    # Hyperparameters
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help="The number of epochs to train the model for.")
    parser.add_argument('-lr', '--learning-rate', type=float, default=1E-5,
                        help="The learning rate for the model. Defaults to 1E-5.")
    parser.add_argument('-bs', '--batch-size', type=int, default=64,
                        help='Batch size for the DataLoader. Defaults to 64.')
    parser.add_argument('-is', '--input-size', type=int, default=1,
                        help='The number of layers. Defaults to 1.')
    parser.add_argument('-l', '--num-layers', type=int, default=2,
                        help='The number of layers. Defaults to 2.')
    parser.add_argument('-hd', '--hidden-dimension', type=int, default=128,
                        help='The number of hidden layer dimensions. Defaults to 128.')
    parser.add_argument('-nh', '--num-heads', type=int, default=4,
                        help='The number of heads for multi-head attention. Defaults to 4.')

    args = vars(parser.parse_args())

    # Logging output to a file.
    logger = logging.getLogger(__name__)
    # Run argument for numbered log files.
    output = f"{args['output']}_{args['run']}.log"
    # Filemode is write, so it clears the file, then appends output.
    logging.basicConfig(filename=output, level=logging.INFO, filemode='w')
    file_path = f"checkpoints/{args['file_path']}_{args['run']}.pth"
    dataset = args['dataset']

    # Preprocessing
    is_data_augmentation = args['data_augmentation'] # @param {type:"boolean"}
    # Pretraining
    # is_next_spectra = args['next_spectra_prediction'] # @param {type:"boolean"}
    # is_masked_spectra = args['masked_spectra_modelling'] # @param {type:"boolean"}
    # Regularization
    is_early_stopping = args['early_stopping'] is not None # @param {type:"boolean"}
    patience = args['early_stopping']
    dropout = args['dropout']
    label_smoothing = args['label_smoothing']
    # Hyperparameters
    num_epochs = args['epochs']
    input_dim = 1023
    num_heads = args['num_heads']
    learning_rate = args['learning_rate']
    batch_size = args['batch_size']
    input_size = args['input_size']  # Each time step is a single value
    n_layers = args['num_layers']# 2-5 layers
    hidden_dim = args['hidden_dimension']

    num_classes_per_dataset = {"species": 2, "part": 6, "oil": 7, "oil_simple": 2, "cross-species": 3}
    if dataset not in num_classes_per_dataset.keys():
        raise ValueError(f"Invalid dataset: {dataset} not in {num_classes_per_dataset.keys()}")
    output_dim = num_classes_per_dataset[dataset]

    # Prepare data
    train_loader, val_loader = preprocess_dataset(
        dataset=dataset, 
        is_data_augmentation=False,
        is_pca=True,
        batch_size=batch_size,
        is_pre_train=False
    )

    # model = LSTMModel(
    #     input_size=input_size,
    #     hidden_dim=hidden_dim,
    #     output_dim=output_dim,
    #     n_layers=n_layers,
    #     dropout=dropout,
    #     num_heads=4  # You can adjust this number
    # )

    # # Initialize the model
    # model = ResidualLSTMModel(
    #     input_size=input_size,
    #     hidden_dim=hidden_dim,
    #     output_dim=output_dim,
    #     n_layers=n_layers,
    #     dropout=dropout,
    #     num_heads=4  # Adjust as needed
    # )

    model = StackedLSTMWithResidual(input_size, hidden_dim, n_layers, output_dim, dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.5)
    is_focal = False
    if is_focal:
        criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # Adaptive learning rate.
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)

    def calculate_accuracy(
            predictions: Iterable,
            labels: Iterable
    ) -> float:
        _, predicted = torch.max(predictions, 1)
        _, actual = torch.max(labels, 1)
        correct = (predicted == actual).float().sum()
        accuracy = correct / len(labels)
        return accuracy.item()

    # Training loop
    def train(model, data_loader, optimizer, criterion):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        for (x, y) in data_loader:
            spectra, labels = x.to(device), y.to(device) 
            optimizer.zero_grad()
            spectra = spectra.unsqueeze(2)  # Add channel dimension: [batch_size, 1023, 1]
            predictions = model(spectra)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_accuracy += calculate_accuracy(predictions, labels)
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        return epoch_loss / len(data_loader), epoch_accuracy / len(data_loader)

    # Validation loop
    def evaluate(model, data_loader, criterion):
        model.eval()
        epoch_loss = 0
        epoch_accuracy = 0
        with torch.no_grad():
            for (x,y) in data_loader:
                spectra, labels = x.to(device), y.to(device) 
                spectra = spectra.unsqueeze(2)  # Add channel dimension: [batch_size, 1023, 1]
                predictions = model(spectra)
                loss = criterion(predictions, labels)
                epoch_loss += loss.item()
                epoch_accuracy += calculate_accuracy(predictions, labels)
        return epoch_loss / len(data_loader), epoch_accuracy / len(data_loader)

    # Train the model
    n_epochs = 1_000
    for epoch in (pbar := tqdm(range(n_epochs), desc="Training")):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        # Adpative learning rate.
        scheduler.step(val_acc)
        message = f'Epoch: {epoch+1:02} \t | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f} \t | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}'
        pbar.set_description(message)