import argparse
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from util import preprocess_dataset


class LSTM(nn.Module):
    def __init__(self, input_size=1023, hidden_size=128, num_layers=2, output_size=2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # We need to unsqueeze the input to add a sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# Handle the command line arguments for the script.
parser = argparse.ArgumentParser(
                prog='Long-short Term Memory (LSTM) Recurrent neural network',
                description='An LSTM for fish species classification.',
                epilog='Implemented in pytorch and written in python.')
parser.add_argument('-f', '--file-path', type=str, default="LSTM_checkpoint",
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
# 
# Regularization
parser.add_argument('-es', '--early-stopping', type=int, default=10,
                    help='Early stopping patience. To disable early stopping set to the number of epochs. Defaults to 10.')
parser.add_argument('-do', '--dropout', type=float, default=0.2,
                    help="Probability of dropout. Defaults to 0.2")
parser.add_argument('-ls', '--label-smoothing', type=float, default=0.1,
                    help="The alpha value for label smoothing. 1 is a uniform distribution, 0 is no label smoothing. Defaults to 0.1")
# Hyperparameters
parser.add_argument('-e', '--epochs', type=int, default=100,
                    help="The number of epochs to train the model for.")
parser.add_argument('-lr', '--learning-rate', type=float, default=1E-4,
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
logger.info(f"Dataset: {dataset}")

# Example usage
dataset = args['dataset']
input_size = 1023  # Number of features in mass spectrometry data
hidden_size = args['hidden_dimension']  # Number of features in hidden state
num_layers = args['num_layers']     # Number of stacked LSTM layers
num_epochs = args['epochs']
learning_rate = args['learning_rate']
label_smoothing = args['label_smoothing']
num_classes_per_dataset = {"species": 2, "part": 6, "oil": 7, "oil_simple": 2, "cross-species": 3}
if dataset not in num_classes_per_dataset.keys():
    raise ValueError(f"Invalid dataset: {dataset} not in {num_classes_per_dataset.keys()}")
output_size = num_classes_per_dataset[dataset]


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_val_acc = float('-inf')
    epochs_without_improvement = 0
    best_model = None

    for epoch in (pbar := tqdm(range(num_epochs), desc="Training")):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            _, actual = labels.max(1)
            train_correct += predicted.eq(actual).sum().item()
            train_total += labels.size(0)

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                _, actual = labels.max(1)
                val_correct += predicted.eq(actual).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total

        message = f'Epoch {epoch+1}/{num_epochs} \tTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\t Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}'
        pbar.set_description(message)
        logger.info(message)

        # Early stopping
        if train_acc == 1:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                best_model = model.state_dict()
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    message = f'Early stopping triggered after {epoch + 1} epochs'
                    logger.info(message)
                    print(message)
                    print(f"Validation accuracy: {best_val_acc}")
                    break

    if best_model is not None:
        model.load_state_dict(best_model)
    return model

model = LSTM(
    input_size=1023,
    hidden_size=128,
    num_layers=2,
    output_size=output_size
)

# Specify the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Label smoothing (Szegedy 2016)
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
# AdamW (Loshchilov 2017)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

train_loader, val_loader = preprocess_dataset(
    dataset=dataset,
    is_data_augmentation=False,
    batch_size=64,
    is_pca=False,
    is_pre_train=False)

# Modify the train_model call to include the early stopping patience
trained_model = train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer,
    num_epochs=num_epochs,
    patience=args['early_stopping']
)

# Save the best model
torch.save(trained_model.state_dict(), file_path)
logger.info(f"Best model saved to {file_path}")