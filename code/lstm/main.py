import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from util import preprocess_dataset
from lstm_with_attention import LSTM
from train import train_model, evaluate_model

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

model = LSTM(
    input_size=input_size,
    hidden_size=args['hidden_dimension'],
    num_layers=args['num_layers'],
    output_size=output_size,
    dropout=args['dropout']
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
    is_fft=False,
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

evaluate_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        dataset=dataset, 
        device=device
)