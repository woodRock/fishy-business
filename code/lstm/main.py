import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from lstm import LSTM
from util import preprocess_dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='Transformer',
                    description='A transformer for fish species classification.',
                    epilog='Implemented in pytorch and written in python.')
    parser.add_argument('-d', '--dataset', type=str, default="species",
                         help="The fish species or part dataset. Defaults to species")
    parser.add_argument('-f', '--file-path', type=str, default="cnn_checkpoint",
                        help="Filepath to store the model checkpoints to. Defaults to transformer_checkpoint")
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
                        help="The number of epochs to train the model for. Defaults to 100")
    parser.add_argument('-lr', '--learning-rate', type=float, default=1E-3,
                        help="The learning rate for the model. Defaults to 1E-3.")
    parser.add_argument('-bs', '--batch-size', type=int, default=64,
                        help='Batch size for the DataLoader. Defaults to 64.')
    parser.add_argument('-l', '--num-layers', type=int, default=5,
                        help="The number layers for the LSTM. Defaults to 5.")
    parser.add_argument('-hs', '--hidden_size', type=int, default=128,
                        help="The number of dimensions for the hidden layers. Defaults to 128.")
    args = vars(parser.parse_args())

    logger = logging.getLogger(__name__) # Logging output to a file.
    output = f"{args['output']}_{args['run']}.log" # Run argument for numbered log files.
    logging.basicConfig(filename=output, level=logging.INFO, filemode='w')     # Filemode is write, so it clears the file, then appends output.
    file_path = f"checkpoints/{args['file_path']}_{args['run']}.pth"
    dataset = args['dataset']
    num_classes_per_dataset = {'species': 2, 'part': 6, 'oil_simple': 2, 'oil': 7, 'cross-species': 3}
    if dataset not in num_classes_per_dataset.keys():
        raise ValueError(f"Invalid dataset: {dataset}, not in {num_classes_per_dataset.keys()}")
    num_classes = num_classes_per_dataset[dataset]

    # Pre-processing
    is_data_augmentation = args['data_augmentation']
    # Hyperparameters
    num_epochs = args['epochs']
    learning_rate = args['learning_rate']
    batch_size = args['batch_size']
    hidden_size = args['hidden_size']
    num_layers = args['num_layers']
    input_size = 1 
    sequence_length = 1023

    # Create dataset and dataloader
    train_loader, val_loader, train_steps, val_steps, data = preprocess_dataset(
        dataset=dataset,
        is_data_augmentation=is_data_augmentation,
        batch_size=batch_size, 
        is_pre_train=False
    )

    # Create model instance
    model = LSTM(input_size, hidden_size, num_layers, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Display model telemetry every n epochs.
    print_every_n_epochs = 100

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Set the model to training mode.
        model.train()

        total_loss = 0
        all_predictions = []
        all_labels = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Get predictions
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy().tolist())
            _, actual = torch.max(y.data, 1)
            all_labels.extend(actual.cpu().numpy().tolist())

        avg_loss = total_loss / len(train_loader)
        logger.debug(f"all_predictions: {all_predictions}\n all_labels{all_labels}")
        accuracy = accuracy_score(all_labels, all_predictions)
    
        if epoch % print_every_n_epochs == 0:
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.4f}')

            # Set the model to evaluation mode.
            model.eval()
            total_loss = 0
            all_predictions = []
            all_labels = []
            with torch.no_grad():
                for x,y in val_loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    loss_as_str = loss.item()

                    # Get predictions
                    _, predicted = torch.max(outputs.data, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    _, actual = torch.max(y.data, 1)
                    all_labels.extend(actual.cpu().numpy())
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(val_loader)
                logger.debug(f"all_predictions: {all_predictions}\n all_labels{all_labels}")
                accuracy = accuracy_score(all_labels, all_predictions)
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}')

