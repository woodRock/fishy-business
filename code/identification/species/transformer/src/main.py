import logging
import sys
import time
import torch
import torch.nn as nn 
import torch.optim as optim
from pre_training import pre_train_masked_spectra, pre_train_model_next_spectra, transfer_learning
from transformer import Transformer 
from util import EarlyStopping, preprocess_dataset
from train import train_model
from plot import plot_attention_map, plot_confusion_matrix

if __name__ == "__main__":
    # Logging output to a file.
    logger = logging.getLogger(__name__)
    # Run argument for numbered log files.
    run = sys.argv[1]
    logging.basicConfig(filename=f'logs/results_{run}.log', level=logging.INFO)
    
    # Preprocessing
    is_data_augmentation = False # @param {type:"boolean"}
    # Pretraining
    is_next_spectra = False # @param {type:"boolean"}
    is_masked_spectra = True # @param {type:"boolean"}
    # Regularization
    is_early_stopping = True # @param {type:"boolean"}
    patience = 5
    dropout = 0.2
    label_smoothing = 0.1
    # Hyperparameters
    num_epochs = 50
    input_dim = 1023
    output_dim = 1023  # Same as input_dim for masked spectra prediction
    num_layers = 3
    num_heads = 3
    hidden_dim = 128
    learning_rate = 1E-5
    file_path = 'transformer_checkpoint.pth'

    logger.info("Reading the dataset.")
    train_loader, val_loader, test_loader, train_steps, val_steps, data = preprocess_dataset(is_data_augmentation)

    if is_masked_spectra:
        # Load the transformer.
        model = Transformer(input_dim, output_dim, num_layers, num_heads, hidden_dim, dropout)
        # Specify the device (GPU or CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # Initialize your model, loss function, and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        # Early stopping (Morgan 1989)
        if is_early_stopping:
            early_stopping = EarlyStopping(patience=patience, delta=0.001, path=file_path)


        logger.info("Pre-training the network: Masked Spectra Modelling")
        startTime = time.time()

        # Pre-training (Devlin 2018)
        model = pre_train_masked_spectra(model, num_epochs=num_epochs,  train_loader=train_loader, val_loader=val_loader,device=device,criterion=criterion,optimizer=optimizer,file_path=file_path)

        # finish measuring how long training took
        endTime = time.time()
        logger.info("Total time taken to pre-train the model: {:.2f}s".format(endTime - startTime))
        
        if is_early_stopping:
            # Early stopping (Morgan 1989)
            # If the model stopped early.
            if early_stopping.early_stop:
                # Load the checkpoint
                checkpoint = torch.load(file_path)
                # Load model parameters with best validation accuracy.
                model.load_state_dict(checkpoint, strict=False)

    output_dim = 2 # Onehot encoded, same or not.

    if is_next_spectra:
        # Early stopping (Morgan 1989)
        if is_early_stopping:
            early_stopping = EarlyStopping(patience=patience, delta=0.001, path=file_path)

        # Initialize the model, criterion, and optimizer
        model = Transformer(input_dim, output_dim, num_layers, num_heads, hidden_dim, dropout)

        # Transfer learning
        if is_masked_spectra:
            model = transfer_learning(model, file_path=file_path)

        # Label smoothing (Szegedy 2016)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # AdamW (Loshchilov 2017)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        # Specify the device (GPU or CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        logger.info("Pre-training the network: Nex Spectra Prediction")
        startTime = time.time()

        # Train the model
        model = pre_train_model_next_spectra(model, num_epochs=num_epochs,  train_loader=train_loader, val_loader=val_loader,device=device,criterion=criterion,optimizer=optimizer,file_path=file_path)

        # finish measuring how long training took
        endTime = time.time()
        logger.info("Total time taken to pre-train the model: {:.2f}s".format(endTime - startTime))

        # Early stopping (Morgan 1989)
        if is_early_stopping:
            # If the model stopped early.
            if early_stopping.early_stop:
                # Load the checkpoint
                checkpoint = torch.load(file_path)
                # Load model parameters with best validation accuracy.
                model.load_state_dict(checkpoint, strict=False)

    # Define hyperparameters
    output_dim = 2  # Example: number of output classes

    # Early stopping (Morgan 1989)
    if is_early_stopping:
        early_stopping = EarlyStopping(patience=patience, delta=0.001, path=file_path)

    # Initialize the model, criterion, and optimizer
    model = Transformer(input_dim, output_dim, num_layers, num_heads, hidden_dim, dropout)

    # Transfer learning
    if is_masked_spectra or is_next_spectra:
        model = transfer_learning(model, file_path=file_path)

    # Label smoothing (Szegedy 2016)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    # AdamW (Loshchilov 2017)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Specify the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info("Training the network")
    startTime = time.time()

    # Train the model
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(model, num_epochs=num_epochs, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, device=device)

    # finish measuring how long training took
    endTime = time.time()
    logger.info("Total time taken to train the model: {:.2f}s".format(endTime - startTime))

    if is_early_stopping:
        # Early stopping (Morgan 1989)
        # If the model stopped early.
        if early_stopping.early_stop:
            # Load the checkpoint
            checkpoint = torch.load(file_path)
            # Load model parameters with best validation accuracy.
            model.load_state_dict(checkpoint, strict=False)

    model.eval()
    # switch off autograd
    with torch.no_grad():
        # loop over the test set
        datasets = [("train", train_loader), ("validation", val_loader), ("test", test_loader)]
        for name, dataset in datasets:
            for (x,y) in dataset:
                (x,y) = (x.to(device), y.to(device))
                pred = model(x, x, src_mask=None)
                test_correct = (pred.argmax(1) == y.argmax(1)).sum().item()
                logger.info(f"{name} accuracy: {test_correct} / {len(x)}")
                plot_confusion_matrix(name, y.argmax(1).cpu(), pred.argmax(1).cpu())
    i = 10
    columns = data.axes[1][1:(i+1)].tolist()
    # First self-attention layer of the encoder.
    attention_weights = model.encoder.layers[0].self_attention.fc_out.weight
    attention_weights = attention_weights[:i,:i].cpu().detach().numpy()
    plot_attention_map("encoder", attention_weights, columns, columns)
    # Last self-attention layer of the decoder.
    attention_weights = model.decoder.layers[-1].self_attention.fc_out.weight
    attention_weights = attention_weights[:i,:i].cpu().detach().numpy()
    plot_attention_map("decoder", attention_weights, columns, columns)
