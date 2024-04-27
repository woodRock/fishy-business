import logging
import argparse
import time
import torch
import torch.nn as nn 
import torch.optim as optim
from pre_training import pre_train_masked_spectra, pre_train_model_next_spectra, pre_train_transfer_learning
from transformer import Transformer 
from util import EarlyStopping, preprocess_dataset 
from train import train_model, transfer_learning
from plot import plot_attention_map, plot_confusion_matrix


if __name__ == "__main__":
    # Handle the command line arguments for the script.
    parser = argparse.ArgumentParser(
                    prog='Transformer',
                    description='A transformer for fish species classification.',
                    epilog='Implemented in pytorch and written in python.')
    parser.add_argument('-f', '--file-path', type=str, default="transformer_checkpoint")
    parser.add_argument('-d', '--dataset', type=str, default="species")
    parser.add_argument('-r', '--run', type=int, default=0)
    parser.add_argument('-o', '--output', type=str, default=f"logs/results")

    # Preprocessing
    parser.add_argument('-da', '--data-augmentation',
                    action='store_true', default=False)  
    # Pre-training
    parser.add_argument('-msm', '--masked-spectra-modelling',
                    action='store_true', default=False)  
    parser.add_argument('-nsp', '--next-spectra-prediction',
                    action='store_true', default=False) 
    # Regularization
    parser.add_argument('-es', '--early-stopping', type=int, default=5) 
    parser.add_argument('-do', '--dropout', type=float, default=0.2)
    parser.add_argument('-ls', '--label-smoothing', type=float, default=0.1)
    # Hyperparameters
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1E-5)

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
    is_next_spectra = args['next_spectra_prediction'] # @param {type:"boolean"}
    is_masked_spectra = args['masked_spectra_modelling'] # @param {type:"boolean"}
    # Regularization
    is_early_stopping = args['early_stopping'] is not None # @param {type:"boolean"}
    patience = args['early_stopping']
    dropout = args['dropout']
    label_smoothing = args['label_smoothing']
    # Hyperparameters
    num_epochs = args['epochs']
    input_dim = 1023
    output_dim = 1023  # Same as input_dim for masked spectra prediction
    num_layers = 3
    num_heads = 3
    hidden_dim = 128
    learning_rate = args['learning_rate']

    
    logger.info(f"Reading the dataset: fish {dataset}")
    
    train_loader, val_loader, test_loader, train_steps, val_steps, data = preprocess_dataset(dataset, is_data_augmentation)

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
            model = pre_train_transfer_learning(dataset, model, file_path=file_path)

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
    if dataset == "species":
        output_dim = 2  # Example: number of output classes
    elif dataset =="part":
        output_dim = 6

    # Early stopping (Morgan 1989)
    if is_early_stopping:
        early_stopping = EarlyStopping(patience=patience, delta=0.001, path=file_path)

    # Initialize the model, criterion, and optimizer
    model = Transformer(input_dim, output_dim, num_layers, num_heads, hidden_dim, dropout)

    # Transfer learning
    if is_masked_spectra or is_next_spectra:
        model = transfer_learning(dataset, model, file_path=file_path)

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
                plot_confusion_matrix(dataset, name, y.argmax(1).cpu(), pred.argmax(1).cpu())
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
