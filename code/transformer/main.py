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
    parser.add_argument('-msm', '--masked-spectra-modelling',
                    action='store_true', default=False,
                    help="Flag to perform masked spectra modelling. Defaults to False.")  
    parser.add_argument('-nsp', '--next-spectra-prediction',
                    action='store_true', default=False,
                    help="Flag to perform next spectra prediction. Defaults to False.") 
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
    # num_layers = 6
    num_heads = 3
    # hidden_dim = 64
    hidden_dim = 128 
    learning_rate = args['learning_rate']
    batch_size = args['batch_size']
    is_decoder_only = True

    logger.info(f"Reading the dataset: fish {dataset}")
    train_loader, val_loader, train_steps, val_steps, data= preprocess_dataset(
        dataset, 
        is_data_augmentation, 
        batch_size=batch_size,
        is_pre_train=True
    )

    if is_masked_spectra:
        # Load the transformer.
        model = Transformer(input_dim, output_dim, num_layers, num_heads, hidden_dim, dropout)
        logger.info(f"model: {model}")
       
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
        logger.info(f"model: {model}")

        # Transfer learning
        if is_masked_spectra:
            model = pre_train_transfer_learning(model, file_path=file_path)

        # Label smoothing (Szegedy 2016)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # AdamW (Loshchilov 2017)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
       
        # Specify the device (GPU or CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        logger.info("Pre-training the network: Next Spectra Prediction")
        startTime = time.time()

        # Train the model
        model = pre_train_model_next_spectra(
            model, 
            num_epochs=num_epochs,  
            train_loader=train_loader, 
            val_loader=val_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            file_path=file_path
        )

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

    # Load the dataset with quality control and other unrelated instances removed.
    # train_loader, val_loader, test_loader, train_steps, val_steps, data = preprocess_dataset(dataset, is_data_augmentation)
    train_loader, val_loader, train_steps, val_steps, data = preprocess_dataset(
        dataset, 
        is_data_augmentation, 
        batch_size=batch_size,
        is_pre_train=False
    )

    # Output dimension is the number of classes in the dataset.
    if dataset == "species" or dataset == "oil":
        output_dim = 2  # ['Hoki', 'Mackerel'] or ['Oil', 'None']
    elif dataset =="part":
        output_dim = 6 # ['Fillet'  'Heads' 'Livers' 'Skins' 'Guts' 'Frames']
    elif dataset =="cross-species":
        output_dim = 3 # ['Hoki, 'Mackerel', 'Hoki-Mackerel']
    else:
        raise ValueError(f"Not a valid dataset: {dataset}")

    # Early stopping (Morgan 1989)
    if is_early_stopping:
        # early_stopping = EarlyStopping(patience=patience, delta=0.001, path=file_path)
        # patience = num_epochs, stores best run, but doesn't stop training early.
        early_stopping = EarlyStopping(patience=patience, delta=0.001, path=file_path)

    # Initialize the model, criterion, and optimizer
    model = Transformer(input_dim, output_dim, num_layers, num_heads, hidden_dim, dropout)
    logger.info(f"model: {model}")

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
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, 
        num_epochs=num_epochs, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion, 
        optimizer=optimizer, 
        device=device,
        is_early_stopping=True,
        early_stopping=early_stopping
    )
    # finish measuring how long training took
    endTime = time.time()
    logger.info("Total time taken to train the model: {:.2f}s".format(endTime - startTime))

    if is_early_stopping:
        # Early stopping (Morgan 1989)
        # If the model stopped early.
        if early_stopping.early_stop:
            # Load the checkpoint
            checkpoint = torch.load(file_path)
            # Load model parameters with best validation loss.
            model.load_state_dict(checkpoint, strict=False)

    model.eval()
    # switch off autograd
    with torch.no_grad():
        # loop over the test set
        datasets = [("train", train_loader), ("validation", val_loader)]
        for name, dataset_x_y in datasets:
            for (x,y) in dataset_x_y:
                (x,y) = (x.to(device), y.to(device))
                pred = model(x, x, src_mask=None)
                test_correct = (pred.argmax(1) == y.argmax(1)).sum().item()
                accuracy = test_correct / len(x)
                logger.info(f"{name} got {test_correct} / {len(x)} correct, accuracy: {accuracy}")
                plot_confusion_matrix(dataset, name, y.argmax(1).cpu(), pred.argmax(1).cpu())
    i = 10
    columns = data.axes[1][1:(i+1)].tolist()
    # First self-attention layer of the encoder.
    attention_weights = model.encoder.layers[0].self_attention.fc_out.weight
    attention_weights = attention_weights[:i,:i].cpu().detach().numpy()
    plot_attention_map("encoder", attention_weights, columns, columns)
    
    if not is_decoder_only:
        # Last self-attention layer of the decoder.
        attention_weights = model.decoder.layers[-1].feed_forward.fc2.weight
        attention_weights = attention_weights[:i,:i].cpu().detach().numpy()
        plot_attention_map("decoder", attention_weights, columns, columns)
