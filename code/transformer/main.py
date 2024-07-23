import logging
import argparse
import time
import torch
import torch.nn as nn 
import torch.optim as optim
from util import preprocess_dataset
from pre_training import pre_train_masked_spectra, pre_train_model_next_spectra, pre_train_transfer_learning
from transformer import Transformer 
from train import train_model, evaluate_model, transfer_learning
from plot import plot_attention_map, plot_confusion_matrix


def parse_arguments():
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
    parser.add_argument('-hd', '--hidden-dimension', type=int, default=128,
                        help="The dimensionality of the hidden dimension. Defaults to 128")
    parser.add_argument('-l', '--num-layers', type=float, default=4,
                        help="Number of layers. Defaults to 3.")
    parser.add_argument('-nh', '--num-heads', type=int, default=3,
                        help='Number of heads. Defaults to 3.')

    return parser.parse_args()

def setup_logging(args):
    logger = logging.getLogger(__name__)
    output = f"{args.output}_{args.run}.log"
    logging.basicConfig(filename=output, level=logging.INFO, filemode='w')
    return logger

def main():
    args = parse_arguments()
    logger = setup_logging(args)

    input_dim = 1023

    logger.info(f"Reading the dataset: fish {args.dataset}")
    train_loader, val_loader, train_steps, val_steps, data = preprocess_dataset(
        args.dataset, 
        args.data_augmentation, 
        batch_size=args.batch_size,
        is_pre_train=True
    )

    if args.masked_spectra_modelling:
        # Load the transformer.
        model = Transformer(
            input_dim, 
            output_dim, 
            args.num_layers, 
            args.num_heads, 
            args.hidden_dimension, 
            args.dropout
        )

        logger.info(f"model: {model}")
       
        # Specify the device (GPU or CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Initialize your model, loss function, and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

        logger.info("Pre-training the network: Masked Spectra Modelling")
        startTime = time.time()

        # Pre-training (Devlin 2018)
        model = pre_train_masked_spectra(
            model, 
            num_epochs=args.epochs,  
            train_loader=train_loader, 
            val_loader=val_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            file_path=args.file_path
        )

        # finish measuring how long training took
        endTime = time.time()
        logger.info("Total time taken to pre-train the model: {:.2f}s".format(endTime - startTime))

    output_dim = 2 # Onehot encoded, same or not.

    if args.next_spectra_prediction:
        # Initialize the model, criterion, and optimizer
        model = Transformer(
            input_dim, 
            output_dim,
            args.num_layers, 
            args.num_heads, 
            args.hidden_dimension, 
            args.dropout
        )
        
        logger.info(f"model: {model}")

        # Transfer learning
        if args.masked_spectra_modelling:
            model = pre_train_transfer_learning(
                model, 
                file_path=args.file_path
            )

        # Label smoothing (Szegedy 2016)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        # AdamW (Loshchilov 2017)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
       
        # Specify the device (GPU or CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        logger.info("Pre-training the network: Next Spectra Prediction")
        startTime = time.time()

        # Train the model
        model = pre_train_model_next_spectra(
            model, 
            num_epochs=args.epochs,  
            train_loader=train_loader, 
            val_loader=val_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            file_path=args.file_path
        )

        # finish measuring how long training took
        endTime = time.time()
        logger.info("Total time taken to pre-train the model: {:.2f}s".format(endTime - startTime))


    n_features = 1023
    n_classes_per_dataset = {"species": 2, "part": 6, "oil": 7, "cross-species": 3}

    if args.dataset not in n_classes_per_dataset:
        raise ValueError(f"Invalid dataset: {args.dataset} not in {n_classes_per_dataset.keys()}")
    
    n_classes = n_classes_per_dataset[args.dataset]

    # Load the dataset with quality control and other unrelated instances removed.
    # train_loader, val_loader, test_loader, train_steps, val_steps, data = preprocess_dataset(dataset, is_data_augmentation)
    train_loader, val_loader, train_steps, val_steps, data = preprocess_dataset(
        args.dataset, 
        args.data_augmentation, 
        batch_size=args.batch_size,
        is_pre_train=False
    )

    # Initialize the model, criterion, and optimizer
    model = Transformer(
        n_features,
        n_classes, 
        args.num_layers, 
        args.num_heads, 
        args.hidden_dimension, 
        args.dropout
    )

    logger.info(f"model: {model}")

    # Transfer learning
    if args.masked_spectra_modelling or args.next_spectra_prediction:
        model = transfer_learning(args.dataset, model, file_path=args.file_path)
    
    # Label smoothing (Szegedy 2016)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # AdamW (Loshchilov 2017)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Specify the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Training the network")
    startTime = time.time()
    
    # Train the model
    model = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion,
        optimizer, 
        num_epochs=args.epochs, 
        patience=args.early_stopping
    )
    
    # finish measuring how long training took
    endTime = time.time()
    logger.info("Total time taken to train the model: {:.2f}s".format(endTime - startTime))

    evaluate_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        dataset=args.dataset, 
        device=device
    )
    
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

if __name__ == "__main__":
    main()