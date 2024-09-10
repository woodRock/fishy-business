import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from util import preprocess_dataset
from train import train_model, evaluate_model
from vae import VAE


def parse_arguments():
    # Handle the command line arguments for the script.
    parser = argparse.ArgumentParser(
                    prog='Variational Autoencoder (VAE) neural network',
                    description='An VAE for fish species classification.',
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
    # 

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

    return parser.parse_args()


def setup_logging(args):    # Logging output to a file.
    logger = logging.getLogger(__name__)
    output = f"{args.output}_{args.run}.log"
    logging.basicConfig(filename=output, level=logging.INFO, filemode='w')
    return logger

def main():
    args = parse_arguments()
    logger = setup_logging(args)

    n_features = 1023
    if args.dataset == "instance-recognition":
        n_features = 2046
    num_classes_per_dataset = {"species": 2, "part": 6, "oil": 7, "oil_simple": 2, "cross-species": 3, "instance-recognition": 2}
    
    if args.dataset not in num_classes_per_dataset.keys():
        raise ValueError(f"Invalid dataset: {args.dataset} not in {num_classes_per_dataset.keys()}")
    
    num_classes = num_classes_per_dataset[args.dataset]

    train_loader, val_loader = preprocess_dataset(
        dataset=args.dataset,
        is_data_augmentation=False,
        batch_size=64,
        is_pre_train=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = VAE(
        input_size=n_features,
        latent_dim=args.hidden_dimension,
        num_classes=num_classes,
        device=device,
        dropout=args.dropout
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    model = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion,
        optimizer=optimizer, 
        num_epochs=args.epochs, 
        patience=args.early_stopping,
    )

    evaluate_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        dataset=args.dataset, 
        device=device
    )

if __name__ == "__main__":
    main()