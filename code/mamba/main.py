import argparse
import logging 
import torch
import torch.nn as nn 
import torch.optim as optim
from mamba import Mamba
from util import preprocess_dataset
from train import train_model, evaluate_model

def parse_arguments():
    parser = argparse.ArgumentParser(
                    prog='Mamba',
                    description='Mamba: Linear-Time Sequence Modeling with Selective State Spaces',
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
    parser.add_argument('-lr', '--learning-rate', type=float, default=1E-3,
                        help="The learning rate for the model. Defaults to 1E-3.")
    parser.add_argument('-bs', '--batch-size', type=int, default=64,
                        help='Batch size for the DataLoader. Defaults to 64.')
    parser.add_argument('-hd', '--hidden-dimension', type=int, default=1023,
                        help='The dimensions of the hidden dimension. Defaults to 1023.')
    parser.add_argument('-sd', '--state-dimension', type=int, default=16,
                        help='The dimensions of the state dimension. Defaults to 16.')
    parser.add_argument('-dc', '--conv-dimension', type=int, default=4,
                        help='The dimensions of the convlutional dimension. Defaults to 4.')
    parser.add_argument('-ex', '--expand', type=int, default=2,
                        help='The number of dimensions to expand. Defaults to 2.')
    parser.add_argument('-nl', '--num-layers', type=int, default=1,
                        help='The dimensions of the hidden dimension. Defaults to 1.')

    return parser.parse_args()


def setup_logging(args):
    logger = logging.getLogger(__name__)
    output = f"{args.output}_{args.run}.log"
    logging.basicConfig(filename=output, level=logging.INFO, filemode='w')
    return logger


def main():
    args = parse_arguments()
    logger = setup_logging(args)

    n_features = 1023
    n_classes_per_dataset = {"species": 2, "part": 6, "oil": 7, "cross-species": 3}

    if args.dataset not in n_classes_per_dataset:
        raise ValueError(f"Invalid dataset: {args.dataset} not in {n_classes_per_dataset.keys()}")
    
    n_classes = n_classes_per_dataset[args.dataset]

    # Load the dataset.
    train_loader, val_loader, _, _, _ = preprocess_dataset(
        args.dataset, 
        args.data_augmentation, 
        batch_size=args.batch_size,
        is_pre_train=False
    )   

    # Example usage
    model = Mamba(
        d_model=args.hidden_dimension,
        d_state=args.state_dimension,
        d_conv=args.conv_dimension,
        expand=args.expand,
        depth=args.num_layers,
        n_classes=n_classes
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    train_model(
        model = model, 
        train_loader = train_loader,
        val_loader = val_loader,
        criterion = criterion,
        optimizer = optimizer,
        num_epochs = args.epochs,
        patience = args.early_stopping
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