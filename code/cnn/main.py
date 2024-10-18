import argparse
import logging
from tqdm import tqdm
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from util import preprocess_dataset
from cnn import CNN
from pre_training import pre_train_masked_spectra, pre_train_transfer_learning
from train import train_model, evaluate_model

def calculate_class_weights(train_loader: DataLoader) -> torch.Tensor:
    """
    Calculate the weights for each class based on their frequency in the dataset.
    
    Args:
        train_loader (DataLoader): The training data loader.
    
    Returns:
        torch.Tensor: A tensor of weights for each class.
    """
    class_counts = {}
    total_samples = 0
    
    for _, labels in train_loader:
        for label in labels:
            class_label = label.argmax().item()
            class_counts[class_label] = class_counts.get(class_label, 0) + 1
            total_samples += 1
    
    class_weights = []
    for i in range(len(class_counts)):
        class_weights.append(1.0 / class_counts[i])
    
    class_weights = torch.FloatTensor(class_weights)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    
    return class_weights


def parse_arguments():
    parser = argparse.ArgumentParser(
                    prog='Convolutional Neural Network (CNN)',
                    description='A CNN for fish species classification.',
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
    parser.add_argument('-lr', '--learning-rate', type=float, default=1E-7,
                        help="The learning rate for the model. Defaults to 1E-3.")
    parser.add_argument('-bs', '--batch-size', type=int, default=64,
                        help='Batch size for the DataLoader. Defaults to 64.')

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
    if args.dataset == "instance-recogntion":
        n_features = 2046
    n_classes_per_dataset = {"species": 2, "part": 6, "oil": 7, "cross-species": 3, "instance-recognition": 2}

    if args.dataset not in n_classes_per_dataset:
        raise ValueError(f"Invalid dataset: {args.dataset} not in {n_classes_per_dataset.keys()}")
    
    n_classes = n_classes_per_dataset[args.dataset]

    if args.masked_spectra_modelling:
        # Load the dataset.
        train_loader, val_loader = preprocess_dataset(
            args.dataset, 
            args.data_augmentation, 
            batch_size=args.batch_size,
            is_pre_train=True
        )    

        # Instantiate model, loss function, and optimizer
        model = CNN(
            input_size=n_features, 
            num_classes=n_features,
            dropout=args.dropout
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

        model = pre_train_masked_spectra(
            model=model,
            num_epochs=3,
            train_loader=train_loader,
            file_path='checkpoints/cnn_checkpoint.pth',
            device = device,
            criterion=criterion,
            optimizer=optimizer,
        )

    # Load the dataset.
    train_loader, data = preprocess_dataset(
        args.dataset, 
        args.data_augmentation, 
        batch_size=args.batch_size,
        is_pre_train=False
    )    

    # Instantiate model, loss function, and optimizer
    model = CNN(
        input_size = n_features, 
        num_classes = n_classes,
        dropout = args.dropout
    )

    # model = pre_train_transfer_learning(
    #     model = model, 
    #     file_path ='checkpoints/cnn_checkpoint.pth', 
    #     output_dim = n_classes
    # )

    # If pre-trained, load the pre-trained model weights.
    if args.masked_spectra_modelling:
        model = pre_train_transfer_learning(
            model = model, 
            file_path ='checkpoints/cnn_checkpoint.pth', 
            output_dim = n_classes
        )

    class_weights = calculate_class_weights(train_loader=train_loader)
    # class_weights = torch.tensor([0.5, 0.5])
    # print(f"class_weights: {class_weights}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # Label smoothing (Szegedy 2016)
    # criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    # AdamW optimizer (Loshchilov 2017)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    model = train_model(
        model=model, 
        train_loader=train_loader, 
        criterion=criterion,
        optimizer=optimizer, 
        num_epochs=args.epochs, 
        patience=args.early_stopping
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