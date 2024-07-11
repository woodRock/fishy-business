import argparse
import logging
from tqdm import tqdm
import torch
import torch.nn as nn 
import torch.optim as optim
from util import preprocess_dataset
from cnn import CNN
from pre_training import pre_train_masked_spectra, pre_train_transfer_learning
from train import train_model, evaluate_model


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

    args = vars(parser.parse_args())

    # Logging output to a file.
    logger = logging.getLogger(__name__)
    # Run argument for numbered log files.
    output = f"{args['output']}_{args['run']}.log"
    # Filemode is write, so it clears the file, then appends output.
    logging.basicConfig(filename=output, level=logging.INFO, filemode='w')
    file_path = f"checkpoints/{args['file_path']}_{args['run']}.pth"

    dataset = args['dataset']
    is_data_augmentation = args['data_augmentation']
    is_next_masked_spectra_modelling = args['masked_spectra_modelling']
    is_next_spectra_prediction = args['next_spectra_prediction']
    num_epochs = args['epochs']
    learning_rate = args['learning_rate']
    batch_size = args['batch_size']
    label_smoothing = args['label_smoothing']
    input_size = 1023
    num_classes_per_dataset = {'species': 2, 'part': 6, 'oil_simple': 2, 'oil': 7, 'cross-species': 3}
    if dataset not in num_classes_per_dataset.keys():
        raise ValueError(f"Invalid dataset: {dataset} not in {num_classes_per_dataset.keys()}")
    num_classes = num_classes_per_dataset[dataset]


    if is_next_masked_spectra_modelling:
        # Load the dataset.
        train_loader, val_loader, train_steps, val_steps, data = preprocess_dataset(
            dataset, 
            is_data_augmentation, 
            batch_size=batch_size,
            is_pre_train=True
        )    

        # Instantiate model, loss function, and optimizer
        model = CNN(
            input_size=input_size, 
            num_classes=1023,
            dropout=args['dropout']
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        model = pre_train_masked_spectra(
            model=model,
            num_epochs=num_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            file_path='checkpoints/cnn_checkpoint.pth',
            device = device,
            criterion=criterion,
            optimizer=optimizer,
            mask_prob=0.2
        )

    # Load the dataset.
    train_loader, val_loader, train_steps, val_steps, data = preprocess_dataset(
        dataset, 
        is_data_augmentation, 
        batch_size=batch_size,
        is_pre_train=False
    )    

    # Instantiate model, loss function, and optimizer
    model = CNN(
        input_size=input_size, 
        num_classes=num_classes,
        dropout=args['dropout']
    )

    model = pre_train_transfer_learning(
        model=model, 
        file_path='checkpoints/cnn_checkpoint.pth', 
        output_dim=num_classes
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    model = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion,
        optimizer=optimizer, 
        num_epochs=num_epochs, 
        patience=args['early_stopping']
    )

    evaluate_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        dataset=dataset, 
        device=device
    )