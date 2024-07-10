import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from util import preprocess_dataset
from train import train, encode_and_classify, evaluate_classification, generate
from vae import VAE


if __name__ == "__main__":
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

    # Preprocessing
    is_data_augmentation = args['data_augmentation'] # @param {type:"boolean"}
    # Pretraining
    # is_next_spectra = args['next_spectra_prediction'] # @param {type:"boolean"}
    # is_masked_spectra = args['masked_spectra_modelling'] # @param {type:"boolean"}
    
    # Regularization
    is_early_stopping = args['early_stopping'] is not None # @param {type:"boolean"}
    patience = args['early_stopping']
    dropout = args['dropout']
    label_smoothing = args['label_smoothing']

    # Hyperparameters
    num_epochs = args['epochs']
    input_dim = 1023
    num_heads = args['num_heads']
    learning_rate = args['learning_rate']

    num_classes_per_dataset = {"species": 2, "part": 6, "oil": 7, "oil_simple": 2, "cross-species": 3}
    if dataset not in num_classes_per_dataset.keys():
        raise ValueError(f"Invalid dataset: {dataset} not in {num_classes_per_dataset.keys()}")
    num_classes = num_classes_per_dataset[dataset]

    # Instantiate the model and move it to GPU
    model = VAE(
        input_size=1023, 
        latent_dim=64, 
        num_classes=num_classes
    )

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    train_loader, val_loader = preprocess_dataset(
        dataset="species",
        is_data_augmentation=False,
        batch_size=64,
        is_pre_train=False
    )

    # Example usage:
    # Assuming you have a DataLoader called 'train_loader'
    model = train(
        model, 
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        device=device, 
        optimizer=optimizer,
        alpha=0.1,
        beta=0.9
    )

    # To generate new samples:
    new_samples = generate(
        model=model, 
        num_samples=10, 
        target_class=0,
        device=device
    )

    first = new_samples[0]
    plt.plot(first)
    plt.title("Generated Mass Spectrum")
    plt.xlabel("m/z")
    plt.ylabel("intensity")
    plt.savefig("figures/generated_spectra.png")

    first = next(iter(train_loader))[0][0]
    print(f"first: {first}")
    plt.plot(first)
    plt.title("Real Mass Spectrum")
    plt.xlabel("m/z")
    plt.ylabel("intensity")
    plt.savefig("figures/real_spectra.png")

    # To get encoded representation of a single spectrum:
    your_spectra_here = first.unsqueeze(0).to(device)
    encoded_spectrum, class_probs = encode_and_classify(
        model=model, 
        data=your_spectra_here, 
        device=device
    )

    print(f"encoded_spectrum: {encoded_spectrum} \n class_probs: {class_probs}")

    # Assuming you have a DataLoader called 'train_loader'
    evaluate_classification(
        model, 
        train_loader, 
        dataset=dataset,
        train_val_test="train", 
        device=device
    )
    evaluate_classification(
        model, 
        val_loader, 
        dataset=dataset,
        train_val_test="val", 
        device=device
    )