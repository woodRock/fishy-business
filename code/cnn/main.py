import argparse
import logging
from tqdm import tqdm
import torch
import torch.nn as nn 
import torch.optim as optim
from util import preprocess_dataset
from cnn import CNN


def accuracy_score(predictions, labels):
    """
    Computes the accuracy of the predictions with respect to the labels.

    Args:
        predictions (torch.Tensor): The predicted class indices (outputs from the model).
        labels (torch.Tensor): The ground truth class indices.

    Returns:
        float: The accuracy score as a percentage.
    """
    # [DEBUG]
    logger.debug(f"predictions: {type(predictions)}\n{predictions}")
    logger.debug(f"labels: {type(labels)}\n{labels.argmax(1)}")

    correct = (predictions == labels.argmax(1)).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy


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

    dataset = args['dataset']
    is_data_augmentation = args['data_augmentation']
    num_epochs = args['epochs']
    learning_rate = args['learning_rate']
    batch_size = args['batch_size']
    input_size = 1023
    num_classes_per_dataset = {'species': 2, 'part': 6, 'oil_simple': 2, 'oil': 7, 'cross-species': 3}
    if dataset not in num_classes_per_dataset.keys():
        raise ValueError(f"Invalid dataset: {dataset} not in {num_classes_per_dataset.keys()}")
    num_classes = num_classes_per_dataset[dataset]


    # Load the dataset.
    train_loader, val_loader, train_steps, val_steps, data = preprocess_dataset(
        dataset, 
        is_data_augmentation, 
        batch_size=batch_size,
        is_pre_train=False
    )    

    # Instantiate model, loss function, and optimizer
    model = CNN(input_size=input_size, num_classes=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    print_every_n_epcochs = 100

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training"):
        running_loss = 0.0
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            # Move inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if epoch % print_every_n_epcochs == 0:    # Print every 10 mini-batches
                logger.info(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0
                _, preds = torch.max(outputs, 1)
                train_accuracy = accuracy_score(preds, labels)
                logger.info(f'[Epoch {epoch + 1}] Train Loss: {running_loss / 10:.3f}, Train Accuracy: {train_accuracy:.3f}')
        
       
        # Print validation accuracy every 50 epochs
        if epoch % print_every_n_epcochs == 0:  

            # Evaluation on validation data
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    # Move inputs and labels to device
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, preds = torch.max(outputs, 1)
                    all_preds.append(preds)
                    all_labels.append(labels)
            
            # Convert lists of tensors to single tensors
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)

            val_loss /= len(val_loader)
            val_accuracy = accuracy_score(all_preds, all_labels)
            logger.info(f'[Epoch {epoch + 1}] Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_accuracy:.3f}')

    logger.info('Finished Training')
