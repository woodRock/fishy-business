import time
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from util import preprocess_dataset
from pso import PSO

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Particle Swarm Optimization (PSO)',
                    description='A PSO for fish species classification.',
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
    # Regularization
    parser.add_argument('-es', '--early-stopping', type=int, default=10,
                        help='Early stopping patience. To disable early stopping set to the number of epochs. Defaults to 5.')
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

    dataset = args['dataset']
    is_data_augmentation = args['data_augmentation']
    num_epochs = args['epochs']
    learning_rate = args['learning_rate']
    batch_size = args['batch_size']
    label_smoothing = args['label_smoothing']

    train_loader, val_loader = preprocess_dataset(
        dataset=dataset,
        batch_size=batch_size,
        is_data_augmentation=is_data_augmentation,
        is_pre_train=False
    )
    
    # Get the number of features and classes from the data
    X_sample, y_sample = next(iter(train_loader))
    n_features = X_sample.shape[1]
    n_classes = y_sample.shape[1]  # Assuming one-hot encoded labels
    
    # Initialize and train PSO classifier
    pso_clf = PSO(
        n_particles=500, 
        n_iterations=50, 
        c1=0.4, c2=0.4, w=0.2,
        n_classes=n_classes, n_features=n_features
    )
    
    start_time = time.time()
    pso_clf.fit(train_loader, val_loader)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.4f} seconds")
    
    # Make predictions and evaluate
    y_pred = pso_clf.predict(train_loader)
    y_true = torch.cat([torch.argmax(y, dim=1) for _, y in train_loader]).numpy()
    train_acc = accuracy_score(y_true, y_pred)
    
    y_pred = pso_clf.predict(val_loader)
    y_true = torch.cat([torch.argmax(y, dim=1) for _, y in val_loader]).numpy()
    val_acc = accuracy_score(y_true, y_pred)
    
    print(f"Final - Train Accuracy: {val_acc:.4f} Validation Accuracy: {val_acc:.4f}")
    
   