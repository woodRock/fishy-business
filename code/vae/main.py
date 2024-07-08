import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from util import preprocess_dataset


class MassSpecVAEClassifier(nn.Module):
    def __init__(self, input_size=1023, latent_dim=64, num_classes=2):
        super(MassSpecVAEClassifier, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Mean and log variance layers
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_size),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to(device)
        return mu + eps * std

    def decode(self, z, c):
        zc = torch.cat([z, c], dim=1)
        return self.decoder(zc)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        class_probs = F.softmax(self.classifier(z), dim=1)
        recon_x = self.decode(z, class_probs)
        return recon_x, mu, logvar, class_probs

# Loss function
def vae_classifier_loss(recon_x, x, mu, logvar, class_probs, labels, alpha=1.0, beta=1.0):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    class_probs = class_probs.float()
    labels = labels.float()
    CE = F.cross_entropy(class_probs, labels, reduction='sum')
    return BCE + alpha * KLD + beta * CE

# Handle the command line arguments for the script.
parser = argparse.ArgumentParser(
                prog='Long-short Term Memory (LSTM) Recurrent neural network',
                description='An LSTM for fish species classification.',
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
model = MassSpecVAEClassifier(input_size=1023, latent_dim=64, num_classes=num_classes)
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Training function
def train(model, data_loader, num_epochs, alpha=1.0, beta=1.0):
    model.train()
    for epoch in (pbar := tqdm(range(num_epochs), desc="Training")):
        total_loss = 0
        for batch, labels in data_loader:
            # Move data to GPU
            inputs = batch.float().to(device)
            labels = labels.long().to(device)
            
            # Forward pass
            recon_batch, mu, logvar, class_probs = model(inputs)
            loss = vae_classifier_loss(recon_batch, inputs, mu, logvar, class_probs, labels, alpha, beta)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print average loss for the epoch
        avg_loss = total_loss / len(data_loader)
        message = f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}'
        pbar.set_description(message)
        logger.info(message)

# Function to get encoded representation and class prediction
def encode_and_classify(model, data):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        mu, logvar = model.encode(data)
        z = model.reparameterize(mu, logvar)
        class_probs = F.softmax(model.classifier(z), dim=1)
    return z.cpu(), class_probs.cpu()

# Function to generate new samples of a specific class
def generate(model, num_samples, target_class):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        c = F.one_hot(torch.tensor([target_class] * num_samples), num_classes=model.num_classes).float().to(device)
        samples = model.decode(z, c)
    return samples.cpu()

train_loader, val_loader = preprocess_dataset(
    dataset="species",
    is_data_augmentation=False,
    batch_size=64,
    is_pre_train=False
)

# Example usage:
# Assuming you have a DataLoader called 'train_loader'
train(model, train_loader, num_epochs=num_epochs)

# To generate new samples:
new_samples = generate(model, num_samples=10, target_class=0)
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
encoded_spectrum, class_probs = encode_and_classify(model, your_spectra_here)
print(f"encoded_spectrum: {encoded_spectrum} \n class_probs: {class_probs}")

def evaluate_classification(model, data_loader, dataset):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch, labels in data_loader:
            # Move data to GPU
            inputs = batch.float().to(device)
            labels = labels.long().to(device)
            
            # Forward pass to get class probabilities
            _, _, _, class_probs = model(inputs)
            
            # Get the predicted class
            _, predicted = torch.max(class_probs, 1)
            _, actual = torch.max(labels, 1)
            
            # Update total and correct counts
            total += labels.size(0)
            correct += (predicted == actual).sum().item()
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    logger.info(f'{dataset} classification Accuracy: {accuracy:.2f}%')

# Assuming you have a DataLoader called 'train_loader'
evaluate_classification(model, train_loader, dataset="training")
evaluate_classification(model, val_loader, dataset="validation")