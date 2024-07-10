import logging
from tqdm import tqdm 
import torch 
from vae import vae_classifier_loss

# Training function
def train(model, data_loader, num_epochs, alpha=1.0, beta=1.0, device=None, optimizer=None):
    logger = logging.getLogger(__name__)
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
def encode_and_classify(model, data, device=None):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        mu, logvar = model.encode(data)
        z = model.reparameterize(mu, logvar)
        class_probs = F.softmax(model.classifier(z), dim=1)
    return z.cpu(), class_probs.cpu()

# Function to generate new samples of a specific class
def generate(model, num_samples, target_class, device=None):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        c = F.one_hot(torch.tensor([target_class] * num_samples), num_classes=model.num_classes).float().to(device)
        samples = model.decode(z, c)
    return samples.cpu()

def evaluate_classification(model, data_loader, dataset, device=None):
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
        message = f'{dataset} classification Accuracy: {accuracy:.2f}%'
        print(message)
        logger.info(message)