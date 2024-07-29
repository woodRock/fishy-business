import torch
import torch.nn.functional as F
import torch.nn as nn 
import torch.optim as optim 
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler
from mamba import Mamba
from util import preprocess_dataset
from train import train_model


class MambaWrapper:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict_proba(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = x.to(device)
        with torch.no_grad():
            logits = self.model(x)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

# Instantiate your transformer model
input_dim = 1023
output_dim = 7
num_layers = 3
num_heads = 3
hidden_dim = 128
dropout = 0.2

# Example usage
model = Mamba(
    d_model=1023,
    d_state=128,
    d_conv=16,
    expand=4,
    depth=1,
    n_classes=output_dim
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_loader, val_loader, _ ,_, data = preprocess_dataset(
    dataset='oil',
    batch_size=64,
    is_data_augmentation=False,
    is_pre_train=False
)

# Label smoothing (Szegedy 2016)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
# AdamW (Loshchilov 2017)
optimizer = optim.AdamW(model.parameters(), lr=1E-5)

# Specify the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Train the model
train_model(
    model = model, 
    train_loader = train_loader,
    val_loader = val_loader,
    criterion = criterion,
    optimizer = optimizer,
    num_epochs = 100,
    patience = 10
)

# Wrap the model
wrapped_model = MambaWrapper(model)

# Generate some dummy data to initialize the LIME explainer
# This should ideally be a sample of your actual data

# Assuming you have a DataLoader named train_loader
data_iter = iter(train_loader)
features, labels = next(data_iter)

class_names = ['50', '25', '10', '5', '1', '0.1', '0']
feature_names = data.axes[1].tolist()

# Standardize the data
scaler = StandardScaler()
data_sample = scaler.fit_transform(features)

# Create a LIME explainer
explainer = LimeTabularExplainer(
    training_data=features.cpu().numpy(), 
    training_labels=labels.cpu().numpy(),
    feature_names=feature_names, 
    class_names=class_names, 
    discretize_continuous=True,
)

# Retrieve the first instance
first_instance = features[0]
first_instance_label = labels[0]

print(f"first_instance_label: {first_instance_label}")

# Generate explanation
explanation = explainer.explain_instance(
    first_instance.cpu().numpy(), 
    wrapped_model.predict_proba, 
    num_features=10,
    num_samples=100
)

# Save the explanation as an image
fig = explanation.as_pyplot_figure()

# Adjust figure size
fig.set_size_inches(10, 8)

# Adjust layout to make sure everything fits
plt.tight_layout()

# Save the figure
fig.savefig('figures/species/lime_explanation.png')

# Optionally, display the explanation
plt.show()