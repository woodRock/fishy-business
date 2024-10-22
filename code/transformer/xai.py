import torch
import torch.nn.functional as F
import torch.nn as nn 
import torch.optim as optim 
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler
from transformer import Transformer
from util import preprocess_dataset
from train import train_model


class TransformerWrapper:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict_proba(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = x.to(device)
        with torch.no_grad():
            # Assuming src and tgt are the same for simplicity
            logits = self.model(x, x)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

# Instantiate your transformer model
dataset = "cross-species"
input_dim = 1023
output_dim = 3
num_layers = 3
num_heads = 3
hidden_dim = 128
dropout = 0.2

model = Transformer(
    input_dim=input_dim, 
    output_dim=output_dim, 
    num_layers=num_layers, 
    num_heads=num_heads, 
    hidden_dim=hidden_dim, 
    dropout=dropout
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_loader, data = preprocess_dataset(
    dataset=dataset,
    batch_size=64,
    is_data_augmentation=False,
    is_pre_train=False
)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
# AdamW (Loshchilov 2017)
optimizer = optim.AdamW(model.parameters(), lr=1E-5)

# Specify the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model
model = train_model(
    model, 
    train_loader, 
    criterion,
    optimizer, 
    num_epochs=100, 
    patience=10
)

# Wrap the model
wrapped_model = TransformerWrapper(model)

# Generate some dummy data to initialize the LIME explainer
# This should ideally be a sample of your actual data

# Assuming you have a DataLoader named train_loader
data_iter = iter(train_loader)
features, labels = next(data_iter)

labels_per_dataset = {
    "species": ["Hoki", "Mackerel"],
    "part": ["Fillet", "Heads", "Livers", "Skins", "Guts", "Frames"],
    "oil": ["50", "25", "10", "05", "01", "0.1"," 0"],
    "oil_simple": ["Oil", "No oil"],
    "cross-species":["Hoki-Mackeral", "Hoki", "Mackerel"],
    "instance-recognition": ["different", "same"]
}

if dataset not in labels_per_dataset.keys():
    raise ValueError(f"Not a valid dataset: {dataset}")

class_names = labels_per_dataset[dataset]
# Give mass-to-charge ratios to 4 decimal places as feature names.
# Skip the first column, which is the label.
feature_names = [f"{float(x):.4f}" for x in data.axes[1].tolist()[1:]]

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
instance = None
label = None
# Retrieve the first instance
for f, l in zip(features, labels):
    # ["Hoki", "Mackerel"]
    # ["Fillet", "Heads", "Livers", "Skins", "Guts", "Frames"]
    # ["Hoki-Mackeral", "Hoki", "Mackerel"]
    if torch.equal(l,torch.tensor([0,0,1])):
        instance = f
        label = l
        break

first_instance = instance
first_instance_label = label

print(f"first_instance_label: {first_instance_label}")

# Generate explanation
explanation = explainer.explain_instance(
    first_instance.cpu().numpy(), 
    wrapped_model.predict_proba, 
    num_features=5,
    num_samples=100
)

# Save the explanation as an image
fig = explanation.as_pyplot_figure()

# Adjust figure size
fig.set_size_inches(10, 8)

# Adjust layout to make sure everything fits
plt.tight_layout()

# Save the figure
fig.savefig('figures/cross-species/lime_transformer_cross-species_mackerel.png')

# Optionally, display the explanation
plt.show()