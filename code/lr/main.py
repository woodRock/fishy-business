import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
from util import preprocess_dataset
from train import train_model
from lr import LogisticRegression
        
    
train_loader, val_loader, _, _, data = preprocess_dataset(
    dataset='cross-species',
    batch_size=64,
    is_data_augmentation=False,
    is_pre_train=False
)

# Initialize model, loss function, and optimizer
model = LogisticRegression(input_dim=1023, output_dim=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Label smoothing (Szegedy 2016)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# AdamW optimizer (Loshchilov 2017)
optimizer = optim.AdamW(model.parameters(), lr=0.01)

# Train the model.
model = train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    epochs=1_000
)
        
# Assuming you have a DataLoader named train_loader
data_iter = iter(train_loader)
features, labels = next(data_iter)

class_names = ['Hoki', 'Mackerel', 'Hoki-Mackerel']
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
    model.predict_proba, 
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
fig.savefig('figures/cross-species/lime_explanation.png')

# Optionally, display the explanation
plt.show()