import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Assuming contrastive_learning_data is already defined as in the previous artifact
contrastive_learning_data = {
    "CNN": {
        "train": [0.9417, 0.9354, 0.9504, 0.9569, 0.9581, 0.9215, 0.9524, 0.9583, 0.9332, 0.9430],
        "test": [0.6550, 0.6677, 0.7318, 0.7057, 0.7423, 0.7618, 0.6600, 0.8598, 0.6096, 0.6883]
    },
    "KAN": {
        "train": [0.9322, 0.9164, 0.9472, 0.9553, 0.9643, 0.9528, 0.9453, 0.9562, 0.9593, 0.9184],
        "test": [0.7098, 0.7710, 0.7098, 0.7526, 0.7529, 0.8288, 0.7469, 0.7648, 0.7178, 0.7540, 0.7715]
    },
    "LSTM": {
        "train": [0.9848, 0.9917, 0.9883, 0.9914, 0.9834, 0.9878, 0.9923, 0.9780, 0.9774, 0.9900],
        "test": [0.7529, 0.7685, 0.6841, 0.6474, 0.7623, 0.7975, 0.8294, 0.8104, 0.7656, 0.6869]
    },
    "Mamba": {
        "train": [0.9289, 0.9450, 0.9424, 0.9438, 0.9726, 0.9579, 0.9591, 0.9491, 0.9554, 0.9698],
        "test": [0.7435, 0.6800, 0.7709, 0.7852, 0.7614, 0.7682, 0.7819, 0.7633, 0.7866, 0.8557]
    },
    "Transformer": {
        "train": [0.9683, 0.9817, 0.9792, 0.9791, 0.9832, 0.9737, 0.9791, 0.9811, 0.9796],
        "test": [0.7365, 0.7556, 0.6921, 0.7552, 0.7217, 0.7873, 0.7142, 0.7310, 0.7476]
    },
    "VAE": {
        "train": [0.9480, 0.9716, 0.9723, 0.9689, 0.9621, 0.9618, 0.9698, 0.7994, 0.9620],
        "test": [0.6493, 0.7054, 0.6328, 0.6985, 0.6431, 0.7270, 0.6929, 0.6660, 0.6905]
    }
}

# Assuming contrastive_learning_data is already defined as in the previous artifact

# Prepare data for plotting
plot_data = []
for model, data in contrastive_learning_data.items():
    for phase in ['train', 'test']:
        for value in data[phase]:
            plot_data.append({'Model': model, 'Phase': phase.capitalize(), 'Accuracy': value})

df = pd.DataFrame(plot_data)

# Set up the plot style
plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")

# Create the violin plot
sns.violinplot(x='Model', y='Accuracy', hue='Phase', data=df, 
               split=True, inner="quartile", cut=0,
               palette={"Train": "#3498db", "Test": "#e74c3c"})

# Customize the plot
plt.title('Distribution of Contrastive Learning Accuracies Across Models', fontsize=16)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Phase', title_fontsize='12', fontsize='10')

# Add a horizontal line for 0.5 accuracy
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

# Adjust layout and display the plot
plt.tight_layout()
plt.savefig("figures/violin_plot.png")

# Print summary statistics
for model in contrastive_learning_data.keys():
    train_mean = np.mean(contrastive_learning_data[model]['train'])
    train_std = np.std(contrastive_learning_data[model]['train'])
    test_mean = np.mean(contrastive_learning_data[model]['test'])
    test_std = np.std(contrastive_learning_data[model]['test'])
    print(f"{model}:")
    print(f"  Train: Mean = {train_mean:.4f}, Std = {train_std:.4f}")
    print(f"  Test:  Mean = {test_mean:.4f}, Std = {test_std:.4f}")
    print()