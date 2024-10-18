import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

ml_models_data = {
    "CNN": {
        "binary_classification": {
            "train": [0.9997, 1.0000, 0.9942, 1.0000, 0.9825, 0.9995, 0.9942, 0.9186, 0.8621, 0.8621],
            "test": [0.5227, 0.5217, 0.5000, 0.5000, 0.5000, 0.5000, 0.5197, 0.5000, 0.5187,0.5000]
        },
        "contrastive_learning": {
            "train": [0.9417, 0.9354, 0.9504, 0.9569, 0.9581, 0.9215, 0.9524, 0.9583, 0.9332, 0.9430],
            "test": [0.6550, 0.6677, 0.7318, 0.7057, 0.7423, 0.7618, 0.6600, 0.8598, 0.6096, 0.6883]
        }
    },
    "RCNN": {
        "binary_classification": {
            "train": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            "test": [0.5000, 0.5000, 0.5217, 0.5228, 0.5238, 0.5000, 0.5227, 0.5000, 0.5395, 0.5000]
        },
        "contrastive_learning": {
            "train": [1.0000, 1.0000, 0.9996, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9991, 1.0000],
            "test": [0.7965, 0.7060, 0.7165, 0.7826, 0.8062, 0.7090, 0.7107, 0.7131, 0.7303, 0.8318]
        }
    },
    "KAN": {
        "binary_classification": {
            "train": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            "test": [0.6262, 0.5838, 0.6231, 0.5871, 0.5871, 0.6706, 0.6335, 0.6065, 0.6109, 0.6119],
        },
        "contrastive_learning": {
            "train": [0.9322, 0.9164, 0.9472, 0.9553, 0.9643, 0.9528, 0.9453, 0.9562, 0.9593, 0.9184],
            "test": [0.7098, 0.7710, 0.7098, 0.7526, 0.7529, 0.8288, 0.7469, 0.7648, 0.7178, 0.7540, 0.7715]
        }
    },
    "LSTM": {
        "binary_classification": {
            "train": [0.5233, 0.6380, 0.5463, 0.5337, 0.5967, 0.5233, 0.6491, 0.5463, 0.5402, 0.5907],
            "test": [0.5217, 0.5414, 0.5424, 0.5238, 0.5425, 0.5455, 0.5414, 0.5424, 0.5238, 0.5425],
        },
        "contrastive_learning": {
            "train": [0.9848, 0.9917, 0.9883, 0.9914, 0.9834, 0.9878, 0.9923, 0.9780, 0.9774, 0.9900],
            "test": [0.7529, 0.7685, 0.6841, 0.6474, 0.7623, 0.7975, 0.8294, 0.8104, 0.7656, 0.6869]
        }
    },
    "Mamba": {
        "binary_classification": {
            "train": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            "test": [0.5725, 0.5838, 0.6034, 0.5602, 0.5633, 0.5414, 0.5580, 0.5434, 0.5789, 0.5207],
        },
        "contrastive_learning": {
            "train": [0.9289, 0.9450, 0.9424, 0.9438, 0.9726, 0.9579, 0.9591, 0.9491, 0.9554, 0.9698],
            "test": [0.7435, 0.6800, 0.7709, 0.7852, 0.7614, 0.7682, 0.7819, 0.7633, 0.7866, 0.8557]
        }
    },
    "Transformer": {
        "binary_classification": {
            "train": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            "test": [0.6254, 0.6105, 0.5990, 0.6446, 0.6497, 0.6696, 0.6302, 0.6609, 0.6129, 0.6510],
        },
        "contrastive_learning": {
            "train": [0.9857, 0.9697, 0.9581, 0.9816, 0.9793, 0.9779, 0.9846, 0.9740, 0.9818, 0.9852],
            "test": [0.7898, 0.7317, 0.7557, 0.7440, 0.7091, 0.6887, 0.8270, 0.7174, 0.8042, 0.8022],
        }
    },
    "VAE": {
        "binary_classification": {
            "train": [0.5003, 0.5005, 0.5097, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
            "test": [0.5000, 0.5204, 0.5297, 0.5010, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
        },
        "contrastive_learning": {
            "train": [0.9480, 0.9716, 0.9723, 0.9689, 0.9621, 0.9618, 0.9698, 0.7994, 0.9620],
            "test": [0.6493, 0.7054, 0.6328, 0.6985, 0.6431, 0.7270, 0.6929, 0.6660, 0.6905]
        }
    }
}

# Prepare data for plotting
plot_data = []
for model, tasks in ml_models_data.items():
    for task, phases in tasks.items():
        for phase, values in phases.items():
            for value in values:
                plot_data.append({
                    'Model': model,
                    'Task': task,
                    'Phase': phase,
                    'Accuracy': value
                })

df = pd.DataFrame(plot_data)

# Create a new column that combines Task and Phase
df['Category'] = df['Task'] + '_' + df['Phase']

# Set up the plot style
plt.figure(figsize=(20, 12))
sns.set_style("whitegrid")

# Define color palette
colors = {
    'binary_classification_train': '#1f77b4',  # blue
    'binary_classification_test': '#ff7f0e',   # orange
    'contrastive_learning_train': '#2ca02c',   # green
    'contrastive_learning_test': '#d62728'     # red
}

# Create the box plot
ax = sns.boxplot(x='Model', y='Accuracy', hue='Category', 
                 data=df, palette=colors, 
                 hue_order=['binary_classification_train', 'binary_classification_test', 
                            'contrastive_learning_train', 'contrastive_learning_test'])

# Customize the plot
plt.title('Performance Comparison of ML Models', fontsize=20)
plt.xlabel('Model', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

# Modify legend
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, ['Binary Classification (Train)', 'Binary Classification (Test)', 
                     'Contrastive Learning (Train)', 'Contrastive Learning (Test)'], 
           title='Category', title_fontsize='14', fontsize='12', 
           bbox_to_anchor=(1.05, 1), loc='upper left')

# Add a horizontal line for 0.5 accuracy
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

# Add vertical grid lines and shading
num_models = len(df['Model'].unique())
for i in range(num_models + 1):
    plt.axvline(x=i-0.5, color='gray', linestyle='-', alpha=0.5)
    if i % 2 == 0:
        plt.axvspan(i-0.5, i+0.5, facecolor='gray', alpha=0.1)

# Remove top and right spines
sns.despine()

# Adjust layout and display the plot
plt.tight_layout()
plt.savefig("figures/boxplot.png")
plt.show()

# Print summary statistics
for model in ml_models_data.keys():
    print(f"\n{model}:")
    for task in ['binary_classification', 'contrastive_learning']:
        if task in ml_models_data[model]:
            for phase in ['train', 'test']:
                data = ml_models_data[model][task][phase]
                print(f"  {task.replace('_', ' ').title()} - {phase.title()}:")
                print(f"    Mean = {np.mean(data):.4f}, Std = {np.std(data):.4f}")