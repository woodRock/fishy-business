import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

ml_models_data = {
    "CNN": {
        "binary_classification": {
            "train": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "test": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        },
        "contrastive_learning": {
            "train": [0.9417, 0.9354, 0.9504, 0.9569, 0.9581, 0.9215, 0.9524, 0.9583, 0.9332, 0.9430],
            "test": [0.6550, 0.6677, 0.7318, 0.7057, 0.7423, 0.7618, 0.6600, 0.8598, 0.6096, 0.6883]
        }
    },
    "KAN": {
        "binary_classification": {
            "train": [1.0, 0.9881, 0.9881, 0.9815, 0.9826, 0.9881, 0.9826, 0.9934, 0.9942, 0.9823],
            "test": [0.6333951762523191, 0.609369202226345, 0.6721706864564008, 0.5659554730983303, 0.5115027829313543, 0.550834879406308, 0.5860853432282004, 0.5189239332096475, 0.6459183673469387, 0.6346011131725418]
        },
        "contrastive_learning": {
            "train": [0.9322, 0.9164, 0.9472, 0.9553, 0.9643, 0.9528, 0.9453, 0.9562, 0.9593, 0.9184],
            "test": [0.7098, 0.7710, 0.7098, 0.7526, 0.7529, 0.8288, 0.7469, 0.7648, 0.7178, 0.7540, 0.7715]
        }
    },
    "LSTM": {
        "binary_classification": {
            "train": [0.9850, 1.0000, 0.9960, 0.9879, 0.9990, 0.9841, 0.9803, 0.9812, 0.9732],
            "test": [0.6807050092764378, 0.6923933209647495, 0.6769944341372913, 0.6633580705009277, 0.6667903525046381, 0.6601113172541744, 0.7115027829313543, 0.6546382189239333, 0.7161410018552876]
        },
        "contrastive_learning": {
            "train": [0.9848, 0.9917, 0.9883, 0.9914, 0.9834, 0.9878, 0.9923, 0.9780, 0.9774, 0.9900],
            "test": [0.7529, 0.7685, 0.6841, 0.6474, 0.7623, 0.7975, 0.8294, 0.8104, 0.7656, 0.6869]
        }
    },
    "Mamba": {
        "binary_classification": {
            "train": [0.9942, 0.9879, 0.9942, 0.9881, 1.0000, 0.9881, 0.9879, 1.0000, 1.0000, 0.9704],
            "test": [0.5715213358070501, 0.5796846011131725, 0.6241187384044526, 0.6003710575139146, 0.6064935064935065, 0.6169758812615955, 0.6593692022263451, 0.5620593692022263, 0.5569573283858997, 0.6034322820037106]
        },
        "contrastive_learning": {
            "train": [0.9289, 0.9450, 0.9424, 0.9438, 0.9726, 0.9579, 0.9591, 0.9491, 0.9554, 0.9698],
            "test": [0.7435, 0.6800, 0.7709, 0.7852, 0.7614, 0.7682, 0.7819, 0.7633, 0.7866, 0.8557]
        }
    },
    "Transformer": {
        "binary_classification": {
            "train": [0.5118, 0.5181, 0.7942, 0.5481, 0.5000, 0.5180, 0.5057, 0.8573, 0.5000, 0.7233, 0.5181],
            "test": [0.7394, 0.6334, 0.8600, 0.7269, 0.6073, 0.6693, 0.6577, 0.8598, 0.5320, 0.7233]
        },
        "contrastive_learning": {
            "train": [0.9683, 0.9817, 0.9792, 0.9791, 0.9832, 0.9737, 0.9791, 0.9811, 0.9796],
            "test": [0.7365, 0.7556, 0.6921, 0.7552, 0.7217, 0.7873, 0.7142, 0.7310, 0.7476]
        }
    },
    "VAE": {
        "binary_classification": {
            "train": [0.5003,0.5005,0.5097,0.5000,0.5000,0.5000,0.5000,0.5000,0.5000,0.5000],
            "test": [0.546660482374768,0.5581632653061225,0.5265306122448979,0.5,0.5530612244897959,0.5173469387755102,0.5153061224489796,0.536734693877551,0.5244897959183673,0.5112244897959184]
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