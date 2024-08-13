import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from lime.lime_tabular import LimeTabularExplainer
from util import preprocess_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(
                    prog='Lime Line Graph',
                    description='Local Interpretable Model-agnostic Explanations (LIME)',
                    epilog='Implemented in matplotlib and written in python.')
    parser.add_argument('-d', '--dataset', type=str, default="species",
                         help="The fish species or part dataset. Defaults to species")

    return parser.parse_args()


def setup_logging(args):
    logger = logging.getLogger(__name__)
    output = f"logs/output.log"
    logging.basicConfig(filename=output, level=logging.INFO, filemode='w')
    return logger


def main():
    args = parse_arguments()
    logger = setup_logging(args)

    class_names_per_dataset = {
        "species": ['Hoki', 'Mackerel'],
        "part": ['fillets', 'heads', 'livers', 'skins', 'guts', 'frames'],
        "oil": ['50', '25', '10', '5', '1', '0.1', '0'],
        "cross-species": ['Hoki-Mackerel', 'Hoki', 'Mackerel']
    }
    
    if args.dataset not in class_names_per_dataset.keys():
        raise ValueError(f"Invalid dataset: {args.dataset} not in {class_names_per_dataset.keys()}")
    
    class_names = class_names_per_dataset[args.dataset]

    train_loader, val_loader, _ , _, data = preprocess_dataset(
        dataset=args.dataset,
        is_data_augmentation=False,
        batch_size=64,
        is_pre_train=False
    )

    data_iter = iter(train_loader)
    features, labels = next(data_iter)
    labels = labels.argmax(1)

    # Create feature names
    feature_names = data.axes[1].tolist()

    # Define and train multiple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        # 'Decision Tree': DecisionTreeClassifier(random_state=42),
        'NB': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    for name, model in models.items():
        model.fit(features, labels)

    # Create a LimeTabularExplainer
    explainer = LimeTabularExplainer(
        training_data=features.cpu().numpy(), 
        training_labels=labels.cpu().numpy(),
        mode="classification",
        feature_names=feature_names,
        class_names=class_names,
        random_state=42
    )

    # Get explanation for a single instance (let's use the first test instance)
    instance = features[0].cpu().numpy()
    label = labels[0].cpu().numpy()
    print(f"label: {label}")

    # Initialize a dictionary to store feature importances for each model
    all_importances = {}

    # Get explanations for all models
    for name, model in models.items():
        exp = explainer.explain_instance(instance, model.predict_proba, num_features=10)
        all_importances[name] = dict(exp.as_list())

    # Get union of all features explained across models
    all_features = list(set().union(*[set(imp.keys()) for imp in all_importances.values()]))

    # Create a DataFrame with all features and their importances for each model
    df = pd.DataFrame(index=all_features, columns=models.keys())
    for model_name, importances in all_importances.items():
        df[model_name] = df.index.map(importances.get)

    # Fill NaN values with 0 (features not in top 10 for a model)
    df = df.fillna(0)

    # Sort features by maximum absolute importance across models
    df['max_importance'] = df.abs().max(axis=1)
    df = df.sort_values('max_importance', ascending=False).drop('max_importance', axis=1)

    # Plot
    fig, ax = plt.subplots(figsize=(20, 10))  # Increased figure width

    x = np.arange(len(df))
    width = 0.15  # width of the bars
    multiplier = 0

    # Calculate the total width of all bars for a single feature
    total_width = width * len(df.columns)

    for model_name, importance in df.items():
        offset = width * multiplier
        ax.bar(x + offset - total_width/2 + width/2, importance, width, label=model_name)
        multiplier += 1

    # Add vertical grid lines
    for i in x:
        ax.axvline(i, color='gray', linestyle='--', alpha=0.3, zorder=0)

    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)

    # Customize the plot
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Importance', fontsize=12)
    ax.set_title(f'Feature Importance Comparison Across Models (LIME) on {args.dataset} for {class_names[label]}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)

    # Add alternating background colors for better readability
    for i in range(len(x)):
        if i % 2 == 0:
            ax.add_patch(patches.Rectangle((i - 0.5, ax.get_ylim()[0]), 1, ax.get_ylim()[1] - ax.get_ylim()[0],
                                        fill=True, alpha=0.1, color='gray', zorder=0))

    # Adjust x-axis limits to show all data points
    ax.set_xlim(-0.5, len(x) - 0.5)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig(f"figures/{args.dataset}/lime_feature_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()