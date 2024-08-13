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

    train_loader, val_loader, _ , _, data = preprocess_dataset(
        dataset=args.dataset,
        is_data_augmentation=False,
        batch_size=64,
        is_pre_train=False
    )

    data_iter = iter(train_loader)
    features, labels = next(data_iter)
    labels = labels.argmax(1)
    
    class_names_per_dataset = {
        "species": ['Hoki', 'Mackerel'],
        "part": ['fillets', 'heads', 'livers', 'skins', 'guts', 'frames'],
        "oil": ['50', '25', '10', '5', '1', '0.1', '0'],
        "cross-species": ['Hoki-Mackerel', 'Hoki', 'Mackerel']
    }
    
    if args.dataset not in class_names_per_dataset.keys():
        raise ValueError(f"Invalid dataset: {args.dataset} not in {class_names_per_dataset.keys()}")
    
    class_names = class_names_per_dataset[args.dataset]
    
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
        features.cpu().numpy(),
        mode="classification",
        feature_names=feature_names,
        class_names=['Class 0', 'Class 1'],
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
        exp = explainer.explain_instance(
            instance, 
            model.predict_proba, 
            num_features=10
        )
        all_importances[name] = dict(exp.as_list())

    # Create a DataFrame with all features and their importances for each model
    df = pd.DataFrame(all_importances)

    # Fill NaN values with 0 (features not in top 10 for a model)
    df = df.fillna(0)

    # Sort features by average absolute importance across models
    df['avg_importance'] = df.abs().mean(axis=1)
    df = df.sort_values('avg_importance', ascending=False)
    df = df.drop('avg_importance', axis=1)

    # Plot
    plt.figure(figsize=(20, 10))

    for model_name in df.columns:
        plt.plot(df.index, df[model_name], marker='o', linestyle='-', linewidth=2, markersize=8, label=model_name)

    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.title(f'Feature Importance Comparison Across Models (LIME) on {args.dataset} for {class_names[label]}', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add vertical lines to separate features
    for i in range(len(df.index)):
        plt.axvline(i, color='gray', linestyle='--', alpha=0.3)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig(f"figures/{args.dataset}/lime_feature_comparison_line.png")
    plt.show()


if __name__ == "__main__":
    main()
