import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    StratifiedGroupKFold,
)  # Added StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.linear_model import LogisticRegression as lor
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC as svm
from sklearn.preprocessing import MinMaxScaler
from lda import lda
from nb import nb
from knn import knn
from data import load_dataset


def calculate_class_weights(y):
    """
    Calculate class weights inversely proportional to class frequencies.

    Parameters:
    -----------
    y : numpy.ndarray
        Array of class labels (assumed to be binary: 0 and 1)

    Returns:
    --------
    dict
        Dictionary with class labels as keys and class weights as values

    Formula used:
    w_j = n_samples / (n_classes * n_samples_j)
    where:
    - n_samples is the total number of samples
    - n_classes is the number of unique classes
    - n_samples_j is the number of samples for class j
    """

    # Get total number of samples
    n_samples = len(y)

    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(unique_classes)

    # Calculate weights for each class
    weights = n_samples / (n_classes * class_counts)

    # Create dictionary mapping class labels to weights
    class_weights = dict(zip(unique_classes, weights))

    return class_weights


def create_pairs(X_raw, y_raw):
    features = []
    labels = []
    all_possible_pairs = [
        ((a, a_idx), (b, b_idx))
        for a_idx, a in enumerate(X_raw)
        for b_idx, b in enumerate(X_raw[a_idx + 1 :])
    ]
    for (a, a_idx), (b, b_idx) in all_possible_pairs:
        concatenated = np.concatenate((a, b))
        label = int(y_raw[a_idx] == y_raw[b_idx])
        features.append(concatenated)
        labels.append(label)
    return np.array(features), np.array(labels)


def run_experiments(datasets, runs=30, k=5):
    results = {}

    for dataset in datasets:
        print(f"Dataset: {dataset}")

        # Load the dataset
        X_original, y_original, groups_original = load_dataset(dataset)  # Modified

        # Class weights are proportional to the inverse frequency of each class.
        class_weights = calculate_class_weights(
            y_original
        )  # Modified to use y_original

        # The models run experiments for.
        models = {
            "knn": knn(class_weights=class_weights),
            "dt": dt(class_weight=class_weights),
            "lor": lor(max_iter=20000, class_weight=class_weights),
            "lda": lda(class_weights=class_weights),
            "nb": nb(class_weights=class_weights),
            "rf": rf(class_weight=class_weights),
            "svm": svm(
                kernel="linear",
                max_iter=10000,
                class_weight=class_weights,
            ),
            # "ensemble": VotingClassifier(
            #     estimators=[
            #         ("knn", knn(class_weights=class_weights)),
            #         ("dt", dt(class_weight=class_weights)),
            #         ("lor", lor(max_iter=20000, class_weight=class_weights)),
            #         ("lda", lda(class_weights=class_weights)),
            #         ("nb", nb(class_weights=class_weights)),
            #         ("rf", rf(class_weight=class_weights)),
            #         (
            #             "svm",
            #             svm(
            #                 kernel="linear", max_iter=10000, class_weight=class_weights
            #             ),
            #         ),
            #     ],
            #     voting="hard",
            # ),
        }

        dataset_results = {}

        # Run experiments for each of the models.
        for name, model in models.items():
            print(f"model: {name}")

            train_accs = []
            test_accs = []

            # Use StratifiedGroupKFold
            sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)

            # Iterate through folds
            for fold, (train_index, test_index) in enumerate(
                sgkf.split(X_original, y_original, groups_original)
            ):
                print(f"  Fold {fold + 1}")  # Added fold print

                # Split the data into train and test sets for this fold (raw, unpaired)
                X_train_raw, X_test_raw = (
                    X_original[train_index],
                    X_original[test_index],
                )
                y_train_raw, y_test_raw = (
                    y_original[train_index],
                    y_original[test_index],
                )

                # For all datasets, use raw data directly
                X_train, y_train = X_train_raw, y_train_raw
                X_test, y_test = X_test_raw, y_test_raw

                # Normalize the dataset between [0,1].
                scaler = MinMaxScaler()
                # Fit scaler only on training data, transform both
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Fit the model to the training data.
                model.fit(X_train, y_train)

                # Evaluate on the train and test dataset.
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                # Get performance metrics for balanced accuracy.
                train_acc = balanced_accuracy_score(y_train, train_pred)
                test_acc = balanced_accuracy_score(y_test, test_pred)

                train_accs.append(
                    train_acc
                )  # Removed np.mean, as it's already a single score per fold
                test_accs.append(
                    test_acc
                )  # Removed np.mean, as it's already a single score per fold

            # Store the balanced accuracy as a percentage.
            train_mean = np.mean(train_accs) * 100
            train_std = np.std(train_accs) * 100
            test_mean = np.mean(test_accs) * 100
            test_std = np.std(test_accs) * 100

            # Append the results to a dictionary.
            dataset_results[name] = {
                "train_acc": train_mean,
                "train_std": train_std,
                "test_acc": test_mean,
                "test_std": test_std,
            }

        results[dataset] = dataset_results

    return results


if __name__ == "__main__":
    # datasets = ["species", "part", "oil", "cross-species"]
    datasets = ["instance-recognition"]
    results = run_experiments(datasets, k=3)  # Set k=3 as requested

    # Print results (for verification)
    for dataset, classifiers in results.items():
        print(f"\nDataset: {dataset}")
        for classifier, metrics in classifiers.items():
            print(f"  {classifier}:")
            print(
                f"    Train: {metrics['train_acc']:.2f}\% $\pm$ {metrics['train_std']:.2f}\%"
            )
            print(
                f"    Test:  {metrics['test_acc']:.2f}\% $\pm$ {metrics['test_std']:.2f}\%"
            )
