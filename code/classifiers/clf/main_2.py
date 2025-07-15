import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
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


def run_experiments(datasets, runs=30, k=5):
    results = {}

    for dataset in datasets:
        print(f"Dataset: {dataset}")

        # Load the dataset
        X, y = load_dataset(dataset)

        # Class weights are proportional to the inverse frequency of each class.
        class_weights = calculate_class_weights(y)

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
            "ensemble": VotingClassifier(
                estimators=[
                    ("knn", knn(class_weights=class_weights)),
                    ("dt", dt(class_weight=class_weights)),
                    ("lor", lor(max_iter=20000, class_weight=class_weights)),
                    ("lda", lda(class_weights=class_weights)),
                    ("nb", nb(class_weights=class_weights)),
                    ("rf", rf(class_weight=class_weights)),
                    (
                        "svm",
                        svm(
                            kernel="linear", max_iter=10000, class_weight=class_weights
                        ),
                    ),
                ],
                voting="hard",
            ),
        }

        dataset_results = {}

        # Run experiments for each of the models.
        for name, model in models.items():
            print(f"model: {name}")

            train_accs = []
            test_accs = []

            for _ in range(runs):
                # Split the data into train and test split (50%-50%).
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

                # Normalize the dataset between [0,1].
                scaler = MinMaxScaler()
                scaler.fit(X_train, y_train)
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

                train_accs.append(np.mean(train_acc))
                test_accs.append(np.mean(test_acc))

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
    results = run_experiments(datasets)

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
