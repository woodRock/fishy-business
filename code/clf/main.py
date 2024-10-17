import numpy as np
import torch 
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.linear_model import LogisticRegression as lor
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC as svm
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
        X, y = load_dataset(dataset)

        class_weights = calculate_class_weights(y)

        models = {
            'knn': knn(class_weights=class_weights),
            'dt': dt(class_weight=class_weights),
            'lor': lor(class_weight=class_weights),
            'lda': lda(class_weights=class_weights),
            'nb': nb(class_weights=class_weights),
            'rf': rf(class_weight=class_weights),
            'svm': svm(kernel='linear', class_weight=class_weights),
            'ensemble': VotingClassifier(
                estimators=[
                    ('knn', knn(class_weights=class_weights)),
                    ('dt', dt(class_weight=class_weights)),
                    ('lor', lor(class_weight=class_weights)),
                    ('lda', lda(class_weights=class_weights)),
                    ('nb', nb(class_weights=class_weights)),
                    ('rf', rf(class_weight=class_weights)),
                    ('svm', svm(kernel='linear', class_weight=class_weights))
                ],
                voting='hard'
            )
        }

        dataset_results = {}

        for name, model in models.items():
            train_accs = []
            test_accs = []

            for _ in range(runs):
                skf = StratifiedKFold(n_splits=k)
                k_fold_train_accs = []
                k_fold_test_accs = []

                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    model.fit(X_train, y_train)

                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)

                    k_fold_train_accs.append(balanced_accuracy_score(y_train, train_pred))
                    k_fold_test_accs.append(balanced_accuracy_score(y_test, test_pred))

                print(f"  {name}: {np.mean(k_fold_train_accs) * 100:.2f}\%")
                print(f"  {name}: {np.mean(k_fold_test_accs) * 100:.2f}\%")

                train_accs.append(np.mean(k_fold_train_accs))
                test_accs.append(np.mean(k_fold_test_accs))

            train_mean = np.mean(train_accs) * 100
            train_std = np.std(train_accs) * 100
            test_mean = np.mean(test_accs) * 100
            test_std = np.std(test_accs) * 100

            dataset_results[name] = {
                'train_acc': train_mean,
                'train_std': train_std,
                'test_acc': test_mean,
                'test_std': test_std
            }

        results[dataset] = dataset_results

    return results

if __name__ == "__main__":
    datasets = ["species", "part", "oil", "cross-species"]
    results = run_experiments(datasets)

    # Print results (for verification)
    for dataset, classifiers in results.items():
        print(f"\nDataset: {dataset}")
        for classifier, metrics in classifiers.items():
            print(f"  {classifier}:")
            print(f"    Train: {metrics['train_acc']:.2f}\% $\pm$ {metrics['train_std']:.2f}\%")
            print(f"    Test:  {metrics['test_acc']:.2f}\% $\pm$ {metrics['test_std']:.2f}\%")