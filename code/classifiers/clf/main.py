import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.linear_model import LogisticRegression as lor
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC as svm
# from lda import lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
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


def run_experiments(datasets, runs=30, k=3):
    results = {}

    for dataset in datasets:
        print(f"Dataset: {dataset}")
        X, y = load_dataset(dataset)

        class_weights = calculate_class_weights(y)

        models = {
            'knn': knn(class_weights=class_weights),
            'dt': dt(class_weight=class_weights),
            'lor': lor(class_weight=class_weights, max_iter=2_000),
            # 'lda': lda(class_weights=class_weights),
            'lda': lda(),
            'nb': nb(class_weights=class_weights),
            'rf': rf(class_weight=class_weights),
            'svm': svm(kernel='linear', class_weight=class_weights, probability=True),
            'ensemble': VotingClassifier(
                estimators=[
                    ('knn', knn(class_weights=class_weights)),
                    ('dt', dt(class_weight=class_weights)),
                    ('lor', lor(class_weight=class_weights, max_iter=2_000)),
                    # ('lda', lda(class_weights=class_weights)),
                    ('lda', lda()),
                    ('nb', nb(class_weights=class_weights)),
                    ('rf', rf(class_weight=class_weights)),
                    ('svm', svm(kernel='linear', class_weight=class_weights,  probability=True))
                ],
                voting='soft'
            )
        }

        dataset_results = {}

        for name, model in models.items():
            train_accs = []
            test_accs = []

            # Step 3: Initialize variables to hold the mean ROC curve
            tprs = []
            mean_fpr = np.linspace(0, 1, 100)
            roc_aucs = []
            f1_scores = []
            precisions = []
            recalls = []

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

                    # Get the predicted probabilities for the positive class
                    y_prob = model.predict_proba(X_test)[:, 1]
                    
                    # Compute the ROC curve and AUC for the current fold
                    # fpr, tpr, _ = roc_curve(y_test, y_prob)
                    fpr, tpr = 0, 0 
                    # roc_auc = auc(fpr, tpr)
                    roc_auc = 0
                    roc_aucs.append(roc_auc)
                    
                    # Interpolate the ROC curve
                    # interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr = np.zeros(len(mean_fpr))
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)

                    # Calculate F1-score, precision, and recall
                    f1_scores.append(f1_score(y_test, test_pred, average='macro'))
                    precisions.append(precision_score(y_test, test_pred, average='macro'))
                    recalls.append(recall_score(y_test, test_pred, average='macro'))

                # print(f"  {name}: {np.mean(k_fold_train_accs) * 100:.2f}\%")
                # print(f"  {name}: {np.mean(k_fold_test_accs) * 100:.2f}\%")

                train_accs.append(np.mean(k_fold_train_accs))
                test_accs.append(np.mean(k_fold_test_accs))

            train_mean = np.mean(train_accs) * 100
            train_std = np.std(train_accs) * 100
            test_mean = np.mean(test_accs) * 100
            test_std = np.std(test_accs) * 100

            # Step 5: Compute the mean and standard deviation of the ROC curve
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(roc_aucs)

            mean_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores)

            mean_precision = np.mean(precisions)
            std_precision = np.std(precisions)

            mean_recall = np.mean(recalls)
            std_recall = np.std(recalls)

            dataset_results[name] = {
                'train_acc': train_mean,
                'train_std': train_std,
                'test_acc': test_mean,
                'test_std': test_std,
                'auc': mean_auc,
                'auc_std': std_auc,
                'f1-score': mean_f1,
                'f1-score_std': std_f1,
                'preision': mean_precision,
                'precision_std': std_precision,
                'recall': mean_recall,
                'recall_std': std_recall
            }

            # Step 6: Plot the mean ROC curve with shaded variability
            plt.figure(figsize=(8, 6))
            plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
            plt.fill_between(mean_fpr, np.maximum(mean_tpr - np.std(tprs, axis=0), 0), 
                            np.minimum(mean_tpr + np.std(tprs, axis=0), 1), color='b', alpha=0.2)
            plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve with Stratified K-Fold Cross-Validation')
            plt.legend(loc='lower right')
            plt.savefig(f"figures/{dataset}_{name}_roc_auc_curve.png")
            plt.show()

            # Step 7: Print the F1-score, precision, and recall
            print(f"ROC AUC score: {mean_auc:.2f} ± {std_auc:2f}")
            print(f'F1-Score: {mean_f1:.2f} ± {std_f1:.2f}')
            print(f'Precision: {mean_precision:.2f} ± {std_precision:.2f}')
            print(f'Recall: {mean_recall:.2f} ± {std_recall:.2f}')

        results[dataset] = dataset_results

    return results

if __name__ == "__main__":
    # datasets = ["species", "part", "oil", "cross-species"]
    datasets = ["species"]
    results = run_experiments(datasets)

    # Print results (for verification)
    for dataset, classifiers in results.items():
        print(f"\nDataset: {dataset}")
        for classifier, metrics in classifiers.items():
            print(f"  {classifier}:")
            print(f"    Train: {metrics['train_acc']:.2f}\% $\pm$ {metrics['train_std']:.2f}\%")
            print(f"    Test:  {metrics['test_acc']:.2f}\% $\pm$ {metrics['test_std']:.2f}\%")
            # Print the ROC AUC score, F1-score, Precision and Recall.
            print(f"    AUC: {metrics['auc']:.2f}\% $\pm$ {metrics['auc_std']:.2f}\%")
            print(f"    F1-score: {metrics['f1-score']:.2f}\% $\pm$ {metrics['f1-score_std']:.2f}\%")
            print(f"    Precision: {metrics['preision']:.2f}\% $\pm$ {metrics['precision_std']:.2f}\%")
            print(f"    Recall: {metrics['recall']:.2f}\% $\pm$ {metrics['recall_std']:.2f}\%")