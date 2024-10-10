import numpy as np
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.linear_model import LogisticRegression as lor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB as nb
from sklearn.svm import SVC as svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from data import load_dataset

def run_experiments(datasets, runs=30):
    results = {}

    for dataset in datasets:
        print(f"Dataset: {dataset}")
        X, y = load_dataset(dataset)

        models = {
            'knn': knn(),
            'dt': dt(),
            'lor': lor(),
            'lda': lda(),
            'nb': nb(),
            'rf': rf(),
            'svm': svm(kernel='linear'),
            'ensemble': VotingClassifier(
                estimators=[
                    ('knn', knn()),
                    ('dt', dt()),
                    ('lor', lor()),
                    ('lda', lda()),
                    ('nb', nb()),
                    ('rf', rf()),
                    ('svm', svm(kernel='linear'))
                ],
                voting='hard'
            )
        }

        dataset_results = {}

        for name, model in models.items():
            train_accs = []
            test_accs = []

            for _ in range(runs):
                skf = StratifiedKFold(n_splits=4)
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