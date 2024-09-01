import logging
import os 
import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.linear_model import LogisticRegression as lr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB as nb
from sklearn.svm import SVC as svm
from skfeature.function.information_theoretical_based import MRMR
from sklearn.feature_selection import SelectKBest


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays or matrices into random train and test subsets.
    
    Parameters:
    X : array-like, shape (n_samples, n_features)
        The data to be split.
    y : array-like, shape (n_samples,)
        The target variable.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=None
        Controls the shuffling applied to the data before applying the split.
    
    Returns:
    X_train, X_test, y_train, y_test : arrays
        The train and test subsets of X and y.
    """
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Ensure X and y have the same number of samples
    assert len(X) == len(y), "X and y must have the same number of samples"
    
    # Calculate the number of test samples
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Create an array of indices and shuffle it
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Split the indices into train and test sets
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Use the indices to create train and test sets
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return (X_train, y_train) , (X_test, y_test)

# Import the dataset.
path = os.path.join("~","Desktop", "fishy-business", "data", "REIMS_data.xlsx")
data = pd.read_excel(path, header=1)
# Filter out the quality control, hoki-mackerel mix, and body parts samples.
data = data[~data.iloc[:, 0].astype(str).str.contains('QC|HM|MO|fillet|frames|gonads|livers|skins|guts|frame|heads', case=False, na=False)]
# Remove the class label column.
X = data.iloc[:, 1:].to_numpy() 
# Take only the class label column.
y = data.iloc[:, 0].to_numpy()

# Preprocessing
# Center the data
X = X - X.mean()
# Normalize between 0 and 1
X = (X - X.min()) / (X.max() - X.min())

features = list() 
labels = list() 

for i, (x_1, x_2) in enumerate(zip(X, X[1:])):
    concatenated = np.concatenate((x_1, x_2))
    features.append(concatenated)
    label = int(y[i] == y[i+1])
    labels.append(label)

# Convert to numpy arrays.
X,y = np.array(features), np.array(labels)
# Train test split
(X_train, y_train) , (X_test, y_test) = train_test_split(X,y)
# One hot encoding 
# y_train, y_test = np.eye(2)[y_train], np.eye(2)[y_test]

models = { 
    # KNN
    'knn-n1': knn(n_neighbors=1), 
    'knn-n2': knn(n_neighbors=2), 
    'knn-n3': knn(n_neighbors=3), 
    'knn-n5': knn(n_neighbors=5), 
    'knn-n10': knn(n_neighbors=10), 
    'knn-n20': knn(n_neighbors=20), 
    'dt': dt(), 
    # LDA
    'lda-lsqr': lda(solver='lsqr'),
    'lda-svd': lda(solver='svd'),
    # 'lda-eigen': lda(solver='eigen'),
    # NB
    'nb': nb(), 
    # RF
    'rf': rf(), 
    # SVM
    'svm-linear': svm(kernel='linear'), 
    'svm-rbf': svm(kernel='rbf'), 
    'svm-poly': svm(kernel='poly'),
    'svm-sigmoid': svm(kernel='sigmoid'),
    'lr': lr(max_iter=2000),
    # Ensemble
    'ensemble': VotingClassifier(
        estimators=[
            ('knn', knn()), 
            ('dt', dt()), 
            ('lr', lr(max_iter=2000)), 
            ('lda', lda()), 
            ('nb', nb()), 
            ('rf', rf()),
            ('svm', svm(kernel='linear'))],
        voting='hard'
    )
}

results = {
    # KNN
    'knn-n1': [],
    'knn-n2': [], 
    'knn-n3': [], 
    'knn-n5': [], 
    'knn-n10': [], 
    'knn-n20': [], 
    'dt': [], 
    # LDA
    'lda-lsqr': [],
    'lda-svd': [],
    # 'lda-eigen': lda(solver='eigen'),
    # NB
    'nb': [], 
    # RF
    'rf': [], 
    # SVM
    'svm-linear': [], 
    'svm-rbf': [], 
    'svm-poly': [],
    'svm-sigmoid': [],
    'lr': [],
    # Ensemble
    'ensemble': []
}

# Training loop
for name, model in (pbar := tqdm(models.items())):
    pbar.set_description(f"Traininig {model}")
    for i in range (30):
        # Resample the train and test each epoch.
        (X_train, y_train) , (X_test, y_test) = train_test_split(X,y)
        
        # Perform mRMR feature selection for classification
        # Perform mRMR feature selection
        num_features_to_select = 20
        selected_features_idx = MRMR.mrmr(X_train, y_train, n_selected_features=num_features_to_select)
        X_train = X_train[:,selected_features_idx]
        X_test = X_test[:,selected_features_idx]
        
        model.fit(X_train, y_train)
        pred = model.predict(X_train)
        train_accuracy = balanced_accuracy_score(y_train, pred)
        pred = model.predict(X_test)
        test_accuracy = balanced_accuracy_score(y_test, pred)
        results[name].append((train_accuracy,test_accuracy))

# Print a pretty table of the results.
table_results = []
for name, result in results.items():
    [train, test] = list(zip(*result))
    train = np.array(list(train))
    test = np.array(list(test))
    table_results.append([f"{name}", f"{train.mean()} +/- {train.std()}", f"{test.mean()} +/- {test.std()}"])
headers = ["Classifier", "Train Accuracy", "Test accuracy"]
table = tabulate(table_results, headers=headers, tablefmt="pretty", floatfmt=".4f", stralign="left")
print(table)
