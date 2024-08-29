import numpy as np
import pandas as pd
import os 
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.scaler = StandardScaler()

    def fit(self, X, y):
        self.X_train = self.scaler.fit_transform(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return np.array([self._predict(x) for x in X_scaled])

    def _predict(self, x):
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        if k_nearest_labels.dtype.kind in ['U', 'S']:
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        else:
            most_common = np.bincount(k_nearest_labels.astype(int)).argmax()
        
        return most_common
    

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
path = os.path.join("~","Desktop", "python", "REIMS_data.xlsx")
data = pd.read_excel(path, header=1)
# Filter out the quality control and hoki-mackerel mix samples.
data = data[~data.iloc[:, 0].astype(str).str.contains('QC|HM|MO|fillet|frames|gonads|livers|skins|guts|frame|heads', case=False, na=False)]
# Remove the class label column.
X = data.iloc[:, 1:].to_numpy() 
# Take only the class label column.
y = data.iloc[:, 0].to_numpy()

features = list() 
labels = list() 

for i, (x_1, x_2) in enumerate(zip(X, X[1:])):
    concatenated = np.concatenate((x_1, x_2))
    features.append(concatenated)
    label = int(y[i] == y[i+1])
    labels.append(label)

# Convert to numpy arrays.
X,y = np.array(features), np.array(labels)

accuracy = []
for x in tqdm(range(30)):
    # Train test split
    (X_train, y_train) , (X_test, y_test) = train_test_split(X,y)
    # One hot encoding 
    # y_train, y_test = np.eye(2)[y_train], np.eye(2)[y_test]
        
    model = KNN(k=3)

    model.fit(X_train, y_train)

    predictions = model.predict(X_train)
    train_accuracy = balanced_accuracy_score(y_train, predictions)

    # Make predictions
    predictions = model.predict(X_test)
    test_accuracy = balanced_accuracy_score(y_test, predictions)

    accuracy.append((train_accuracy, test_accuracy))

train, test = list(zip(*accuracy))
train, test = np.array(list(train)), np.array(list(test))
print(f"Train: {train.mean()} +/- {train.std()} Test: {test.mean()} +/- {test.std()} ")
