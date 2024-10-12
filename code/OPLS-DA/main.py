import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, classification_report
from pyopls import OPLS
from data import load_dataset

dataset = "cross-species"
X,y = load_dataset(dataset=dataset)

# 1. Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Stratified K-Fold Cross-Validation setup (k=5)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store results
train_balanced_accuracies = []
test_balanced_accuracies = []
fold = 1

for train_index, test_index in skf.split(X_scaled, y):
    print(f"Fold {fold}")
    fold += 1

    # Split the data into train and test sets for this fold
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 3. Apply OPLS-DA for feature extraction
    n_components = 1  # Number of predictive components to extract
    opls = OPLS(n_components=n_components)
    
    # Fit OPLS on training data and transform both train and test sets
    X_train_opls = opls.fit_transform(X_train, y_train)
    X_test_opls = opls.transform(X_test)

    # 4. Train LDA classifier on OPLS-transformed training data
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_opls, y_train)

    # 5. Predict on both train and test sets
    y_train_pred = lda.predict(X_train_opls)
    y_test_pred = lda.predict(X_test_opls)

    # 6. Compute balanced accuracy for both train and test sets
    train_bal_acc = balanced_accuracy_score(y_train, y_train_pred)
    test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    
    train_balanced_accuracies.append(train_bal_acc)
    test_balanced_accuracies.append(test_bal_acc)

    print(f"Train Balanced Accuracy: {train_bal_acc * 100:.2f}%")
    print(f"Test Balanced Accuracy: {test_bal_acc * 100:.2f}%\n")

# 7. Final Evaluation: Mean balanced accuracy across all folds
mean_train_bal_acc = np.mean(train_balanced_accuracies)
std_train_bal_acc = np.std(train_balanced_accuracies)
mean_test_bal_acc = np.mean(test_balanced_accuracies)
std_test_bal_acc = np.std(test_balanced_accuracies)

print(f"Dataset: {dataset}")
print(f"\t Train: {mean_train_bal_acc * 100:.2f}\% $\pm$ {std_train_bal_acc * 100:.2f}\%")
print(f"\t Test: {mean_test_bal_acc * 100:.2f}\% $\pm$ {std_test_bal_acc * 100:.2f}\%")
