import numpy as np
from tqdm import tqdm
import warnings
# warnings.filterwarnings("ignore")
from skfeature.function.similarity_based import reliefF
from skfeature.function.information_theoretical_based import MRMR
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC as svm
from sklearn.metrics import balanced_accuracy_score
import pickle
from .PSO import Swarm, pso
from .utility import plot_accuracy, show_results
from .data import load_data, get_labels, normalize


run = 1
seed = 1617 * run
np.random.seed(seed)
dataset="Part"
X,y = load_data(dataset=dataset)
y_, labels = get_labels(y)
inc = 50 
no_features = X.shape[1] + inc
j = np.arange(inc,no_features,inc) # [50,4800]
methods = { "reliefF" : reliefF.reliefF, "mrmr": MRMR.mrmr, "chi2": chi2}
results = { "reliefF" : [], "mrmr": [], "chi2": [], "pso": []}

runs = 15 
name = "pso"
folds = 10

for k in tqdm(range(runs)):
    train_accs = []
    test_accs = []
    skf = StratifiedKFold(n_splits=folds, random_state=1234, shuffle=True)

    # DEBUG: Fold counter
    f = 1

    for train, test in skf.split(X, y):
        X_train, X_test = (X[train], X[test])
        y_train, y_test = y[train], y[test]
        X_train, X_test = normalize(X_train, X_test)

        sel_fea = pso(X_train, y_train)

        # DEBUG: Measure progress.
        print(f"Run {k}, fold {f} ")

        model = svm(penalty='l1', dual=False, tol=1e-3, max_iter=5_000)
        model.fit(X_train[:, sel_fea], y_train)

        y_predict = model.predict(X_train[:, sel_fea])
        train_acc = balanced_accuracy_score(y_train, y_predict)
        train_accs.append(train_acc)

        y_predict = model.predict(X_test[:, sel_fea])
        test_acc = balanced_accuracy_score(y_test, y_predict)
        test_accs.append(test_acc)

        # DEBUG: Increment fold counter
        f += 1

    no_fea = len(sel_fea)
    results[name].append((no_fea, np.mean(train_accs), np.mean(test_accs)))

# [CHECKPOINT 1] Save PSO results
# Save results to a dictionary to facilitate combination with cloud results (later).
with open('results-pso-local.pkl', 'wb') as f:
    pickle.dump(results, f)

for k in tqdm(j):
    for name, fs_method in methods.items(): 
        if name == "pso":
          continue

        train_accs = []
        test_accs = []
        skf = StratifiedKFold(n_splits=folds, random_state=1234, shuffle=True)

        for train, test in skf.split(X, y):
            X_train, X_test = (X[train], X[test])
            y_train, y_test = y[train], y[test]
            X_train, X_test = normalize(X_train, X_test)

            fs = SelectKBest(fs_method, k=k)
            X_train = fs.fit_transform(X_train, y_train)
            X_test = fs.transform(X_test)

            model = svm(penalty='l1', dual=False, tol=1e-3, max_iter=5_000)
            clf = model.fit(X_train, y_train)

            y_predict = model.predict(X_train)
            train_acc = balanced_accuracy_score(y_train, y_predict)
            train_accs.append(train_acc)
            y_predict = model.predict(X_test)
            test_acc = balanced_accuracy_score(y_test, y_predict)
            test_accs.append(test_acc)

        no_fea = k 
        results[name].append((no_fea, np.mean(train_accs), np.mean(test_accs)))

plot_accuracy(results, dataset)
print(results)
show_results(results)

# [CHECKPOINT 2] Save full results 
# Save results to a dictionary so they can be accessed later. 
with open('results-full-local.pkl', 'wb+') as f:
    pickle.dump(results, f)

# We can load a results pickle file as follows: 
# with open('saved_dictionary.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)