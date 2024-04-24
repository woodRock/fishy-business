"""
Problem - problem.py
====================

This module is contains a Wrapper-based PSO implementation that balances classification accuracy and selection ratio.
The balanced accuracy is measured with a LinearSVC classifier with l1 regularization.
This classifier is evaluated on test performance on stratified dataset with (90-10) train-test split.

References:
1. Kennedy, J., & Eberhart, R. (1995, November). Particle swarm optimization. 
    In Proceedings of ICNN'95-international conference on neural networks 
    (Vol. 4, pp. 1942-1948). IEEE.
2. Nguyen, H. B., Xue, B., Liu, I., & Zhang, M. (2014, July). 
    Filter based backward elimination in wrapper based PSO for feature selection in classification. 
    In 2014 IEEE congress on evolutionary computation (CEC) (pp. 3111-3118). IEEE.
"""
import math
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC as SVM


class Problem:

    def __init__(self, minimized):
        """
        Args:
            minimized (boolean): If true, we minimize the objective function. Otherwise, if false we maxmimize it.
        """
        self.minimized = minimized

    def fitness(self, sol):
        """ Returns the fitness for a given solution.

        Args:
            sol (list): A candidate solution.

        Returns:
            float: The fitness of a candidate solution.
        """
        return 10 * sol.shape[0] + np.sum(sol ** 2 - 10 * np.cos(2 * math.pi * sol))

    def worst_fitness(self):
        """ Returns the worst fitness value possible.

        Returns:
            Infinity if minimized, otherwise negative infinity.
        """
        w_f = float('inf') if self.minimized else float('-inf')
        return w_f

    def is_better(self, first, second):
        """ Compares two fitness scores to eachother.

        If we are minimizing, the best fitness is smaller.
        If we are maximizing the best fitness is larger.

        Returns:
            (bool): true if better, false otherwise.
        """
        if self.minimized:
            return first < second
        else:
            return first > second


class FeatureSelection(Problem):

    def __init__(self, minimized, X, y):
        """
        Args:
            minimized (boolean): If true, we minimize the objective function. Otherwise, if false we maxmimize it.
            X: the features for the dataset.
            y: the class labels for the dataset.
        """
        Problem.__init__(self, minimized=minimized)
        self.X = X
        self.y = y
        self.n_samples, self.n_features = self.X.shape[0], self.X.shape[1]
        self.n_classes = len(np.unique(self.y))
        self.threshold = 0.6

    def fitness(self, sol):
        """ Measure the fitness for a given candidate solution. 

        This method measures the fitness of a candidate solution in two parts, classification accuracy and selection ratio.
        The selection ratio is a penalty of how many features are chosen, less feautres result in a smaller penatly.
        Accuracy is measured as balanced LinearSVC classifier accuracy on the test set given a stratified (90-10) train-test split.

        Args:
            sol: A solution is an array of length n_features, with scores assigned to each feature.

        Returns:
            fitness: The fitness of a given solution as a balance of classification accuracy and selection ratio.  
        """
        sel_fea = np.where(sol > self.threshold)[0]
        sel_ratio = len(sel_fea) / self.n_features
        X = self.X[:, sel_fea]
        if len(sel_fea) == 0:
            return self.worst_fitness()

        clf = SVM(penalty='l1', dual=False, tol=1e-3, max_iter=5_000)
        err = 0.0
        X_train, X_test, y_train, y_test = train_test_split(
            X, self.y, test_size=0.10, random_state=42, stratify=self.y)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
        err += 1 - acc
        fitness = 0.98 * err + 0.02 * sel_ratio
        return fitness
