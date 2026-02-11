# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin


class ProbabilisticSVC(BaseEstimator, ClassifierMixin):
    """
    SVM with Platt scaling for well-calibrated probabilistic outputs.
    """

    def __init__(self, C=1.0, kernel="rbf", degree=3, gamma="scale", random_state=42):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.random_state = random_state
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            degree=self.degree,
            gamma=self.gamma,
            probability=True,  # Enables Platt scaling
            random_state=self.random_state,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
