# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.optimize import minimize


class BayesianLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression with Laplace Approximation for the posterior.
    Provides uncertainty estimates for classification.
    """

    def __init__(self, C=1.0, random_state=42):
        self.C = C
        self.random_state = random_state
        self.model = LogisticRegression(
            C=self.C, random_state=self.random_state, solver="lbfgs"
        )
        self.w_map = None
        self.precision_matrix = None

    def fit(self, X, y):
        # 1. Find MAP estimate using standard Logistic Regression
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.w_map = self.model.coef_.flatten()
        self.intercept_map = self.model.intercept_

        # 2. Calculate Hessian (Precision Matrix) at the MAP estimate
        # For simplicity in high-dim spectral data, we'll use a diagonal approximation
        # or a regularization-based prior.
        n_samples, n_features = X.shape
        probs = self.model.predict_proba(X)[:, 1]
        r = probs * (1 - probs)

        # Hessian of the negative log-likelihood + log-prior
        # Prior is Gaussian with precision alpha = 1/C
        alpha = 1.0 / self.C
        self.precision_matrix = (X.T * r) @ X + alpha * np.eye(n_features)
        self.covariance_matrix = np.linalg.inv(self.precision_matrix)

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.w_map is None:
            return self.model.predict_proba(X)

        # Probit approximation for predictive distribution
        # a = w^T @ x
        # sigma^2 = x^T @ Cov @ x
        # p(y=1|x) \approx sigmoid( a / sqrt(1 + pi/8 * sigma^2) )
        mu_a = X @ self.w_map + self.intercept_map
        sigma_a_sq = np.sum((X @ self.covariance_matrix) * X, axis=1)

        kappa = 1.0 / np.sqrt(1 + (np.pi / 8.0) * sigma_a_sq)
        p1 = 1.0 / (1.0 + np.exp(-kappa * mu_a))

        return np.column_stack([1 - p1, p1])

    def get_uncertainty(self, X):
        probs = self.predict_proba(X)
        # Entropy-based uncertainty
        return -np.sum(probs * np.log(probs + 1e-10), axis=1)
