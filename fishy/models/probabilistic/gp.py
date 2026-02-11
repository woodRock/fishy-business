# -*- coding: utf-8 -*-
"""
Probabilistic models for spectral classification.
Includes Gaussian Processes and other Bayesian approaches.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel as C,
    Matern,
    WhiteKernel,
)
from sklearn.base import BaseEstimator, ClassifierMixin


class GaussianProcess(BaseEstimator, ClassifierMixin):
    """
    Gaussian Process Classifier tailored for spectral data.
    Provides calibrated probability estimates and handles high-dimensional inputs.
    """

    def __init__(
        self,
        kernel_type: str = "matern",
        nu: float = 1.5,
        random_state: int = 42,
        n_restarts_optimizer: int = 5,
    ):
        self.kernel_type = kernel_type
        self.nu = nu
        self.random_state = random_state
        self.n_restarts_optimizer = n_restarts_optimizer
        self._setup_kernel()

        self.model = GaussianProcessClassifier(
            kernel=self.kernel,
            random_state=self.random_state,
            n_restarts_optimizer=self.n_restarts_optimizer,
            copy_X_train=False,
        )

    def _setup_kernel(self):
        if self.kernel_type == "rbf":
            self.kernel = 1.0 * RBF(length_scale=1.0)
        elif self.kernel_type == "matern":
            self.kernel = 1.0 * Matern(length_scale=1.0, nu=self.nu)
        else:
            self.kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_uncertainty(self, X):
        """
        Returns the predictive variance/uncertainty.
        Note: Sklearn GPC doesn't provide variance directly like GPR,
        but we can use the probability entropy or max-probability as a proxy.
        """
        probs = self.predict_proba(X)
        return 1.0 - np.max(probs, axis=1)
