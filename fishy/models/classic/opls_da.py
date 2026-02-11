# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from typing import Optional

try:
    from pyopls import OPLS
except ImportError:
    OPLS = None


class OPLS_DA(BaseEstimator, ClassifierMixin):
    """
    Orthogonal Partial Least Squares Discriminant Analysis (OPLS-DA) classifier.

    This model combines OPLS for dimensionality reduction and LDA for classification.
    It requires the `pyopls` package.

    Attributes:
        n_components (int): Number of orthogonal components.
        opls (OPLS): The fitted OPLS transformer.
        lda (LinearDiscriminantAnalysis): The fitted LDA classifier.
        scaler (StandardScaler): Scaler for input features.
    """

    def __init__(self, n_components: int = 1):
        """
        Initializes the OPLS-DA classifier.

        Args:
            n_components (int, optional): Number of orthogonal components. Defaults to 1.
        """
        self.n_components = n_components
        self.opls = None
        self.lda = None
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OPLS_DA":
        """
        Fits the OPLS-DA model.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Target labels.

        Returns:
            OPLS_DA: The fitted estimator.

        Raises:
            ImportError: If pyopls is not installed.
        """
        if OPLS is None:
            raise ImportError(
                "pyopls is not installed. Please install it to use OPLS-DA."
            )

        X_scaled = self.scaler.fit_transform(X)

        # OPLS for dimensionality reduction
        self.opls = OPLS(n_components=self.n_components)
        X_opls = self.opls.fit_transform(X_scaled, y)

        # LDA for classification
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(X_opls, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted labels.
        """
        X_scaled = self.scaler.transform(X)
        X_opls = self.opls.transform(X_scaled)
        return self.lda.predict(X_opls)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Class probabilities.
        """
        X_scaled = self.scaler.transform(X)
        X_opls = self.opls.transform(X_scaled)
        return self.lda.predict_proba(X_opls)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Projects data into the OPLS space.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Projected features.
        """
        X_scaled = self.scaler.transform(X)
        return self.opls.transform(X_scaled)
