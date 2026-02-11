import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd
from typing import Optional, Dict, Any, Union


class lda(BaseEstimator, ClassifierMixin):
    """
    Weighted Linear Discriminant Analysis (LDA) implementation.

    This class provides a custom implementation of LDA that supports class weights,
    useful for imbalanced datasets.

    Attributes:
        n_components (int): Number of components for dimensionality reduction.
        class_weights (Optional[Union[Dict[int, float], np.ndarray]]): Weights for each class.
        solver (str): Solver to use ('svd' is supported).
        scaler (StandardScaler): Scaler for input features.
        classes_ (np.ndarray): Unique class labels.
        means_ (np.ndarray): Weighted means per class.
        overall_mean_ (np.ndarray): Weighted overall mean.
        components_ (np.ndarray): Eigenvectors for projection.
        means_transformed_ (np.ndarray): Projected class means.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        class_weights: Optional[Union[Dict[int, float], np.ndarray]] = None,
        solver: str = "svd",
    ) -> None:
        """
        Initializes the LDA classifier.

        Args:
            n_components (Optional[int], optional): Number of components. Defaults to None.
            class_weights (Optional[Union[Dict[int, float], np.ndarray]], optional): Class weights. Defaults to None.
            solver (str, optional): Solver type. Defaults to "svd".
        """
        self.n_components = n_components
        self.class_weights = class_weights
        self.solver = solver
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "lda":
        """
        Fits the LDA model to the provided data.

        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).

        Returns:
            lda: The fitted estimator.
        """
        X = self.scaler.fit_transform(X)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if self.n_components is None:
            self.n_components = min(n_classes - 1, X.shape[1])

        # Set up class weights if not provided
        if self.class_weights is None:
            class_counts = np.bincount(y)
            self.class_weights = len(y) / (n_classes * class_counts)
        elif isinstance(self.class_weights, dict):
            self.class_weights = np.array(
                [self.class_weights[c] for c in range(n_classes)]
            )

        # Calculate weighted means
        self.means_ = []
        for i, cls in enumerate(self.classes_):
            mask = y == cls
            weighted_mean = np.average(
                X[mask], weights=np.full(mask.sum(), self.class_weights[i]), axis=0
            )
            self.means_.append(weighted_mean)
        self.means_ = np.array(self.means_)

        # Calculate weighted overall mean
        weights = np.array([self.class_weights[yi] for yi in y])
        self.overall_mean_ = np.average(X, weights=weights, axis=0)

        # Calculate within-class scatter matrix (Sw)
        Sw = np.zeros((X.shape[1], X.shape[1]))
        for i, cls in enumerate(self.classes_):
            mask = y == cls
            X_centered = X[mask] - self.means_[i]
            class_weights_expanded = np.full(mask.sum(), self.class_weights[i])
            Sw += np.dot(
                (X_centered * class_weights_expanded.reshape(-1, 1)).T, X_centered
            )

        # Calculate between-class scatter matrix (Sb)
        Sb = np.zeros_like(Sw)
        for i, mean in enumerate(self.means_):
            diff = (mean - self.overall_mean_).reshape(-1, 1)
            Sb += self.class_weights[i] * np.dot(diff, diff.T)

        # Solve the eigenvalue problem using SVD
        if self.solver == "svd":
            # Calculate Sw^(-1/2)
            U, s, Vt = svd(Sw + np.eye(Sw.shape[0]) * 1e-6)
            # Add small constant to avoid division by zero
            s = np.maximum(s, 1e-10)
            Sw_sqrt_inv = np.dot(U * (1.0 / np.sqrt(s)), Vt)

            # Transform Sb
            Sb_transformed = np.dot(np.dot(Sw_sqrt_inv, Sb), Sw_sqrt_inv.T)

            # Perform SVD on transformed Sb
            U, s, Vt = svd(Sb_transformed)

            # Get eigenvectors
            self.components_ = np.dot(Sw_sqrt_inv.T, U[:, : self.n_components])

        # Project means
        self.means_transformed_ = np.dot(self.means_, self.components_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Projects data into the LDA space.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Projected features.
        """
        X = self.scaler.transform(X)
        return np.dot(X, self.components_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities based on distance to projected means.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Class probabilities.
        """
        X_transformed = self.transform(X)

        # Calculate distances to means in transformed space
        proba = np.zeros((X.shape[0], len(self.classes_)))
        for i, mean in enumerate(self.means_transformed_):
            diff = X_transformed - mean
            dist = np.sum(diff**2, axis=1)
            proba[:, i] = -dist  # Negative distance as log-probability

        # Convert to probabilities using softmax
        proba = np.exp(proba - np.max(proba, axis=1, keepdims=True))
        proba /= np.sum(proba, axis=1, keepdims=True)

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for the provided features.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
