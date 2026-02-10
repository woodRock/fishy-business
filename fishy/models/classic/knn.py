import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import euclidean_distances
from scipy.special import softmax


class knn(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, n_neighbors=5, class_weights=None, batch_size=1000):
        self.n_neighbors = n_neighbors
        self.class_weights = class_weights
        self.batch_size = batch_size

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)

        # Set up class weights if not provided
        if self.class_weights is None:
            class_counts = np.bincount(y)
            self.class_weights = len(y) / (len(self.classes_) * class_counts)
        return self

    def _get_neighbors(self, X_batch):
        # Calculate distances for this batch
        distances = euclidean_distances(X_batch, self.X_train)

        # Get indices of k nearest neighbors
        nearest_neighbors = np.argpartition(distances, self.n_neighbors, axis=1)
        nearest_neighbors = nearest_neighbors[:, : self.n_neighbors]

        # Get actual distances for these neighbors
        neighbor_distances = np.take_along_axis(distances, nearest_neighbors, axis=1)

        return nearest_neighbors, neighbor_distances

    def predict_proba(self, X):
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, len(self.classes_)))

        # Process in batches to reduce memory usage
        for i in range(0, n_samples, self.batch_size):
            batch_slice = slice(i, min(i + self.batch_size, n_samples))
            X_batch = X[batch_slice]

            # Get neighbors and their distances
            indices, distances = self._get_neighbors(X_batch)

            # Convert distances to weights using softmax
            weights = softmax(-distances, axis=1)

            # Calculate probabilities for this batch
            batch_proba = np.zeros((len(X_batch), len(self.classes_)))
            for j, sample_neighbors in enumerate(indices):
                neighbor_labels = self.y_train[sample_neighbors]
                for k, c in enumerate(self.classes_):
                    class_mask = neighbor_labels == c
                    batch_proba[j, k] = np.sum(
                        weights[j, class_mask] * self.class_weights[k]
                    )

            # Normalize probabilities
            batch_proba /= batch_proba.sum(axis=1, keepdims=True)
            proba[batch_slice] = batch_proba

        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
