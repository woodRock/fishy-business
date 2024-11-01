import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd

class lda(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=None, class_weights=None, solver='svd'):
        self.n_components = n_components
        self.class_weights = class_weights
        self.solver = solver
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        if self.n_components is None:
            self.n_components = min(n_classes - 1, X.shape[1])
            
        # Set up class weights if not provided
        if self.class_weights is None:
            class_counts = np.bincount(y)
            self.class_weights = len(y) / (n_classes * class_counts)
        else:
            self.class_weights = np.array([self.class_weights[c] for c in range(n_classes)])
            
        # Calculate weighted means
        self.means_ = []
        for i, cls in enumerate(self.classes_):
            mask = y == cls
            weighted_mean = np.average(X[mask], weights=np.full(mask.sum(), self.class_weights[i]), axis=0)
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
            Sw += np.dot((X_centered * class_weights_expanded.reshape(-1, 1)).T, X_centered)
            
        # Calculate between-class scatter matrix (Sb)
        Sb = np.zeros_like(Sw)
        for i, mean in enumerate(self.means_):
            diff = (mean - self.overall_mean_).reshape(-1, 1)
            Sb += self.class_weights[i] * np.dot(diff, diff.T)
            
        # Solve the eigenvalue problem using SVD
        if self.solver == 'svd':
            # Calculate Sw^(-1/2)
            U, s, Vt = svd(Sw)
            # Add small constant to avoid division by zero
            s = np.maximum(s, 1e-10)
            Sw_sqrt_inv = np.dot(U * (1.0 / np.sqrt(s)), Vt)
            
            # Transform Sb
            Sb_transformed = np.dot(np.dot(Sw_sqrt_inv, Sb), Sw_sqrt_inv.T)
            
            # Perform SVD on transformed Sb
            U, s, Vt = svd(Sb_transformed)
            
            # Get eigenvectors
            self.components_ = np.dot(Sw_sqrt_inv.T, U[:, :self.n_components])
            
        # Project means
        self.means_transformed_ = np.dot(self.means_, self.components_)
        
        return self
    
    def transform(self, X):
        X = self.scaler.transform(X)
        return np.dot(X, self.components_)
    
    def predict_proba(self, X):
        X_transformed = self.transform(X)
        
        # Calculate distances to means in transformed space
        proba = np.zeros((X.shape[0], len(self.classes_)))
        for i, mean in enumerate(self.means_transformed_):
            diff = X_transformed - mean
            dist = np.sum(diff ** 2, axis=1)
            proba[:, i] = -dist  # Negative distance as log-probability
            
        # Convert to probabilities using softmax
        proba = np.exp(proba - np.max(proba, axis=1, keepdims=True))
        proba /= np.sum(proba, axis=1, keepdims=True)
        
        return proba
    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

# Example usage:
def test_weighted_lda():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    
    # Generate imbalanced dataset
    X, y = make_classification(
        n_samples=1000,
        n_classes=3,
        n_features=20,
        n_informative=15,
        weights=[0.6, 0.3, 0.1],
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define class weights (inverse class frequency)
    unique, counts = np.unique(y_train, return_counts=True)
    class_weights = dict(zip(unique, len(y_train) / (len(unique) * counts)))
    
    # Train classifier
    clf = lda(
        n_components=2,  # Reduce to 2 dimensions
        class_weights=class_weights
    )
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualize results if desired
    def plot_results(X_test, y_test, clf):
        import matplotlib.pyplot as plt
        
        X_transformed = clf.transform(X_test)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], 
                            c=y_test, cmap='viridis')
        plt.colorbar(scatter)
        
        # Plot transformed means
        for i, mean in enumerate(clf.means_transformed_):
            plt.plot(mean[0], mean[1], 'r*', markersize=15, 
                    label=f'Class {i} Mean')
        
        plt.title('LDA Projection with Class Weights')
        plt.xlabel('First discriminant')
        plt.ylabel('Second discriminant')
        plt.legend()
        plt.show()
    
    try:
        plot_results(X_test, y_test, clf)
    except ImportError:
        print("Matplotlib not available for visualization")

if __name__ == "__main__":
    test_weighted_lda()