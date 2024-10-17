import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import norm

class nb(BaseEstimator, ClassifierMixin):
    def __init__(self, class_weights=None, var_smoothing=1e-9):
        self.class_weights = class_weights
        self.var_smoothing = var_smoothing
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Initialize class weights
        if self.class_weights is None:
            class_counts = np.bincount(y)
            self.class_weights_ = len(y) / (n_classes * class_counts)
        else:
            self.class_weights_ = np.array([self.class_weights[c] for c in range(n_classes)])
        
        # Initialize parameters
        self.theta_ = np.zeros((n_classes, n_features))  # Mean
        self.sigma_ = np.zeros((n_classes, n_features))  # Variance
        self.class_priors_ = np.zeros(n_classes)
        
        # Calculate weighted parameters for each class
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            weight_c = self.class_weights_[i]
            
            # Weighted mean
            self.theta_[i, :] = np.average(X_c, weights=np.full(X_c.shape[0], weight_c), axis=0)
            
            # Weighted variance
            diff_sq = (X_c - self.theta_[i, :]) ** 2
            self.sigma_[i, :] = np.average(diff_sq, weights=np.full(X_c.shape[0], weight_c), axis=0)
            
            # Add smoothing to variance
            self.sigma_[i, :] = self.sigma_[i, :] + self.var_smoothing
            
            # Weighted class prior
            self.class_priors_[i] = np.sum(y == c) * weight_c
            
        # Normalize class priors
        self.class_priors_ /= np.sum(self.class_priors_)
        
        return self
    
    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []
        for i in range(len(self.classes_)):
            # Calculate log probability for each feature
            log_probs = norm.logpdf(X, self.theta_[i, :], np.sqrt(self.sigma_[i, :]))
            
            # Sum log probabilities and add log prior
            joint_log_likelihood.append(
                np.sum(log_probs, axis=1) + 
                np.log(self.class_priors_[i])
            )
            
        return np.array(joint_log_likelihood).T
    
    def predict_proba(self, X):
        log_prob = self._joint_log_likelihood(X)
        # Normalize log probabilities
        log_prob_norm = log_prob - np.max(log_prob, axis=1)[:, np.newaxis]
        proba = np.exp(log_prob_norm)
        proba /= np.sum(proba, axis=1)[:, np.newaxis]
        return proba
    
    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]
    
    def score(self, X, y):
        """Weighted accuracy score"""
        y_pred = self.predict(X)
        weights = np.array([self.class_weights_[yi] for yi in y])
        return np.average(y_pred == y, weights=weights)

# Example usage and testing
def test_weighted_gnb():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt
    
    # Generate imbalanced dataset
    X, y = make_classification(
        n_samples=1000,
        n_classes=3,
        n_features=20,
        n_informative=15,
        weights=[0.6, 0.3, 0.1],  # Imbalanced classes
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Calculate class weights
    unique, counts = np.unique(y_train, return_counts=True)
    class_weights = dict(zip(unique, len(y_train) / (len(unique) * counts)))
    
    # Train classifier
    clf = nb(class_weights=class_weights)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    
    # Print results
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualize probability distributions
    def plot_feature_distributions(X, y, feature_idx=0):
        plt.figure(figsize=(10, 6))
        for c in np.unique(y):
            X_c = X[y == c][:, feature_idx]
            plt.hist(X_c, bins=30, alpha=0.5, 
                    label=f'Class {c}', density=True)
            
            # Plot fitted Gaussian
            x_range = np.linspace(X_c.min(), X_c.max(), 100)
            plt.plot(x_range, 
                    norm.pdf(x_range, 
                            clf.theta_[c, feature_idx], 
                            np.sqrt(clf.sigma_[c, feature_idx])),
                    label=f'Fitted Class {c}')
        
        plt.title(f'Feature {feature_idx} Distribution by Class')
        plt.xlabel('Feature Value')
        plt.ylabel('Density')
        plt.legend()
        plt.show()
    
    try:
        # Plot distributions for first feature
        plot_feature_distributions(X_test, y_test)
    except ImportError:
        print("Matplotlib not available for visualization")

if __name__ == "__main__":
    test_weighted_gnb()