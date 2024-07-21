import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from pso import PSO

class PSORandomForestClassifier:
    def __init__(self, n_particles, n_iterations, c1, c2, w, n_classes, n_features, 
                 w_start=0.9, w_end=0.4, n_estimators=100, random_state=42):
        self.pso = PSO(n_particles, n_iterations, c1, c2, w, n_classes, n_features, w_start, w_end)
        self.rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.device = self.pso.device

    def fit(self, train_loader, val_loader, patience=10):
        # First, fit the PSO model
        self.pso.fit(train_loader, val_loader, patience)

        # Then, use PSO's best particle to transform the data
        X_train, y_train = self._transform_data(train_loader)

        # Fit the Random Forest on the transformed data
        self.rf.fit(X_train, y_train)

    def _transform_data(self, data_loader):
        X_list, y_list = [], []
        self.pso.gbest = self.pso.gbest.to(self.device)
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                transformed_X = X_batch @ self.pso.gbest.T
                X_list.append(transformed_X.cpu().numpy())
                y_list.append(y_batch.cpu().numpy())
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

    def predict(self, data_loader):
        # Transform the data using PSO's best particle
        X_test, _ = self._transform_data(data_loader)

        # Use Random Forest to make predictions
        return self.rf.predict(X_test)

    def predict_proba(self, data_loader):
        # Transform the data using PSO's best particle
        X_test, _ = self._transform_data(data_loader)

        # Use Random Forest to make probability predictions
        return self.rf.predict_proba(X_test)
    
    def evaluate(self, data_loader):
        X_test, y_test = self._transform_data(data_loader)
        return self.rf.score(X_test, y_test)
    