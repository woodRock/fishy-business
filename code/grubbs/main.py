import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from data import load_dataset
from typing import Iterable

class DetectFliers:        
    def fit(self, 
            X: Iterable
    ) -> None: 
        outlier_indexes = []
        for i,_ in enumerate(range(X[0].shape[0])):
            feature_slice = X[:,i]
            # print(f"feature_slice: {feature_slice}")
            q3, q1 = np.percentile(feature_slice, [75 ,25])
            iqr = q3 - q1
            lower = (q1 - 1.5 * iqr)
            upper = (q3 + 1.5 * iqr)
            lower_idxs = np.where(lower >= feature_slice)[0]
            upper_idxs = np.where(feature_slice <= upper)[0]
            idx = np.concatenate((lower_idxs,upper_idxs))
            outlier_indexes.append(idx)
        self.outlier_indexes = outlier_indexes


    def predict(self,
            X: Iterable
    ) -> Iterable:
        predictions = []
        outliers_exist = False
        for i,_ in enumerate(range(len(X))):
            outliers_exist = np.any(i in out for out in self.outlier_indexes)
            prediction = 1 if outliers_exist else 0
            predictions.append(prediction)
        return predictions

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    # Run argument for numbered log files.
    output = f"logs/out.log"
    # Filemode is write, so it clears the file, then appends output.
    logging.basicConfig(filename=output, level=logging.INFO, filemode='w')

    dataset = "oil"
    X,y = load_dataset(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    model = DetectFliers()
    
    model.fit(X_train)
    y_preds = model.predict(X_train)
    train_accuracy = balanced_accuracy_score(y_train, y_preds)

    model.fit(X_test)
    y_preds = model.predict(X_test)
    test_accuracy = balanced_accuracy_score(y_test, y_preds)

    logger.info(f"Training: {train_accuracy}, Test: {test_accuracy}")