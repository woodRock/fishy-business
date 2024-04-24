"""
Data - data.py
==============

This module contains helper methods that process the data.
The data is stored in text form as a matlab file.
This file is imported, converted into the correct datatype, and normalized in range [0,1].
"""

import scipy.io
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


def load_data(dataset="Fish", folder="../data/matlab/"):
    """ Load a dataset from a matlab file.

    Args:
        dataset: The name of the dataset, defaults to "Fish".
        folder: Relative path to folder location, defaults to "../data/matlab/".
    Returns:
        X,y: the features (X) and labels (y), respectively.
    """
    mat = scipy.io.loadmat(folder + dataset + '.mat')
    X = mat['X']
    X = X.astype(float)
    y = mat['Y']
    y = y[:, 0]
    return X, y


def get_labels(y):
    """Convert labels from strings to numbers.

    Args:
        y: Labels as strings.

    Returns:
        y_: Labels encoded as numbers.
        labels: A dictionary to retrieve the string versions from.
    """
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y_ = le.transform(y)
    labels = le.inverse_transform(np.unique(y_))
    return y_, labels


def normalize(X_train, X_test):
    """ Normalize the features within a range.

    Args:
        X_train: the training set.
        X_test: the test set.

    Returns:
        X_train, X_test: both the training and test sets.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test
