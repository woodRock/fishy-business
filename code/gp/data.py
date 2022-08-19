"""
Data - data.py
==============

This is the data module. It contains the functions for loading, preparing, normalizing and encoding the data.
"""

import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import scipy.io

def encode_labels(y, y_test=None):
    """
    Convert text labels to numbers.

    Args:
        y: The labels.
        y_test: The test labels. Defaults to None.
    """
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    if y_test is not None:
        y_test = le.transform(y_test)
    return y, y_test, le

def load(filename, folder=''):
    """
    Load the data from the mat file.

    Args:
        filename: The name of the mat file.
        folder: The folder where the mat file is located.
    """
    path = folder + filename
    mat = scipy.io.loadmat(path)
    return mat

def prepare(mat):
    """
    Load the data from matlab format into memory. 

    Args:
        mat: The data in matlab format.
    """
    X = mat['X']   
    X = X.astype(float)
    y = mat['Y']    
    y = y[:, 0]
    return X,y

def normalize(X_train, X_test):
    """
    Normalize the input features within range [0,1].

    Args:
        X_train: The training data.
        X_test: The test data.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test
