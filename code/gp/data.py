import logging 
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def load_dataset(dataset="species"):
    logger = logging.getLogger(__name__)

    path = ['~/Desktop', 'fishy-business', 'data','REIMS_data.xlsx']
    path = os.path.join(*path)

    # Load the dataset
    data = pd.read_excel(path)

    logger.info(f"Reading dataset fish: {dataset}")
    raw = pd.read_excel(path)

    data = raw[~raw['m/z'].str.contains('HM')]
    data = data[~data['m/z'].str.contains('QC')]
    data = data[~data['m/z'].str.contains('HM')]
    X = data.drop('m/z', axis=1) # X contains only the features.
    y = [] 
    if dataset == "species":
        # Binary encodings for class labels (1 for Hoki, 0 for Mackeral)
        y = data['m/z'].apply(lambda x: 1 if 'H' in x else 0)
    elif dataset == "part":
        y = data['m/z'].apply(lambda x:
                          0 if 'Fillet' in x
                    else  1 if 'Heads' in x
                    else (2 if 'Livers' in x
                    else (3 if 'Skins' in x
                    else (4 if 'Guts' in x
                    else (5 if 'Frames' in x
                    else None )))))  # For fish parts
    y = np.array(y)

    xs = []
    ys = []
    for (x,y) in zip(X.to_numpy(),y):
        if y is not None and not np.isnan(y):
            xs.append(x)
            ys.append(y)
    X = np.array(xs)
    y = np.array(ys)

    classes, class_counts = np.unique(y, axis=0, return_counts=True)
    n_features = X.shape[1]
    n_instances = X.shape[0]
    n_classes = len(np.unique(y, axis=0))
    class_ratios = np.array(class_counts) / n_instances

    logger.info(f"Class Counts: {class_counts}, Class Ratios: {class_ratios}")
    logger.info(f"Number of features: {n_features}\nNumber of instances: {n_instances}\nNumber of classes {n_classes}.")
    return X,y 
