import logging 
import numpy as np
import os
import pandas as pd
from typing import Iterable, Union

def load_dataset(
        dataset: str = "species"
    ) -> Union[Iterable, Iterable]:
    """Load and prepare the dataset from an excel spreadsheet.

    This method loads the dataset from an excel spreadsheet.
    The task is specified with the `dataset` argument. 
    There is a choice between species, part, oil or cross-species.
    An exception is thrown, if no valid dataset is specified.

    Args: 
        dataset (str): the species, part, oil or cross-species dataset

    Returns: 
        X,y (np.array, np.array): Returns the dataset split into features X, and class labels y.
    """
    logger = logging.getLogger(__name__)

    # Path for university computers
    path = ["/", "vol", "ecrg-solar", "woodj4", "fishy-business", "data", "REIMS_data.xlsx"]
    # Path for home computer
    # path = ["~/", "Desktop", "fishy-business", "data", "REIMS_data.xlsx"]

    path = os.path.join(*path)

    # Load the dataset
    logger.info(f"Reading dataset fish: {dataset}")
    data = pd.read_excel(path)
    y = []
    # Remove the quality control samples.
    data = data[~data['m/z'].str.contains('QC')]
    
    # Exclude cross-species samples from the dataset.
    if dataset == "species" or dataset == "part" or dataset == "oil":
        data = data[~data['m/z'].str.contains('HM')]
    
    # Exclude mineral oil samples from the dataset.
    if dataset == "species" or dataset == "part" or dataset == "cross-species":
        data = data[~data['m/z'].str.contains('MO')]
    
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
    elif dataset == "oil":
        y = data['m/z'].apply(lambda x:
                          0 if 'MO 50' in x
                    else (1 if 'MO 25' in x
                    else (2 if 'MO 10' in x
                    else (3 if 'MO 05' in x
                    else (4 if 'MO 01' in x
                    else (5 if 'MO 0.1' in x
                    else (6 if 'MO 0' in x
                    else None )))))))
    elif dataset == "oil_simple":
        # Binary encodings for class labels (1 for Oil, 0 for No Oil)
        # Oil contaminated samples contain 'MO' in their class label.
        y = data['m/z'].apply(lambda x: 1 if 'MO' in x else 0)
    elif dataset == "cross-species":
        # Mutli-label encodings for class labels (1 for Hoki, 2 for Mackeral, 3 for Cross-species)
        # Cross-species contaminated samples contain 'HM' in their class label.
        y = data['m/z'].apply(lambda x: 
                              0 if 'HM' in x
                        else (1 if 'H' in x 
                        else (2 if 'M' in x
                        else None)))
    elif dataset == "cross-species-hard":
        # Mutli-label encodings for class labels (1 for Hoki, 2 for Mackeral, 3 for Cross-species)
        # Cross-species contaminated samples contain 'HM' in their class label.
        data = data[~data.iloc[:, 0].astype(str).str.contains('QC|MO|fillet|frames|gonads|livers|skins|guts|frame|heads', case=False, na=False)]    
        y = data['m/z'].apply(lambda x:
                          0 if 'HM 01' in x
                    else (1 if 'HM 02' in x
                    else (2 if 'HM 03' in x
                    else (3 if 'HM 04' in x
                    else (4 if 'HM 05' in x
                    else (5 if 'HM 06' in x
                    else (6 if 'HM 07' in x
                    else (7 if 'HM 08' in x  
                    else (8 if 'HM 09' in x
                    else (9 if 'HM 10' in x
                    else (10 if 'HM 11' in x
                    else (11 if 'HM 12' in x
                    else (12 if 'HM 13' in x
                    else (13 if 'HM 14' in x
                    else (14 if 'HM 15' in x
                    else None )))))))))))))))
    elif dataset == "instance-recognition":
        data = data[~data.iloc[:, 0].astype(str).str.contains('QC|HM|MO|fillet|frames|gonads|livers|skins|guts|frame|heads', case=False, na=False)]    
        X = data.iloc[:, 1:].to_numpy() 
        # Take only the class label column.
        y = data.iloc[:, 0].to_numpy()
        features = list() 
        labels = list() 

        all_possible_pairs = [((a, a_idx), (b, b_idx)) for a_idx, a in enumerate(X) for b_idx, b in enumerate(X[a_idx + 1:])]
        for (a, a_idx), (b, b_idx) in all_possible_pairs:
            concatenated = np.concatenate((a, b))
            label = int(y[a_idx] == y[b_idx])
            features.append(concatenated)
            labels.append(label)
        X,y = np.array(features), np.array(labels)
        # We don't want onehot encoding for multi-tree GP.
        # y = np.eye(2)[y]
        return X,y
    else: 
        # Return an excpetion if the dataset is not valid.
        raise ValueError(f"No valid dataset was specified: {dataset}")
   
    X = data.drop('m/z', axis=1) # X contains only the features.
    y = np.array(y)

    # Remove the classes that are not related to this dataset, 
    # i.e. the instances whose class is None are discarded.
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
    logger.info(f"Number of features: {n_features}")
    logger.info(f"Number of instances: {n_instances}")
    logger.info(f"Number of classes {n_classes}.")
    return X,y 
