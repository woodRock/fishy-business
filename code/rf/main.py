import argparse
import logging
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from data import load_dataset

if __name__ == "__main__":
    # Handle the command line arguments for the script.
    parser = argparse.ArgumentParser(
                    prog='Random Forest',
                    description='A random forest for fish species classification.',
                    epilog='Implemented in sklearn and written in python.')
    parser.add_argument('-d', '--dataset', type=str, default="species",
                        help="The fish species or part dataset. Defaults to species")
    args = vars(parser.parse_args())

    datasets = ["species","part","oil","cross-species"]
    dataset = args['dataset']
    if dataset not in datasets:
        raise ValueError(f"Invalid dataset specified: {dataset}")
    
    logger = logging.getLogger(__name__)
    # Run argument for numbered log files.
    output = f"logs/{dataset}/out.log"
    # Filemode is write, so it clears the file, then appends output.
    logging.basicConfig(filename=output, level=logging.INFO, filemode='w')

    # Load the dataset.
    X,y = load_dataset(dataset)
        
    # The different models to try out.
    models = {'rf': rf(), 'knn': knn(), 'dt': dt()}
    
    runs = 30
    logger.info(f"Running {runs} experiments")
    # Evaluate for two
    for name, model in models.items():
        train_accs = []
        test_accs = []
        # Perform 30 indepdnent runs of the random forest.
        for run in tqdm(range(1, runs+1), desc=f"{name}"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=run)
            # Training the classifier.
            model.fit(X_train, y_train)
            # Train evaluation.
            y_pred = model.predict(X_train)
            train_acc = balanced_accuracy_score(y_train, y_pred)
            train_accs.append(train_acc)
            # Test evaluation.
            y_pred = model.predict(X_test)
            test_acc = balanced_accuracy_score(y_test, y_pred)
            test_accs.append(test_acc)
        
        # Convert to numpy arrays.
        train_accs = np.array(train_accs)
        test_accs = np.array(test_accs)

        # Display the results of the 30 independent runs.
        mean, std = np.mean(train_accs), np.std(train_accs)
        logger.info(f"Classifier: {name}")
        logger.info(f"training: {mean} +\- {std}")
        mean, std = np.mean(test_accs), np.std(test_accs)
        logger.info(f"test: {mean} +\- {std}")