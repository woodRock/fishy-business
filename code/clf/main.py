import logging
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.linear_model import LogisticRegression as lr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB as nb
from sklearn.svm import SVC as svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from data import load_dataset

if __name__ == "__main__":
    datasets = ["species","part","oil","cross-species"]

    logger = logging.getLogger(__name__)
    # Run argument for numbered log files.
    output = f"logs/out.log"
    # Filemode is write, so it clears the file, then appends output.
    logging.basicConfig(filename=output, level=logging.INFO, filemode='w')
    
    for dataset in tqdm(datasets, desc=f"Training"):
        print(f"Dataset: {dataset}")
        
        # Load the dataset.
        X,y = load_dataset(dataset)
            
        # The different models to try out.
        models = { 
            'knn': knn(), 
            'dt': dt(), 
            'lda': lda(), 
            'lr': lr(max_iter=2000), 
            'nb': nb(), 
            'rf': rf(), 
            'svm': svm(kernel='linear'), 
            'ensemble': VotingClassifier(
                estimators=[('knn', knn()),('nb', nb()), ('dt', dt()), ('rf', rf()), ('svm', svm(kernel='linear')), ('lda', lda()), ('lr', lr(max_iter=2000))],
                voting='hard'),
        }
        
        runs = 30
        logger.info(f"Running {runs} experiments")
        # Evaluate for two
        for name, model in models.items():
            train_accs = []
            test_accs = []
            # Perform 30 indepdnent runs of the random forest.
            for run in tqdm(range(1, runs+1), desc=f"{dataset} - {name}"):
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