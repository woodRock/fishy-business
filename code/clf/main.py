import logging
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.linear_model import LinearRegression as lr
from sklearn.linear_model import LogisticRegression as lor
from sklearn.svm import SVR as svr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.naive_bayes import GaussianNB as nb
from sklearn.svm import SVC as svm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from data import load_dataset

if __name__ == "__main__":
    datasets = ["species", "part", "oil_simple", "oil", "oil_regression", "cross-species"]
    
    logger = logging.getLogger(__name__)
    # Run argument for numbered log files.
    output = f"logs/out.log"
    # Filemode is write, so it clears the file, then appends output.
    logging.basicConfig(filename=output, level=logging.INFO, filemode='w')
    
    for dataset in tqdm(datasets, desc=f"Training"):
        print(f"Dataset: {dataset}")

        is_classification = dataset != "oil_regression"
        
        # Load the dataset.
        X,y = load_dataset(dataset)

        if is_classification:
            # The different models to try out.
            models = { 
                'knn': knn(), 
                'dt': dt(), 
                'lda': lda(), 
                'nb': nb(), 
                'rf': rf(), 
                'svm': svm(kernel='linear'), 
                'ensemble': VotingClassifier(
                    estimators=[('knn', knn()), ('dt', dt()), ('lda', lda()), ('nb', nb()), ('rf', rf()),('svm', svm(kernel='linear'))],
                    voting='hard'
                )
            }
        else:   
            # The different models to try out.
            models = { 
                'lr': lr(), 
                'svr': svr(),
                'xgb': xgb.XGBRegressor(),
                'ensemble': VotingRegressor(
                    estimators=[('lr', lr()), ('svr', svr()), ('xgb', xgb.XGBRegressor())],
                ) # voting='hard')
            }
        
        runs = 30
        logger.info(f"Running {runs} experiments")
        # Evaluate for two
        for name, model in models.items():
            train_accs = []
            test_accs = []
            # Perform 30 indepdnent runs of the random forest.
            loss =  balanced_accuracy_score if is_classification else mean_absolute_error
            for run in tqdm(range(1, runs+1), desc=f"{dataset} - {name}"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=run)
                # Training the classifier.
                model.fit(X_train, y_train)
                # Train evaluation.
                y_pred = model.predict(X_train)
                train_acc = loss(y_train, y_pred)
                train_accs.append(train_acc)
                # Test evaluation.
                y_pred = model.predict(X_test)
                test_acc = loss(y_test, y_pred)
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