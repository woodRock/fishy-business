# run_logreg_gridsearch.py
#
# Dataset: species
# Best average balanced accuracy found: 1.0000
# Best parameters found:
#   - C: 1
#   - penalty: l1
#   - solver: liblinear
#
# Dataset: part
# Best average balanced accuracy found: 0.7222
# Best parameters found:
#   - C: 1
#   - penalty: l2
#   - solver: liblinear
#
# Dataset: oil
# Best average balanced accuracy found: 0.3905
# Best parameters found:
#   - C: 10
#   - penalty: l1
#   - solver: saga
#
# Dataset: cross-species
# Best average balanced accuracy found: 0.8874
# Best parameters found:
#   - C: 10
#   - l1_ratio: 0.5
#   - penalty: elasticnet
#   - solver: saga

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.impute import SimpleImputer
import warnings
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning

from data import load_dataset

# --- Configuration ---
target_threshold = 0.7143
random_seed = 42
# ---------------------

# Suppress warnings for cleaner output during grid search
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FitFailedWarning)


def main(dataset="species"):
    print(f"Starting Logistic Regression GridSearchCV script...")
    X, y = load_dataset(dataset=dataset)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    print(
        f"Found {len(numerical_features)} numerical features and {len(categorical_features)} categorical features."
    )

    # --- Preprocessing Setup ---
    print("Setting up preprocessing pipelines...")
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",  # Keep columns not specified, if any
    )
    # ---------------------------

    # --- Pipeline Definition ---
    # Define the pipeline with the base classifier
    # Use high max_iter for solvers like 'saga', and balanced class weight
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    random_state=random_seed, max_iter=3000, class_weight="balanced"
                ),
            ),
        ]
    )
    # ---------------------------

    # --- Parameter Grid Definition ---
    # Separate dictionaries for compatible solver/penalty combinations
    param_grid = [
        {
            "classifier__solver": ["liblinear"],
            "classifier__penalty": ["l1", "l2"],
            "classifier__C": [0.01, 0.1, 1, 10, 100],
        },
        {
            "classifier__solver": ["saga"],
            "classifier__penalty": ["l1", "l2"],
            "classifier__C": [0.01, 0.1, 1, 10, 100],
        },
        {
            "classifier__solver": ["saga"],
            "classifier__penalty": ["elasticnet"],
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__l1_ratio": [0.2, 0.5, 0.8],  # Only for elasticnet + saga
        },
        # Optional: Add other solvers like lbfgs if desired
        # {
        #     'classifier__solver': ['lbfgs'],
        #     'classifier__penalty': ['l2'], # lbfgs only supports 'l2' or None
        #     'classifier__C': [0.01, 0.1, 1, 10, 100]
        # }
    ]
    print(f"Defined parameter grid for GridSearchCV.")
    # -------------------------------

    # --- CV and Scorer Definition ---
    # Check n_splits against smallest class size for robustness
    min_class_count = y.value_counts().min()

    # Set n_cv_splits based on dataset
    n_cv_splits = 3 if dataset == "part" else 5
    actual_n_splits = min(n_cv_splits, min_class_count)

    if actual_n_splits < n_cv_splits:
        print(
            f"Warning: The smallest class has only {min_class_count} members. "
            f"Reducing n_splits for StratifiedKFold from {n_cv_splits} to {actual_n_splits}."
        )

    if actual_n_splits < 2:
        print(
            f"Error: The smallest class has only {min_class_count} members. "
            f"Cannot perform cross-validation with n_splits={actual_n_splits}."
        )
        return

    cv = StratifiedKFold(
        n_splits=actual_n_splits, shuffle=True, random_state=random_seed
    )
    balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)
    # --------------------------------

    # --- GridSearchCV Execution ---
    print(
        f"Starting GridSearchCV for Logistic Regression using {actual_n_splits}-fold CV..."
    )
    print("This may take a few minutes...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=balanced_accuracy_scorer,
        cv=cv,
        n_jobs=-1,  # Use all available CPU cores
        error_score=0.0,  # Assign score of 0 if a parameter combination fails
    )

    try:
        # Fit the grid search
        grid_search.fit(X, y)

        # --- Results ---
        print("\n--- GridSearchCV Results for Logistic Regression ---")
        best_score = grid_search.best_score_
        print(f"Best average balanced accuracy found: {best_score:.4f}")

        print("Best parameters found:")
        best_params = grid_search.best_params_

        for param in sorted(best_params.keys()):
            # Only print relevant params (e.g., don't print l1_ratio if not elasticnet)

            if (
                param == "classifier__l1_ratio"
                and best_params.get("classifier__penalty") != "elasticnet"
            ):
                continue
            # Print cleaned parameter name and its value
            print(f"  - {param.replace('classifier__', '')}: {best_params[param]}")

        # Check against threshold
        print("\n--- Comparison to Threshold ---")
        print(f"Required average balanced accuracy threshold: {target_threshold}")
        if best_score > target_threshold:
            print(f"SUCCESS: The best score ({best_score:.4f}) exceeds the threshold!")
        else:
            print(
                f"INFO: The best score ({best_score:.4f}) did not exceed the threshold."
            )
        # -------------

    except ValueError as ve:
        if "Cannot have number of splits n_splits=" in str(ve):
            print(
                f"\nError during GridSearchCV execution: {ve}. "
                "This often happens if a class has too few members for the number of CV folds."
            )
        else:
            print(
                f"\nAn unexpected ValueError occurred during the GridSearchCV execution: {ve}"
            )
    except Exception as e:
        print(f"\nAn unexpected error occurred during the GridSearchCV execution: {e}")

    print("\nScript finished.")


# --- Run the main function ---
if __name__ == "__main__":
    # Take dataset as a command line argument with flag -d.
    # If no argument is given, use the default dataset "species".
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Logistic Regression GridSearchCV."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="species",
        help="Dataset to use (default: species).",
    )
    args = parser.parse_args()
    dataset = args.dataset
    print(f"Dataset specified: {dataset}")
    if dataset not in ["species", "part", "oil", "cross-species"]:
        print(f"Invalid dataset specified: {dataset}. Using default dataset 'species'.")
        dataset = "species"
    # Call the main function with the specified dataset
    main(dataset=dataset)
# -----------------------------
