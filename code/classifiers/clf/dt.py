# run_dt_gridsearch.py (Updated filename suggestion)
#
# Example output structure (will vary based on actual runs):
# Dataset: species
# Best average balanced accuracy found: 0.9500
# Best parameters found:
#   - criterion: gini
#   - max_depth: 5
#   - min_samples_leaf: 1
#   - min_samples_split: 2
#   - ccp_alpha: 0.0
# ... (similar outputs for other datasets)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, StratifiedGroupKFold # Added StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier  # Changed from LogisticRegression
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.impute import SimpleImputer
import warnings
from sklearn.exceptions import (
    ConvergenceWarning,
    FitFailedWarning,
)  # FitFailedWarning might still be relevant

# --- Plotting Imports ---
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for scripts
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------

from data import load_dataset  # Assuming data.py and load_dataset are available

# --- Configuration ---
target_threshold = (
    0.7143  # This threshold can be adjusted based on expectations for Decision Trees
)
random_seed = 42
# ---------------------

# Suppress warnings
warnings.filterwarnings(
    "ignore", category=ConvergenceWarning
)  # Less relevant for DT, but harmless
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FitFailedWarning)


# --- Plotting Function for Feature Importances ---
def plot_feature_importances(importances, feature_names, title, dataset_name, top_n=30):
    """
    Visualizes the feature importances of a Decision Tree model.

    Args:
        importances (np.ndarray): The model's feature importances.
        feature_names (list): The names of the features.
        title (str): The base title for the plot.
        dataset_name (str): Name of the dataset for filename uniqueness.
        top_n (int, optional): Display top N features by importance. None to display all.
    """
    if len(importances) != len(feature_names):
        print(
            f"Error: Mismatch in number of importances ({len(importances)}) and feature names ({len(feature_names)})."
        )
        print(f"Cannot generate plot for: {title}")
        return

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    # Sort by importance for consistent plotting and top_n selection
    importance_df = importance_df.sort_values("importance", ascending=False)

    if top_n and top_n < len(importance_df):
        plot_df = importance_df.head(top_n)
        plot_title_full = f"{title} (Top {top_n} Features)"
    else:
        plot_df = importance_df  # plot_df is already sorted by importance
        plot_title_full = title

    plt.figure(figsize=(12, max(6, len(plot_df) * 0.38)))
    sns.barplot(
        x="importance", y="feature", data=plot_df, palette="viridis_r"
    )  # Using a sequential palette
    plt.title(plot_title_full)
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()

    safe_base_title = "".join(c if c.isalnum() else "_" for c in title)
    plot_filename = f"dt_feature_importances_{dataset_name}_{safe_base_title}.png"

    try:
        plt.savefig(plot_filename)
        print(f"Saved feature importance visualization to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot '{plot_filename}': {e}")
    plt.close()


# ---------------------------------------------


def main(dataset="species"):
    print(
        f"Starting Decision Tree Classifier GridSearchCV script for dataset: {dataset}..."
    )  # Updated
    X_original, y_original, groups_original = load_dataset(dataset=dataset) # Modified
    X = pd.DataFrame(X_original) # Modified
    y = pd.Series(y_original) # Modified

    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    print(
        f"Found {len(numerical_features)} numerical features and {len(categorical_features)} categorical features."
    )

    # --- Preprocessing Setup (remains the same) ---
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
        remainder="passthrough",
    )
    # ---------------------------

    # --- Pipeline Definition (Classifier changed to DecisionTreeClassifier) ---
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(  # Changed here
                    random_state=random_seed, class_weight="balanced"
                ),
            ),
        ]
    )
    # ---------------------------

    # --- Parameter Grid Definition (Updated for DecisionTreeClassifier) ---
    param_grid = [
        {
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [None, 5, 10, 15, 20, 30],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 5, 10],
            "classifier__ccp_alpha": [
                0.0,
                0.001,
                0.005,
                0.01,
            ],  # Cost-complexity pruning
        }
    ]
    print(f"Defined parameter grid for GridSearchCV (Decision Tree).")
    # -------------------------------

    # --- CV and Scorer Definition (Updated for StratifiedGroupKFold) ---
    n_cv_splits = 3 # Fixed to 3 as per request
    cv = StratifiedGroupKFold(
        n_splits=n_cv_splits, shuffle=True, random_state=random_seed
    )
    balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)
    # --------------------------------

    # --- GridSearchCV Execution ---
    print(
        f"Starting GridSearchCV for Decision Tree Classifier using {n_cv_splits}-fold CV..."  # Updated
    )
    print("This may take a few minutes...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=balanced_accuracy_scorer,
        cv=cv,
        n_jobs=-1,
        error_score=0.0,
    )

    try:
        grid_search.fit(X, y, groups=groups_original)

        print("\n--- GridSearchCV Results for Decision Tree Classifier ---")  # Updated
        best_score = grid_search.best_score_
        print(f"Best average balanced accuracy found: {best_score:.4f}")

        print("Best parameters found:")
        best_params = grid_search.best_params_
        for param_name_full in sorted(best_params.keys()):
            if param_name_full.startswith(
                "classifier__"
            ):  # Print only classifier hyperparams
                param_name_short = param_name_full.replace("classifier__", "")
                print(f"  - {param_name_short}: {best_params[param_name_full]}")

        # --- Visualizing Feature Importances (Updated from weights) ---
        print(
            "\n--- Visualizing Feature Importances for Best Decision Tree Model ---"
        )  # Updated
        best_pipeline = grid_search.best_estimator_
        dt_model = best_pipeline.named_steps[
            "classifier"
        ]  # Now a DecisionTreeClassifier
        fitted_preprocessor = best_pipeline.named_steps["preprocessor"]

        feature_names = None
        try:
            feature_names = list(fitted_preprocessor.get_feature_names_out())
        except AttributeError:
            print(
                "Warning: `get_feature_names_out` not available on the preprocessor object."
            )
            print(
                "This typically means an older scikit-learn version. Consider upgrading for full feature name support."
            )
            print("Skipping feature importance visualization.")
        except Exception as e_fn:
            print(f"An error occurred while trying to get feature names: {e_fn}")
            print("Skipping feature importance visualization.")

        if feature_names:
            importances = dt_model.feature_importances_  # Use feature_importances_
            # Feature importances are a single array, no need to distinguish binary/multi-class like coefficients
            plot_title = "Feature Importances for Best Decision Tree"
            plot_feature_importances(
                importances, feature_names, plot_title, dataset_name=dataset
            )
        # -------------------------------------------------------------

        print("\n--- Comparison to Threshold ---")
        print(f"Required average balanced accuracy threshold: {target_threshold}")
        if best_score > target_threshold:
            print(f"SUCCESS: The best score ({best_score:.4f}) exceeds the threshold!")
        else:
            print(
                f"INFO: The best score ({best_score:.4f}) did not exceed the threshold."
            )

    except ValueError as ve:
        if "Cannot have number of splits n_splits=" in str(
            ve
        ) or "The least populated class" in str(ve):
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
        import traceback

        traceback.print_exc()

    print(f"\nScript finished for dataset: {dataset}.")


# --- Run the main function ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Decision Tree Classifier GridSearchCV with feature importance visualization."  # Updated
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="species",
        help="Dataset to use (default: species). Valid options: species, part, oil, cross-species.",
        choices=["species", "part", "oil", "cross-species"],
    )
    args = parser.parse_args()

    print(f"Dataset specified via command line: {args.dataset}")
    main(dataset=args.dataset)
# -----------------------------
