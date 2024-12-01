import sys
from pathlib import Path

# Add the 'helper' folder to sys.path
sys.path.append(str(Path.cwd() / "helper"))

import gc
import os
from typing import List, Optional, Tuple

import catboost
import numpy as np
import optuna
import pandas as pd
from sklearn import metrics, model_selection, pipeline
from tqdm import tqdm

from helper import pre_process, utils


def run_catboost_CV(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 10,
    n_repeats: int = 1,
    pipeline: Optional[object] = None,
) -> Tuple[float, float]:
    """
    Perform Cross-Validation with CatBoost for regression.

    This function conducts Cross-Validation using CatBoost for regression tasks. It iterates
    through folds, trains CatBoost models, and computes the mean and standard deviation of the
    Root Mean Squared Error (RMSE) scores across folds.

    Parameters:
    - X (pd.DataFrame): The feature matrix.
    - y (pd.Series): The target variable.
    - n_splits (int, optional): The number of splits in K-Fold cross-validation.
      Defaults to 10.
    - n_repeats (int, optional): The number of times the K-Fold cross-validation is repeated.
      Defaults to 1.
    - pipeline (object, optional): Optional data preprocessing pipeline. If provided,
      it's applied to the data before training the model. Defaults to None.

    Returns:
    - Tuple[float, float]: A tuple containing the mean RMSE and standard deviation of RMSE
      scores across cross-validation folds.

    Example:
    ```python
    # Load your feature matrix (X) and target variable (y)
    X, y = load_data()

    # Perform Cross-Validation with CatBoost
    mean_rmse, std_rmse = run_catboost_CV(X, y, n_splits=5, n_repeats=2, pipeline=data_pipeline)

    print(f"Mean RMSE: {mean_rmse:.4f}")
    print(f"Standard Deviation of RMSE: {std_rmse:.4f}")
    ```

    Notes:
    - Ensure that the input data `X` and `y` are properly preprocessed and do not contain any
      missing values.
    - The function uses CatBoost for regression with optional data preprocessing via the `pipeline`.
    - RMSE is a common metric for regression tasks, and lower values indicate better model
      performance.
    """
    results = []

    # Extract feature names and data types
    # features = X.columns[~X.columns.str.contains("price")]
    # numerical_features = X.select_dtypes("number").columns.to_list()
    categorical_features = X.select_dtypes("object").columns.to_list()

    # Create a K-Fold cross-validator
    CV = model_selection.RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=utils.Configuration.seed
    )

    for train_fold_index, val_fold_index in CV.split(X):
        X_train_fold, X_val_fold = X.loc[train_fold_index], X.loc[val_fold_index]
        y_train_fold, y_val_fold = y.loc[train_fold_index], y.loc[val_fold_index]

        # Apply optional data preprocessing pipeline
        if pipeline is not None:
            X_train_fold = pipeline.fit_transform(X_train_fold)
            X_val_fold = pipeline.transform(X_val_fold)

        # Create CatBoost datasets
        catboost_train = catboost.Pool(
            X_train_fold,
            y_train_fold,
            cat_features=categorical_features,
        )
        catboost_valid = catboost.Pool(
            X_val_fold,
            y_val_fold,
            cat_features=categorical_features,
        )

        # Initialize and train the CatBoost model
        model = catboost.CatBoostRegressor(**utils.Configuration.catboost_params)
        model.fit(
            catboost_train,
            eval_set=[catboost_valid],
            early_stopping_rounds=utils.Configuration.early_stopping_round,
            verbose=utils.Configuration.verbose,
            use_best_model=True,
        )

        # Calculate OOF validation predictions
        valid_pred = model.predict(X_val_fold)

        RMSE_score = metrics.root_mean_squared_error(y_val_fold, valid_pred)

        del (
            X_train_fold,
            y_train_fold,
            X_val_fold,
            y_val_fold,
            catboost_train,
            catboost_valid,
            model,
            valid_pred,
        )
        gc.collect()

        results.append(RMSE_score)

    return np.mean(results), np.std(results)


class Optuna_Objective:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        """
        Initialize the Objective class with the dataset for hyperparameter optimization.

        Parameters:
        - X (pd.DataFrame): Features DataFrame.
        - y (pd.Series): Target Series.
        """
        self.X = X
        self.y = y

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function for tuning CatBoost hyperparameters.

        This function takes an Optuna trial and explores hyperparameters for a CatBoost
        model to minimize the Root Mean Squared Error (RMSE) using K-Fold cross-validation.

        Parameters:
        - trial (optuna.Trial): Optuna trial object for hyperparameter optimization.

        Returns:
        - float: Mean RMSE across K-Fold cross-validation iterations.

        Example use case:
        ```python
        # Create an Optuna study and optimize hyperparameters
        study = optuna.create_study(direction="minimize")
        study.optimize(Objective(X, y), n_trials=100)

        # Get the best hyperparameters
        best_params = study.best_params
        ```
        """
        catboost_params = {
            "iterations": trial.suggest_int("iterations", 10, 1000),
            "depth": trial.suggest_int("depth", 1, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1),
            "random_strength": trial.suggest_float("random_strength", 1e-9, 10),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 2, 30),
            "border_count": trial.suggest_int("border_count", 1, 255),
            "thread_count": os.cpu_count(),
        }

        results = []
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Extract feature names and data types
        # features = X.columns[~X.columns.str.contains("price")]
        # numerical_features = X.select_dtypes("number").columns.to_list()
        categorical_features = self.X.select_dtypes("object").columns.to_list()

        # Create a K-Fold cross-validator
        CV = model_selection.RepeatedKFold(
            n_splits=10, n_repeats=1, random_state=utils.Configuration.seed
        )

        for train_fold_index, val_fold_index in CV.split(self.X):
            X_train_fold, X_val_fold = (
                self.X.loc[train_fold_index],
                self.X.loc[val_fold_index],
            )
            y_train_fold, y_val_fold = (
                self.y.loc[train_fold_index],
                self.y.loc[val_fold_index],
            )

            # Create CatBoost datasets
            catboost_train = catboost.Pool(
                X_train_fold,
                y_train_fold,
                cat_features=categorical_features,
            )
            catboost_valid = catboost.Pool(
                X_val_fold,
                y_val_fold,
                cat_features=categorical_features,
            )

            # Initialize and train the CatBoost model
            model = catboost.CatBoostRegressor(**catboost_params)
            model.fit(
                catboost_train,
                eval_set=[catboost_valid],
                early_stopping_rounds=utils.Configuration.early_stopping_round,
                verbose=utils.Configuration.verbose,
                use_best_model=True,
            )

            # Calculate OOF validation predictions
            valid_pred = model.predict(X_val_fold)

            RMSE_score = metrics.mean_squared_error(
                y_val_fold, valid_pred, squared=False
            )

            del (
                X_train_fold,
                y_train_fold,
                X_val_fold,
                y_val_fold,
                catboost_train,
                catboost_valid,
                model,
                valid_pred,
            )
            gc.collect()

            results.append(RMSE_score)
        return np.mean(results)


def train_catboost(
    X: pd.DataFrame, y: pd.Series, catboost_params: dict
) -> catboost.CatBoostRegressor:
    """
    Train a CatBoostRegressor model using the provided data and parameters.

    Parameters:
        X (pd.DataFrame): The feature dataset.
        y (pd.Series): The target variable.
        catboost_params (dict): CatBoost hyperparameters.

    Returns:
        CatBoostRegressor: The trained CatBoost model.

    This function takes the feature dataset `X`, the target variable `y`, and a dictionary of CatBoost
    hyperparameters. It automatically detects categorical features in the dataset and trains a CatBoostRegressor
    model with the specified parameters.

    Example:
        X, y = load_data()
        catboost_params = {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            # ... other hyperparameters ...
        }
        model = train_catboost(X, y, catboost_params)
    """
    categorical_features = X.select_dtypes("object").columns.to_list()

    catboost_train = catboost.Pool(
        X,
        y,
        cat_features=categorical_features,
    )

    model = catboost.CatBoostRegressor(**catboost_params)
    model.fit(
        catboost_train,
        verbose=utils.Configuration.verbose,
    )

    return model
