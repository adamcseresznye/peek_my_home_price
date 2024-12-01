import sys
from pathlib import Path

# Add the 'helper' folder to sys.path
sys.path.append(str(Path.cwd() / "helper"))


from pathlib import Path

import catboost
import numpy as np
import pandas as pd
from sklearn import metrics


def predict_catboost(
    model: catboost.CatBoostRegressor,
    X: pd.DataFrame,
    thread_count: int = -1,
    verbose: int = None,
) -> np.ndarray:
    """
    Make predictions using a CatBoost model on the provided dataset.

    Parameters:
        model (catboost.CatBoostRegressor): The trained CatBoost model.
        X (pd.DataFrame): The dataset for which predictions are to be made.
        thread_count (int, optional): The number of threads for prediction. Default is -1 (auto).
        verbose (int, optional): Verbosity level. Default is None.

    Returns:
        np.ndarray: Predicted values.

    This function takes a trained CatBoost model, a dataset `X`, and optional parameters for
    specifying the number of threads (`thread_count`) and verbosity (`verbose`) during prediction.
    It returns an array of predicted values.

    Example:
        model = load_catboost_model()
        X_new = load_new_data()
        predictions = predict_catboost(model, X_new, thread_count=4, verbose=2)
    """
    prediction = model.predict(data=X, thread_count=thread_count, verbose=verbose)
    return prediction


def score_prediction(y_true, y_pred):
    """
    Calculate regression evaluation metrics based on
    true and predicted values.

    Parameters:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values.

    Returns:
        tuple: A tuple containing Root Mean Squared Error (RMSE)
        and R-squared (R2).

    This function calculates RMSE and R2 to evaluate the goodness
    of fit between the true target values and predicted values.

    Example:
        y_true = [3, 5, 7, 9]
        y_pred = [2.8, 5.2, 7.1, 9.3]
        rmse, r2 = score_prediction(y_true, y_pred)
    """
    RMSE = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    R2 = metrics.r2_score(y_true, y_pred)

    return RMSE, R2
