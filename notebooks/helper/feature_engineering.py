import sys
from pathlib import Path

from tqdm import tqdm

# Add the 'helper' folder to sys.path
sys.path.append(str(Path.cwd() / "helper"))

from pathlib import Path


import numpy as np
import pandas as pd

from sklearn import (
    impute,
    pipeline,
    preprocessing,
)
import train_model


def FE_categorical_transform(
    X: pd.DataFrame, y: pd.Series, transform_type: str = "mean"
) -> pd.DataFrame:
    """
    Feature Engineering: Transform categorical features using CatBoost Cross-Validation.

    This function performs feature engineering by transforming categorical features using CatBoost
    Cross-Validation. It calculates the mean and standard deviation of Out-Of-Fold (OOF) Root Mean
    Squared Error (RMSE) scores for various combinations of categorical and numerical features.

    Parameters:
    - X (pd.DataFrame): The input DataFrame containing both categorical and numerical features.
    - transform_type (str, optional): The transformation type, such as "mean" or other valid
      CatBoost transformations. Defaults to "mean".

    Returns:
    - pd.DataFrame: A DataFrame with columns "mean_OOFs," "std_OOFs," "categorical," and "numerical,"
      sorted by "mean_OOFs" in ascending order.

    Example:
    ```python
    # Load your DataFrame with features (X)
    X = load_data()

    # Perform feature engineering
    result_df = FE_categorical_transform(X, transform_type="mean")

    # View the DataFrame with sorted results
    print(result_df.head())
    ```

    Notes:
    - This function uses CatBoost Cross-Validation to assess the quality of transformations for
      various combinations of categorical and numerical features.
    - The resulting DataFrame provides insights into the effectiveness of different transformations.
    - Feature engineering can help improve the performance of machine learning models.
    """
    # Initialize a list to store results
    results = []

    # Get a list of categorical and numerical columns
    categorical_columns = X.select_dtypes("object").columns
    numerical_columns = X.select_dtypes("number").columns

    # Combine the loops to have a single progress bar
    for categorical in tqdm(categorical_columns, desc="Progress"):
        for numerical in tqdm(numerical_columns):
            # Create a deep copy of the input data
            temp = X.copy(deep=True)

            # Calculate the transformation for each group within the categorical column
            temp["new_column"] = temp.groupby(categorical)[numerical].transform(
                transform_type
            )

            # Run CatBoost Cross-Validation with the transformed data
            mean_OOF, std_OOF = train_model.run_catboost_CV(temp, y)

            # Store the results as a tuple
            result = (mean_OOF, std_OOF, categorical, numerical)
            results.append(result)

            del temp, mean_OOF, std_OOF

    # Create a DataFrame from the results and sort it by mean OOF scores
    result_df = pd.DataFrame(
        results, columns=["mean_OOFs", "std_OOFs", "categorical", "numerical"]
    )
    result_df = result_df.sort_values(by="mean_OOFs")
    return result_df


def FE_continuous_transform(
    X: pd.DataFrame, y: pd.Series, transform_type: str = "mean"
) -> pd.DataFrame:
    """
    Feature Engineering: Transform continuous features using CatBoost Cross-Validation.

    This function performs feature engineering by transforming continuous features using CatBoost
    Cross-Validation. It calculates the mean and standard deviation of Out-Of-Fold (OOF) Root Mean
    Squared Error (RMSE) scores for various combinations of discretized and transformed continuous
    features.

    Parameters:
    - X (pd.DataFrame): The input DataFrame containing both continuous and categorical features.
    - y (pd.Series): The target variable for prediction.
    - transform_type (str, optional): The transformation type, such as "mean" or other valid
      CatBoost transformations. Defaults to "mean".

    Returns:
    - pd.DataFrame: A DataFrame with columns "mean_OOFs," "std_OOFs," "discretized_continuous,"
      and "transformed_continuous," sorted by "mean_OOFs" in ascending order.

    Example:
    ```python
    # Load your DataFrame with features (X) and target variable (y)
    X, y = load_data()

    # Perform feature engineering
    result_df = FE_continuous_transform(X, y, transform_type="mean")

    # View the DataFrame with sorted results
    print(result_df.head())
    ```

    Notes:
    - This function uses CatBoost Cross-Validation to assess the quality of transformations for
      various combinations of discretized and transformed continuous features.
    - The number of bins for discretization is determined using Sturges' rule.
    - The resulting DataFrame provides insights into the effectiveness of different transformations.
    - Feature engineering can help improve the performance of machine learning models.
    """
    # Initialize a list to store results
    results = []

    # Get a list of continuous and numerical columns
    continuous_columns = X.select_dtypes("number").columns
    optimal_bins = int(np.floor(np.log2(X.shape[0])) + 1)

    # Combine the loops to have a single progress bar
    for discretized_continuous in tqdm(continuous_columns, desc="Progress:"):
        for transformed_continuous in tqdm(continuous_columns):
            if discretized_continuous != transformed_continuous:
                # Create a deep copy of the input data
                temp = X.copy(deep=True)

                discretizer = pipeline.Pipeline(
                    steps=[
                        ("imputer", impute.SimpleImputer(strategy="median")),
                        (
                            "add_bins",
                            preprocessing.KBinsDiscretizer(
                                encode="ordinal", n_bins=optimal_bins
                            ),
                        ),
                    ]
                )

                temp[discretized_continuous] = discretizer.fit_transform(
                    X[[discretized_continuous]]
                )

                # Calculate the transformation for each group within the categorical column
                temp["new_column"] = temp.groupby(discretized_continuous)[
                    transformed_continuous
                ].transform(transform_type)

                # Run CatBoost Cross-Validation with the transformed data
                mean_OOF, std_OOF = train_model.run_catboost_CV(temp, y)

                # Store the results as a tuple
                result = (
                    mean_OOF,
                    std_OOF,
                    discretized_continuous,
                    transformed_continuous,
                )
                results.append(result)

                del temp, mean_OOF, std_OOF

    # Create a DataFrame from the results and sort it by mean OOF scores
    result_df = pd.DataFrame(
        results,
        columns=[
            "mean_OOFs",
            "std_OOFs",
            "discretized_continuous",
            "transformed_continuous",
        ],
    )
    result_df = result_df.sort_values(by="mean_OOFs")
    return result_df
