import sys
from pathlib import Path

# Add the 'helper' folder to sys.path
sys.path.append(str(Path.cwd() / "helper"))

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn import compose, impute, neighbors, pipeline, preprocessing

import utils


def pre_process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses and cleans the input DataFrame for data analysis.

    This function performs various preprocessing steps on the input DataFrame:
    - Renames columns to follow a consistent naming convention.
    - Extracts numeric values from specified columns and converts them to float.
    - Maps boolean values in specified columns to True, False, or None.
    - Performs data cleaning and type conversion for specific columns.

    Args:
        df (pd.DataFrame): The input DataFrame to be preprocessed.

    Returns:
        pd.DataFrame: The preprocessed DataFrame ready for analysis.

    Example:
        To preprocess a DataFrame for analysis:
        >>> data = pd.read_csv("raw_data.csv")
        >>> preprocessed_data = pre_process_dataframe(data)
        >>> print(preprocessed_data.head())

    Notes:
        - The function renames columns, extracts numeric values, and maps boolean values.
        - It also processes additional columns like 'flood_zone_type' and 'connection_to_sewer_network'.
        - Specific columns such as 'cadastral_income' and 'price' undergo special processing.
        - Any errors encountered during processing will be printed with column details.
    """

    def extract_numbers(df: pd.DataFrame, columns: list):
        """
        Extracts numeric values from specified columns in the DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame to extract values from.
            columns (list): List of column names to extract numeric values from.

        Returns:
            pandas.DataFrame: The DataFrame with extracted numeric values.
        """
        for column in columns:
            try:
                df[column] = df[column].str.extract(r"(\d+)").astype("float32")
            except Exception as e:
                print(f"Error processing column {column}: {e}")
        return df

    def map_values(df: pd.DataFrame, columns: list):
        """
        Maps boolean values in specified columns to True, False, or None.

        Args:
            df (pandas.DataFrame): The DataFrame to map values in.
            columns (list): List of column names with boolean values to be mapped.

        Returns:
            pandas.DataFrame: The DataFrame with mapped boolean values.
        """
        for column in columns:
            try:
                df[column] = df[column].map({"Yes": 1, None: np.nan, "No": 0})
            except Exception as e:
                print(f"Error processing column {column}: {e}")
        return df

    number_columns = [
        "construction_year",
        "street_frontage_width",
        "number_of_frontages",
        "covered_parking_spaces",
        "outdoor_parking_spaces",
        "living_area",
        "living_room_surface",
        "kitchen_surface",
        "bedrooms",
        "bedroom_1_surface",
        "bedroom_2_surface",
        "bedroom_3_surface",
        "bathrooms",
        "toilets",
        "surface_of_the_plot",
        "width_of_the_lot_on_the_street",
        "garden_surface",
        "primary_energy_consumption",
        "co2_emission",
        "yearly_theoretical_total_energy_consumption",
    ]

    boolean_columns = [
        "basement",
        "furnished",
        "gas_water__electricity",
        "double_glazing",
        "planning_permission_obtained",
        "tv_cable",
        "dining_room",
        "proceedings_for_breach_of_planning_regulations",
        "subdivision_permit",
        "tenement_building",
        "possible_priority_purchase_right",
        "office",
    ]

    return (
        df.sort_index(axis=1)
        .fillna(np.nan)
        .rename(
            columns=lambda column: column.lower()
            .replace(" ", "_")
            .replace("&", "")
            .replace(",", "")
        )
        .rename(columns={"co₂_emission": "co2_emission"})
        .pipe(lambda df: extract_numbers(df, number_columns))
        .pipe(lambda df: map_values(df, boolean_columns))
        .assign(
            flood_zone_type=lambda df: df.flood_zone_type.map(
                {
                    "Non flood zone": 0,
                    "No": 0,
                    "Possible flood zone": 1,
                }
            ),
            connection_to_sewer_network=lambda df: df.connection_to_sewer_network.map(
                {
                    "Connected": 1,
                    "Not connected": 0,
                }
            ),
            as_built_plan=lambda df: df.as_built_plan.map(
                {
                    "Yes, conform": 1,
                    "No": 0,
                }
            ),
            cadastral_income=lambda df: df.cadastral_income.str.split(" ", expand=True)[
                3
            ].astype("float32"),
            price=lambda df: df.price.str.rsplit(" ", expand=True, n=2)[1].astype(
                float
            ),
        )
    )


def identify_outliers(df: pd.DataFrame) -> pd.Series:
    """
    Identify outliers in a DataFrame.

    This function uses a Local Outlier Factor (LOF) algorithm to identify outliers in a given
    DataFrame. It operates on both numerical and categorical features, and it returns a binary
    Series where `True` represents an outlier and `False` represents a non-outlier.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing features for outlier identification.

    Returns:
    - pd.Series: A Boolean Series indicating outliers (True) and non-outliers (False).

    Example:
    ```python
    # Load your DataFrame with features (df)
    df = load_data()

    # Identify outliers using the function
    outlier_mask = identify_outliers(df)

    # Use the outlier mask to filter your DataFrame
    filtered_df = df[~outlier_mask]  # Keep non-outliers
    ```

    Notes:
    - The function uses Local Outlier Factor (LOF) with default parameters for identifying outliers.
    - Numerical features are imputed using median values, and categorical features are one-hot encoded
    and imputed with median values.
    - The resulting Boolean Series is `True` for outliers and `False` for non-outliers.
    """

    # Extract numerical and categorical feature names
    NUMERICAL_FEATURES = df.select_dtypes("number").columns.tolist()
    CATEGORICAL_FEATURES = df.select_dtypes("object").columns.tolist()

    # Define transformers for preprocessing
    numeric_transformer = pipeline.Pipeline(
        steps=[("imputer", impute.SimpleImputer(strategy="median"))]
    )

    categorical_transformer = pipeline.Pipeline(
        steps=[
            ("encoder", preprocessing.OneHotEncoder(handle_unknown="ignore")),
            ("imputer", impute.SimpleImputer(strategy="median")),
        ]
    )

    # Create a ColumnTransformer to handle both numerical and categorical features
    preprocessor = compose.ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    # Initialize the LOF model
    clf = neighbors.LocalOutlierFactor()

    # Fit LOF to preprocessed data and make predictions
    y_pred = clf.fit_predict(preprocessor.fit_transform(df))

    # Adjust LOF predictions to create a binary outlier mask
    y_pred_adjusted = [1 if x == -1 else 0 for x in y_pred]
    outlier_mask = pd.Series(y_pred_adjusted) == 0

    return outlier_mask


def prepare_data_for_modelling(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for machine learning modeling.

    This function takes a DataFrame and prepares it for machine learning by performing the following steps:
    1. Randomly shuffles the rows of the DataFrame.
    2. Converts the 'price' column to the base 10 logarithm.
    3. Fills missing values in categorical variables with 'missing value'.
    4. Separates the features (X) and the target (y).
    5. Identifies and filters out outlier values based on LocalOutlierFactor.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the dataset.

    Returns:
    - Tuple[pd.DataFrame, pd.Series]: A tuple containing the prepared features (X) and the target (y).

    Example use case:
    ```python
    # Load your dataset into a DataFrame (e.g., df)
    df = load_data()

    # Prepare the data for modeling
    X, y = prepare_data_for_modelling(df)

    # Now you can use X and y for machine learning tasks.
    ```

    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the prepared features (X) and the target (y).
    """

    processed_df = (
        df.sample(frac=1, random_state=utils.Configuration.seed)
        .reset_index(drop=True)
        .assign(
            price=lambda df: np.log10(df.price),
            city_group=lambda df: df.groupby("city")["cadastral_income"].transform(
                "median"
            ),
            building_condition_group=lambda df: df.groupby("building_condition")[
                "yearly_theoretical_total_energy_consumption"
            ].transform("median"),
            energy_efficiency_1=lambda df: df.yearly_theoretical_total_energy_consumption
            / df.primary_energy_consumption,
            energy_efficiency_2=lambda df: df.primary_energy_consumption
            / df.living_area,
            bargain_1=lambda df: df.cadastral_income / df.bedrooms,
            bargain_2=lambda df: df.cadastral_income / df.living_area,
        )
    )

    # Fill missing categorical variables with "missing value"
    for col in processed_df.columns:
        if processed_df[col].dtype.name in ("bool", "object", "category"):
            processed_df[col] = processed_df[col].fillna("missing value")

    # Separate features (X) and target (y)
    X = processed_df.loc[:, utils.Configuration.features_to_keep_v2]
    y = processed_df[utils.Configuration.target_col]

    outlier_mask = identify_outliers(X)

    X_wo_outliers = X.loc[outlier_mask, :].reset_index(drop=True)
    y_wo_outliers = y.loc[outlier_mask].reset_index(drop=True)

    print(f"Shape of X and y with outliers: {X.shape}, {y.shape}")
    print(
        f"Shape of X and y without outliers: {X_wo_outliers.shape}, {y_wo_outliers.shape}"
    )

    return X_wo_outliers, y_wo_outliers
