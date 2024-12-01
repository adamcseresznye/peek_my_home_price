from typing import Tuple

import numpy as np
import pandas as pd

from src.utils import ModelConfiguration


def preprocess_and_split_data(
    df: pd.DataFrame, target_column: str = ModelConfiguration.TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    This function preprocesses the input DataFrame by transforming the target column. It then splits the processed DataFrame
    into features (X) and target (y).
    The user can specify the target column. If it is not provided, the target_column defaults to the value of ModelConfiguration.TARGET_COLUMN

    Parameters:
    - df (pd.DataFrame): The original DataFrame to be preprocessed.

    Returns:
    - Tuple[pd.DataFrame, pd.Series]: A tuple containing the features DataFrame (X) and the target Series (y).
    """
    processed_df = df.assign(
        **{target_column: lambda x: np.log10(x[target_column])}
    ).reset_index(drop=True)
    X, y = processed_df.drop(columns=target_column), processed_df[target_column]
    return X, y
