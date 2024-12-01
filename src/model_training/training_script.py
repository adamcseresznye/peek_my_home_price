from src.database import connect
from src.utils import ModelConfiguration, DatabaseConfiguration
from src.model_training import data_preprocessing, model
import mapie
import joblib
import logging
import pandas as pd

from sklearn import model_selection

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# mute the logging coming from other libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)


def calculate_facts(df) -> pd.DataFrame:
    """
    Calculate key statistics from a real estate DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'price', 'province', 'construction_year',
        'energy_class', and 'zip_code'.

    Returns
    -------
    pd.DataFrame
        A dataframe containing various calculated statistics, such as
        average price, most active province, oldest property, most expensive
        property, and energy classifications.
    """
    # Required columns check
    required_columns = ["price", "province", "construction_year", "energy_class"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain {required_columns}")

    # Calculate each fact, handling edge cases as appropriate
    avg_price = df.price.mean(skipna=True)
    most_active_province = df.province.value_counts().idxmax()
    oldest = df.sort_values(by="construction_year", ascending=True).iloc[0]
    most_expensive = df.sort_values(by="price", ascending=False).iloc[0]
    most_common_energy = df.energy_class.value_counts(normalize=True).head(1)
    energy_b_or_above = round(
        df.query("energy_class.isin(['B', 'A', 'A+', 'A++'])").shape[0]
        * 100
        / df.shape[0],
        1,
    )

    # Structure results into a dictionary
    result = {
        "avg_price": int(avg_price),
        "most_active_province": most_active_province,
        "oldest": int(oldest.construction_year),
        "oldest_province": oldest.province,
        "oldest_zip": oldest.zip_code,
        "most_expensive": int(most_expensive.price),
        "most_expensive_province": most_expensive.province,
        "most_expensive_zip": most_expensive.zip_code,
        "most_common_energy": most_common_energy.index[0],
        "most_common_energy_percent": most_common_energy.mul(100).round(1).values[0],
        "energy_b_or_above": energy_b_or_above,
    }

    return pd.DataFrame(result, index=[0])


def main():
    logging.info("Connecting to database.")

    try:
        connection = connect.DataBase()

    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        return

    # Retrieve the most recent data from the database
    df = connection.retrieve_data(
        table_name=DatabaseConfiguration.TRAINING_DATA,
        most_recent=True,
    ).drop(columns=["created_at", "id", "ad_url"])

    logging.info(f"Shape of df: {df.shape}.")
    # Split data to X and y, then split further into train and test
    X, y = data_preprocessing.preprocess_and_split_data(df)
    logging.info(f"Shape of X: {X.shape}.")
    logging.info(f"Shape of y: {y.shape}.")

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=ModelConfiguration.RANDOM_SEED
    )
    logging.info(f"Shape of X_train: {X_train.shape}, Shape of X_test: {X_test.shape}.")
    logging.info(f"Shape of y_train: {y_train.shape}, Shape of y_test: {y_test.shape}.")

    # Model training with scikit-learn
    logging.info("Model training with scikit-learn started.")
    try:
        regressor = model.create_tuned_pipeline(X_train, y_train)
    except Exception as e:
        logging.error(f"Sklearn fitting failed: {e}")
        return

    model_evaluation = model.evaluate_model(regressor, X_train, y_train, X_test, y_test)
    logging.info(f"Model performance:{model_evaluation.to_dict()}")
    connection.insert_dataframe(
        table_name=DatabaseConfiguration.MODEL_PERFORMANCE,
        df=model_evaluation,
        delete_existing=False,
    )

    # Mapie model fitting
    logging.info("Mapie regressor fitting started.")
    mapie_model = mapie.regression.MapieRegressor(regressor, method="base", cv=5)
    try:
        mapie_model.fit(X_train, y_train)
    except Exception as e:
        logging.error(f"MapieRegressor fitting failed: {e}")
        return

    # Saving the trained model
    joblib.dump(mapie_model, ModelConfiguration.MODEL.joinpath("mapie_model.pkl"))
    logging.info(
        f"Model saved to: {ModelConfiguration.MODEL.joinpath('mapie_model.pkl')}."
    )

    # Updating databases with the new data
    logging.info("Database update started.")

    # Updating the short_term_analytics table
    # Group by province to calculate median price and ad counts
    median_prices = df.groupby("province")["price"].median()
    ad_counts = df["province"].value_counts()

    short_term_analytics_df = pd.concat(
        [median_prices, ad_counts], axis=1
    ).reset_index()

    connection.insert_dataframe(
        table_name=DatabaseConfiguration.SHORT_TERM_ANALYTICS,
        df=short_term_analytics_df,
        delete_existing=True,
    )

    # Updating the long_term_price table
    connection.insert_dataframe(
        table_name=DatabaseConfiguration.LONG_TERM_PRICE,
        df=median_prices.to_frame().reset_index(),
        delete_existing=False,
    )

    # Updating the facts table
    facts = calculate_facts(df)
    connection.insert_dataframe(
        table_name=DatabaseConfiguration.FACTS,
        df=facts,
        delete_existing=True,
    )

    # Deleting the training_data table
    connection.delete_data(
        table_name=DatabaseConfiguration.TRAINING_DATA, confirm_delete_all=True
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
