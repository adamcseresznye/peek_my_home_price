import pandas as pd
from datetime import datetime
from src.database import connect
from src.utils import DatabaseConfiguration


def convert_avg_price(value: pd.Series) -> str:
    return f"€{value}"


def convert_last_updated(value: pd.Series) -> str:
    dt = datetime.fromisoformat(value)

    formatted_dt = dt.strftime("%B %d, %Y")
    return f"Last updated on {formatted_dt}"


def get_most_active_province(value: pd.Series):
    return f"{value.sum()} listings"


def calculate_price_change():
    db = connect.DataBase()

    long_term_price = db.retrieve_data(
        table_name=DatabaseConfiguration.LONG_TERM_PRICE, most_recent=False
    )

    mean_prices = (
        long_term_price.assign(
            created_at=lambda df: pd.to_datetime(
                df.created_at,
                format="mixed",
                yearfirst=True,
            )
        )
        .groupby("created_at")
        .price.mean()
        .reset_index()
        .sort_values(by="created_at", ascending=False)
    )

    if len(mean_prices) < 2:
        return "Not enough data to calculate price change."

    try:
        new_price = mean_prices.iloc[0].price.item()
        old_price = mean_prices.iloc[1].price.item()
    except IndexError as e:
        return f"Error: {e}"

    percent_change = round(((new_price - old_price) / old_price) * 100, 1)
    return f"{percent_change}%" if percent_change < 0 else f"+{percent_change}%"
