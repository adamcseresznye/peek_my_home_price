import os
from typing import List

import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# mute the logging coming from other libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)


class DataBase:
    """
    A class to manage connections to a Supabase database and retrieve data
    as a pandas DataFrame for further analysis or model training.
    """

    def __init__(self):
        """
        Initializes the DataBase class, loading environment variables for Supabase
        credentials and establishing a Supabase client connection.
        """
        load_dotenv()
        self.url: str = os.environ.get("SUPABASE_URL")
        self.key: str = os.environ.get("SUPABASE_KEY")
        self.supabase: Client = create_client(self.url, self.key)

    def retrieve_data(
        self, table_name: str, most_recent: bool = False, columns: list[str] = None
    ):
        """
        Retrieves data from the Supabase table as a pandas DataFrame, allowing
        optional filtering for the most recent records and column selection.

        Parameters
        ----------
        table_name : str
            Name of the database to connect to.
        most_recent : bool, optional
            If True, retrieves only records from the most recent date in the
            "created_at" column. If False, retrieves all records (default is False).
        columns : list of str, optional
            A list of specific columns to retrieve. If None, retrieves all
            columns (default is None).

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the queried data from the Supabase table.
            If no records are found (e.g., with an empty table), returns an
            empty DataFrame.

        Examples
        --------
        >>> db = DataBase()
        >>> df_all = db.retrieve_data()
        >>> df_recent = db.retrieve_data(most_recent=True, columns=["price", "zip_code", "created_at"])
        """
        select_columns = "*" if columns is None else ", ".join(columns)

        # Handle most_recent filtering
        if most_recent:
            latest_date_query = (
                self.supabase.table(table_name)
                .select("created_at")
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            if latest_date_query.data:
                most_recent_date = latest_date_query.data[0]["created_at"][:10]
                query = (
                    self.supabase.table(table_name)
                    .select(select_columns)
                    .gte("created_at", f"{most_recent_date}T00:00:00")
                    .lt("created_at", f"{most_recent_date}T23:59:59")
                )
            else:
                return pd.DataFrame()
        else:
            query = self.supabase.table(table_name).select(select_columns)

        # Fetch data with pagination
        all_data = []
        offset = 0
        page_size = 1000  # Supabase row limit per query

        while True:
            paginated_query = query.range(offset, offset + page_size - 1).execute()
            if not paginated_query.data:
                break
            all_data.extend(paginated_query.data)
            offset += page_size

        return pd.DataFrame(all_data)

    def insert_dataframe(
        self, df: pd.DataFrame, table_name: str, delete_existing: bool
    ):
        """
        Inserts a pandas DataFrame into a specified table in the Supabase database.
        Optionally, deletes existing records in the table before inserting new data.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing data to be inserted into the database table.
        table_name : str
            The name of the database table to insert data into.
        connection : DataBase
            A database connection instance that provides access to Supabase.
            This should be an instance of a class with methods for interacting
            with Supabase, including `table()` for table operations and `insert_data()`
            for data insertion.
        delete_existing : bool
            If True, deletes all existing records in the specified table before
            inserting new data. If False, appends the data to the existing records.

        Returns
        -------
        None
            This function doesn't return a value. It prints status messages to indicate
            success or failure of the data insertion operation.

        Raises
        ------
        APIError
            If there is an error with the Supabase API during deletion or insertion.

        Example
        -------
        >>> connection = DataBase()  # Your Supabase database connection
        >>> df = pd.DataFrame({'column1': [1, 2, 3], 'column2': ['a', 'b', 'c']})
        >>> insert_dataframe(df, "my_table", connection, delete_existing=True)
        """
        if delete_existing:
            # Delete all rows in the table by specifying a condition that always matches
            response = (
                self.supabase.table(table_name)
                .delete()
                .neq("id", "00000000-0000-0000-0000-000000000000")
                .execute()
            )

        # Convert DataFrame to a list of dictionaries
        data = df.to_dict(orient="records")

        # Insert all rows at once
        response = self.supabase.table(table_name).insert(data).execute()

        if response is None:
            logging.error(f"Bulk insert failed to {table_name}.")
        else:
            logging.info(f"Data inserted successfully to {table_name}.")

    def delete_data(self, table_name: str, confirm_delete_all: bool = False):
        """
        Deletes all rows in the specified table.

        Parameters
        ----------
        table_name : str
            The name of the table to delete data from.
        confirm_delete_all : bool, optional
            A flag to confirm deletion of all rows, by default False.

        Returns
        -------
        None
        """
        if not confirm_delete_all:
            logging.warning("Deletion aborted: confirm_delete_all flag is False.")
            return

        # Delete all rows by specifying a condition that matches all rows
        response = (
            self.supabase.table(table_name)
            .delete()
            .neq("id", "00000000-0000-0000-0000-000000000000")
            .execute()
        )

        if response is None:
            logging.error(f"Data deletion failed in {table_name}: {response.error}")
        else:
            logging.info(f"Data in {table_name} deleted successfully.")
