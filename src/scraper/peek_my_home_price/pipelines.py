# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem
from supabase import create_client, Client
import os
from dotenv import load_dotenv

# load environmental variables
load_dotenv()


class TypeConversionPipeline:
    """converts columns to more appropriate data types"""

    def process_item(self, item, spider):
        adapter = ItemAdapter(item=item)

        int_cols = [
            "id",
            "bedrooms",
            "bathrooms",
            "number_of_frontages",
            "surface_of_the_plot",
            "toilets",
            "zip_code",
            "construction_year",
            "primary_energy_consumption",
            "living_area",
            "price",
        ]

        bool_cols = ["tenement_building", "double_glazing"]

        # Convert str to integer columns
        for col in int_cols:
            value = adapter.get(col)
            if value:  # Ensure non-empty and non-null value
                try:
                    adapter[col] = int(value)
                except ValueError:
                    spider.logger.warning(f"Cannot convert {col} to int: {value}")

        # Convert boolean columns
        for col in bool_cols:
            value = adapter.get(col)
            if value is not None:
                adapter[col] = (
                    value.strip().lower() == "yes"
                )  # "yes" -> True, "no" -> False

        return item


class DropDuplicatesPipeline:
    """drops duplicate items"""

    def __init__(self):
        self.names_seen = set()

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        if adapter["ad_url"] in self.names_seen:
            raise DropItem(f"Duplicate item found: {item!r}")
        else:
            self.names_seen.add(adapter["ad_url"])
            return item


class SupabasePipeline:
    def __init__(self):
        # Initialize Supabase client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        self.supabase: Client = create_client(url, key)

    def process_item(self, item, spider):

        # Convert item to dictionary format for Supabase insertion
        adapter = ItemAdapter(item)
        data = adapter.asdict()

        try:

            # Insert the item data into the Supabase table
            response = self.supabase.table("training_data").insert(data).execute()

        except Exception as e:
            spider.logger.error(f"Failed to insert item into Supabase: {e}")

        return item
