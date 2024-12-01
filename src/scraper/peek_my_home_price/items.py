# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html


import scrapy
from scrapy.loader import ItemLoader
from itemloaders.processors import TakeFirst, MapCompose
from w3lib.html import remove_tags
import re
from typing import Optional


def extract_number(value: str) -> Optional[str]:
    # Use regular expression to extract all digits and possibly a decimal point
    match = re.search(r"[\d,.]+", value)
    if match:
        # Remove commas and return the number (you can cast to int or float as needed)
        return match.group().replace(",", "")
    return None


def extract_zip_from_url(value: str) -> Optional[str]:
    # Use regular expression to extract the zip code from a given URL
    return re.search(r"/(\d{4})/", value).group(1)


def convert_zip_to_province(value: str) -> str:
    """
    This function converts a Belgian postal code into the corresponding province
    based on https://www.spotzi.com/en/data-catalog/categories/postal-codes/belgium/.

    Parameters:
    - value (str, optional): The postal code to convert. If None, the function returns None.

    Returns:
    - str: The name of the province corresponding to the postal code.
           If the postal code does not correspond to any province, or if the input is None, the function returns None.
    """
    if value is None:
        return None

    if not len(value) == 4:
        raise ValueError("Invalid postal code")

    first_two_digits = int(value[:2])

    province_dict = {
        range(10, 13): "Brussels",
        range(13, 15): "Walloon Brabant",
        range(15, 20): "Flemish Brabant",
        range(30, 35): "Flemish Brabant",
        range(20, 30): "Antwerp",
        range(35, 40): "Limburg",
        range(40, 50): "Liege",
        range(50, 60): "Namur",
        range(60, 66): "Hainaut",
        range(70, 80): "Hainaut",
        range(66, 70): "Luxembourg",
        range(80, 90): "West Flanders",
        range(90, 100): "East Flanders",
    }

    for key in province_dict:
        if first_two_digits in key:
            return province_dict[key]

    return None


class PeekMyHomePriceItem(scrapy.Item):

    price = scrapy.Field()
    zip_code = scrapy.Field()
    energy_class = scrapy.Field()
    primary_energy_consumption = scrapy.Field()
    bedrooms = scrapy.Field()
    tenement_building = scrapy.Field()
    living_area = scrapy.Field()
    surface_of_the_plot = scrapy.Field()
    bathrooms = scrapy.Field()
    double_glazing = scrapy.Field()
    number_of_frontages = scrapy.Field()
    building_condition = scrapy.Field()
    toilets = scrapy.Field()
    heating_type = scrapy.Field()
    construction_year = scrapy.Field()
    # additional fields
    ad_url = scrapy.Field()
    province = scrapy.Field()


class PeakMyHomePriceLoader(ItemLoader):
    # Apply TakeFirst() as the default output processor to all fields
    default_input_processor = MapCompose(remove_tags, str.strip)
    # Apply remove_tags as the default input processor to all fields
    default_output_processor = TakeFirst()

    # Override the input processor for specific fields
    price_in = MapCompose(remove_tags, str.strip, extract_number)
    primary_energy_consumption_in = MapCompose(remove_tags, str.strip, extract_number)
    living_area_in = MapCompose(remove_tags, str.strip, extract_number)
    surface_of_the_plot_in = MapCompose(remove_tags, str.strip, extract_number)

    # to include the additional fields in the data
    def load_item(self):
        item = super().load_item()

        # Add ad_url
        item["ad_url"] = self.context["response"].url

        # Extract zip code from URL and convert to get province
        item["zip_code"] = extract_zip_from_url(item["ad_url"])
        item["province"] = convert_zip_to_province(item["zip_code"])

        return item
