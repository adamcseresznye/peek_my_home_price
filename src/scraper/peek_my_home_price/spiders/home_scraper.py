import scrapy
from scrapy_playwright.page import PageMethod
from peek_my_home_price.items import PeekMyHomePriceItem, PeakMyHomePriceLoader
from dotenv import load_dotenv
import os

# load environmental variables
load_dotenv()


def should_abort_request(request):
    if request.resource_type in ["image", "stylesheet", "font", "media"]:
        return True
    if ".jpg" in request.url or ".png" in request.url:
        return True
    if request.method.lower() == "post":
        return True
    return False


class HomeScraper(scrapy.Spider):
    name = "home_scraper"
    allowed_domains = [os.getenv("ALLOWED_DOMAINS")]

    def __init__(self, *args, **kwargs):
        super(HomeScraper, self).__init__(*args, **kwargs)

        self.current_page = 0  # Define starting page here
        self.max_pages = 333  # Define last page here
        # Build base URL template
        self.url_template = os.getenv("URL_TEMPLATE")
        self.next_page_url_template = os.getenv("NEXT_PAGE_URL")
        # generate start_urls from template
        self.start_urls = [self.url_template.format(self.current_page)]

    def start_requests(self):
        yield scrapy.Request(
            self.start_urls[0],
            meta={
                "playwright": True,
                "playwright_include_page": True,
                "errback": self.errback,
                "playwright_page_methods": [
                    PageMethod(
                        "wait_for_selector", "li.pagination__item", timeout=10000
                    ),
                    PageMethod(
                        "wait_for_selector", "a.card__title-link", timeout=10000
                    ),
                ],
            },
        )

    async def parse(self, response):
        # Log current page
        self.logger.info(f"Parsing page {self.current_page}")
        for link in response.css("a.card__title-link::attr(href)"):
            # Check if any of the keywords are in the link if so follow them
            if any(
                keyword in link.get()
                for keyword in [
                    "apartment",
                    "duplex",
                    "house",
                    "villa",
                    "exceptional-property",
                    "penthouse",
                ]
            ):
                yield response.follow(link.get(), callback=self.parse_item)

        # Check if we should continue to next page
        if self.current_page < self.max_pages:
            self.current_page += 1
            next_page_url = self.next_page_url_template.format(page=self.current_page)

            self.logger.info(f"Moving to next page: {next_page_url}")

            yield response.follow(next_page_url, callback=self.parse)
        else:
            self.logger.info("Reached maximum page limit")

    async def parse_item(self, response):
        # Use the custom PeakMyHomePriceLoader here instead of the standard ItemLoader
        l = PeakMyHomePriceLoader(
            item=PeekMyHomePriceItem(), selector=response, response=response
        )

        l.add_css("price", "p.classified__price span")
        l.add_css("zip_code", "span.classified__information--address-row")
        l.add_css("energy_class", "th:contains('Energy class') + td")
        l.add_css(
            "primary_energy_consumption",
            "th:contains('Primary energy consumption') + td",
        )
        l.add_css("bedrooms", "th:contains('Bedrooms') + td")
        l.add_css("tenement_building", "th:contains('Tenement building') + td")
        l.add_css("living_area", "th:contains('Living area') + td")
        l.add_css("surface_of_the_plot", "th:contains('Surface of the plot') + td")
        l.add_css("bathrooms", "th:contains('Bathrooms') + td")
        l.add_css("double_glazing", "th:contains('Double glazing') + td")
        l.add_css("number_of_frontages", "th:contains('Number of frontages') + td")
        l.add_css("building_condition", "th:contains('Building condition') + td")
        l.add_css("toilets", "th:contains('Toilets') + td")
        l.add_css("heating_type", "th:contains('Heating type') + td")
        l.add_css("construction_year", "th:contains('Construction year') + td")

        yield l.load_item()

    async def errback(self, failure):
        page = failure.request.meta["playwright_page"]
        await page.close()
