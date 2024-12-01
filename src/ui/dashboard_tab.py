from nicegui import ui
from src.database import connect
from src.utils import DatabaseConfiguration
from src.ui import helper


class DashBoardTab:
    def __init__(self):
        self.facts = None
        self.short_term_analytics = None
        self._load_data()

    def _load_data(self):
        db = connect.DataBase()
        self.facts = db.retrieve_data(
            table_name=DatabaseConfiguration.FACTS, most_recent=True
        )
        self.short_term_analytics = db.retrieve_data(
            table_name=DatabaseConfiguration.SHORT_TERM_ANALYTICS, most_recent=True
        )

    def display_tab(self) -> None:
        with ui.column().classes("w-full h-full justify-center items-center"):
            with ui.column().classes(
                "w-full max-w-5xl gap-4 md:flex-row md:gap-8 px-4"
            ):
                self.display_tiles(
                    price=helper.convert_avg_price(self.facts["avg_price"].item()),
                    percentage=helper.calculate_price_change(),
                    province=self.facts["most_active_province"].item(),
                    last_updated=helper.convert_last_updated(
                        self.facts["created_at"].item()
                    ),
                    listing_number=helper.get_most_active_province(
                        self.short_term_analytics["count"]
                    ),
                )
            with ui.column().classes(
                "w-full space-y-8 md:space-y-0 md:flex-row md:gap-8 p-4"
            ):
                with ui.column().classes("w-full md:w-1/3"):
                    self.display_facts(
                        oldest_house=[
                            self.facts["oldest"].item(),
                            self.facts["oldest_zip"].item(),
                            self.facts["oldest_province"].item(),
                        ],
                        most_expensive_house=[
                            self.facts["most_expensive"].item(),
                            self.facts["most_expensive_zip"].item(),
                            self.facts["most_expensive_province"].item(),
                        ],
                        common_energy_rating=[
                            self.facts["most_common_energy"].item(),
                            self.facts["most_common_energy_percent"].item(),
                        ],
                        perc_above=self.facts["energy_b_or_above"].item(),
                    )
                with ui.column().classes("w-full md:w-1/3"):
                    self.display_bar_chart(
                        title="Median prices (€)",
                        provinces=self.short_term_analytics["province"].tolist(),
                        values=self.short_term_analytics["price"].tolist(),
                    )
                with ui.column().classes("w-full md:w-1/3"):
                    self.display_bar_chart(
                        title="Number of ads",
                        provinces=self.short_term_analytics["province"].tolist(),
                        values=self.short_term_analytics["count"].tolist(),
                    )

    def display_tiles(
        self,
        price,
        percentage,
        province,
        last_updated,
        listing_number,
        frequency="Month-to-month",
    ):
        icons = ["home", "trending_up", "signal_cellular_alt"]
        titles = ["Average Price", "Price Trend", "Most Active Region"]
        values = [price, percentage, province]
        descriptions = [last_updated, frequency, listing_number]

        with ui.row().classes(
            "items-start w-full flex flex-col md:flex-row md:gap-8 md:justify-center"
        ).style("align-items: stretch;"):
            for icon, title, description, value in zip(
                icons, titles, descriptions, values
            ):
                with ui.card().classes(
                    "w-full md:w-64 p-4 rounded-lg shadow-md hover:shadow-2xl transition-shadow duration-300 mb-4 md:mb-0 bg-gray-800 flex flex-col"
                ).style("display: flex; flex-grow: 1;"):
                    with ui.row().classes("justify-between items-center w-full"):
                        with ui.row().classes("items-center gap-4"):
                            ui.icon(icon).classes("text-3xl text-blue-400")
                            ui.label(title).classes(
                                "font-semibold text-lg text-gray-200"
                            )
                    with ui.column().classes("items-start flex-grow"):
                        ui.label(value).classes("text-2xl font-bold text-white")
                        ui.label(description).classes("text-sm text-gray-400")

    def display_facts(
        self, oldest_house, most_expensive_house, common_energy_rating, perc_above
    ):
        facts = [
            (
                "history",
                "amber",
                f"The oldest house, built in {oldest_house[0]}, is in {oldest_house[1]}, {oldest_house[2]}.",
            ),
            (
                "euro",
                "green",
                f"The most expensive house, costing €{most_expensive_house[0]}, is in {most_expensive_house[1]}, {most_expensive_house[2]}.",
            ),
            (
                "bolt",
                "yellow",
                f"The most common energy rating is {common_energy_rating[0]}, accounting for {common_energy_rating[1]}% of the ads.",
            ),
            (
                "eco",
                "green",
                f"{perc_above}% of the houses have energy efficiency ratings of B and above.",
            ),
        ]

        with ui.card().classes("p-4 rounded-lg shadow-lg bg-gray-800 w-full mx-auto"):
            ui.label("Dataset Statistics").classes("text-xl font-bold mb-4 text-white")

            with ui.column().classes("space-y-4 w-full"):
                for icon, color, text in facts:
                    with ui.row().classes("items-center gap-3 w-full flex-nowrap"):
                        ui.icon(icon).classes(f"text-{color}-400 flex-shrink-0")
                        ui.label(text).classes("text-base text-gray-300 break-words")

    def display_bar_chart(self, title: str, provinces: list[str], values: list[int]):
        if not values or not provinces or len(values) != len(provinces):
            return

        # Format values with 'k' suffix for values >= 1000
        def format_value(val):
            if val >= 1000:
                return f"{val/1000:.0f}k"
            return str(val)

        series = [
            {
                "type": "bar",
                "data": [
                    {
                        "value": values[i],
                        "label": {"show": True, "formatter": format_value(values[i])},
                    }
                    for i in range(len(values))
                ],
            }
        ]

        with ui.card().classes("p-4 rounded-lg shadow-xl bg-gray-800 w-full mx-auto"):
            ui.echart(
                {
                    "title": {
                        "text": title,
                        "left": "center",
                        "top": "1%",
                        "textStyle": {
                            "fontSize": 18,
                            "fontWeight": "bold",
                            "color": "#ffffff",
                        },
                    },
                    "tooltip": {"trigger": "axis"},
                    "toolbox": {"feature": {"saveAsImage": {}}},
                    "grid": {
                        "left": "1%",
                        "right": "1%",
                        "top": "15%",
                        "bottom": "5%",
                        "containLabel": True,
                    },
                    "xAxis": {
                        "type": "value",
                        "axisLabel": {
                            "show": False,
                        },
                        "position": "top",
                    },
                    "yAxis": {
                        "type": "category",
                        "data": provinces,
                        "axisLabel": {
                            "color": "#ffffff",
                            "fontSize": 12,
                            "align": "right",
                            "padding": [0, 0, 0, 10],
                            "textStyle": {"overflow": "truncate", "ellipsis": "..."},
                        },
                    },
                    "colorBy": "data",
                    "series": series,
                }
            ).style("height: 40vh; width: 100%;")
