from nicegui import ui
import pandas as pd
from src.database import connect
from src.utils import DatabaseConfiguration


class TrendsTab:
    def __init__(self):
        self.long_term_price = None
        self._load_data()

    def _load_data(self):
        db = connect.DataBase()
        self.long_term_price = db.retrieve_data(
            table_name=DatabaseConfiguration.LONG_TERM_PRICE, most_recent=False
        )

    def display_tab(self) -> None:
        with ui.column().classes("w-full h-screen justify-start items-center gap-2"):
            dates = (
                pd.to_datetime(
                    self.long_term_price.created_at, format="mixed", yearfirst=True
                )
                .dt.strftime("%B %d, %Y")
                .unique()
                .tolist()
            )
            series = [
                {
                    "name": province,
                    "type": "line",
                    "data": self.long_term_price.query("province == @province")["price"]
                    .astype(float)
                    .tolist(),
                    "lineStyle": {"width": 3},
                    "symbol": "circle",
                    "symbolSize": 8,
                    "areaStyle": {
                        "color": {
                            "type": "linear",
                            "x": 0,
                            "y": 0,
                            "x2": 0,
                            "y2": 1,
                            "colorStops": [
                                {"offset": 0, "color": "rgba(59, 130, 246, 0.5)"},
                                {"offset": 1, "color": "rgba(59, 130, 246, 0.1)"},
                            ],
                        }
                    },
                }
                for province in self.long_term_price.province.unique()
            ]

            with ui.card().classes(
                "w-full h-[80vh] max-w-7xl bg-gray-800 rounded-xl shadow-2xl"
            ):

                ui.echart(
                    {
                        "title": {
                            "text": "Median Price (€) Over Time",
                            "left": "center",
                            "top": "5px",
                            "textStyle": {
                                "fontSize": 20,
                                "fontWeight": "bold",
                                "color": "#ffffff",
                            },
                        },
                        "tooltip": {
                            "trigger": "axis",
                            "backgroundColor": "rgba(31, 41, 55, 0.9)",
                            "borderColor": "#475569",
                            "textStyle": {"color": "#ffffff"},
                            "axisPointer": {
                                "type": "line",
                                "lineStyle": {"color": "#475569", "width": 1},
                            },
                        },
                        "legend": {
                            "top": "30px",
                            "textStyle": {"color": "#ffffff"},
                            "icon": "roundRect",
                            "itemGap": 12,
                            "itemWidth": 12,
                            "itemHeight": 8,
                            "width": "90%",
                            "type": "scroll",
                            "orient": "horizontal",
                            "pageButtonPosition": "end",
                            "pageTextStyle": {"color": "#ffffff"},
                        },
                        "grid": {
                            "left": "3%",
                            "right": "4%",
                            "bottom": "8%",
                            "containLabel": True,
                            "top": "80px",
                        },
                        "toolbox": {
                            "feature": {
                                "saveAsImage": {"title": "Save as Image"},
                                "restore": {"title": "Reset"},
                            },
                            "iconStyle": {"borderColor": "#ffffff"},
                            "itemSize": 10,
                            "right": "5px",
                            "top": "5px",
                        },
                        "dataZoom": [
                            {"type": "inside", "start": 0, "end": 100},
                            {
                                "type": "slider",
                                "height": 16,
                                "bottom": 10,
                                "borderColor": "#475569",
                                "textStyle": {"color": "#ffffff"},
                                "start": 0,
                                "end": 100,
                            },
                        ],
                        "xAxis": {
                            "type": "category",
                            "boundaryGap": False,
                            "data": dates,
                            "axisLine": {"lineStyle": {"color": "#475569"}},
                            "axisLabel": {
                                "color": "#ffffff",
                                "fontSize": 10,
                            },
                        },
                        "yAxis": {
                            "type": "value",
                            "axisLine": {"lineStyle": {"color": "#475569"}},
                            "splitLine": {
                                "lineStyle": {"color": "#374151", "type": "dashed"}
                            },
                            "axisLabel": {
                                "color": "#ffffff",
                                "fontSize": 10,
                                "formatter": "{value}",
                            },
                        },
                        "series": series,
                        "media": [
                            {
                                "query": {"maxWidth": 575.98},
                                "option": {
                                    "legend": {
                                        "itemWidth": 8,
                                        "itemHeight": 6,
                                        "itemGap": 6,
                                        "padding": [3, 3],
                                    },
                                    "grid": {
                                        "top": "80px",
                                    },
                                },
                            },
                        ],
                    }
                ).classes("flex-grow")
