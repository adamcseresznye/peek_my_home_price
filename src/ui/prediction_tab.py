from nicegui import ui
import joblib
from src.utils import ModelConfiguration
import pandas as pd
from fastapi import HTTPException


class PredictionTab:
    def __init__(self):
        self.model = None
        self._load_model()

        self.ui_components = {}

        self.user_inputs = {
            "province": None,
            "zip_code": None,
            "bedrooms": None,
            "bathrooms": None,
            "toilets": None,
            "number_of_frontages": None,
            "surface_of_the_plot": None,
            "living_area": None,
            "tenement_building": None,
            "primary_energy_consumption": None,
            "energy_class": None,
            "double_glazing": None,
            "heating_type": None,
            "construction_year": None,
            "building_condition": None,
        }

    def _load_model(self):
        """
        Load the saved CatBoost regression model.

        Returns:
            catboost.CatBoostRegressor: The loaded CatBoost regression model.
        """

        self.model = joblib.load(ModelConfiguration.MODEL.joinpath("mapie_model.pkl"))

    def display_tab(self):
        """Builds the UI components for the prediction tab."""
        with ui.column().classes("w-full h-full justify-start items-center sm:p-8"):
            with ui.card().classes(
                "w-full lg:max-w-5xl sm:p-6 bg-gray-800 mb-8 text-center"
            ):
                ui.label("Property Price Prediction").classes(
                    "text-xl sm:text-3xl font-bold mb-2 sm:mb-4 text-white"
                )
                ui.label(
                    "Enter property details below to get an estimated market price. "
                    "For the most accurate predictions, provide as many details as possible."
                ).classes(
                    "text-gray-300 max-w-lg sm:max-w-3xl mx-auto text-sm sm:text-base"
                )

            with ui.card().classes("w-full lg:max-w-5xl bg-gray-800"):
                with ui.tabs().classes("w-full bg-gray-700 rounded-t-lg") as tabs:
                    ui.tab("Geography", icon="location_on")
                    ui.tab("Construction", icon="construction")
                    ui.tab("Energy", icon="bolt")

                with ui.tab_panels(tabs, value="Geography").classes(
                    "w-full p-4 sm:p-6"
                ):
                    self.display_geography_panel()
                    self.display_construction_panel()
                    self.display_energy_panel()

                with ui.row().classes(
                    "w-full justify-end gap-2 sm:gap-4 p-2 sm:p-4 bg-gray-700 rounded-b-lg"
                ):
                    ui.button("Reset", icon="refresh").props(
                        'flat color="gray"'
                    ).on_click(self.reset_form)
                    ui.button("Calculate Price", icon="calculate").props(
                        'color="blue"'
                    ).on_click(self.get_prediction)

    def display_geography_panel(self):
        with ui.tab_panel("Geography").classes("w-full"):
            with ui.column().classes("space-y-6 w-full"):
                self.ui_components["province"] = (
                    ui.select(
                        options=[
                            "Liege",
                            "East Flanders",
                            "Brussels",
                            "Antwerp",
                            "Flemish Brabant",
                            "Walloon Brabant",
                            "Hainaut",
                            "Luxembourg",
                            "West Flanders",
                            "Namur",
                            "Limburg",
                        ],
                        value=None,
                        label="Region",
                        on_change=lambda x: self.user_inputs.update(province=x.value),
                    )
                    .classes("w-full")
                    .props('filled dark color="blue"')
                )

                self.ui_components["zip_code"] = (
                    ui.number(
                        label="Postal Code",
                        min=1000,
                        max=9999,
                        value=None,
                        on_change=lambda x: self.user_inputs.update(zip_code=x.value),
                    )
                    .classes("w-full")
                    .props("filled dark")
                )

    def display_construction_panel(self):
        with ui.tab_panel("Construction").classes("w-full"):
            with ui.column().classes("space-y-6 w-full"):
                self.ui_components["bedrooms"] = (
                    ui.number(
                        label="Bedrooms",
                        min=0,
                        on_change=lambda x: self.user_inputs.update(bedrooms=x.value),
                    )
                    .classes("w-full")
                    .props("filled dark")
                )
                self.ui_components["bathrooms"] = (
                    ui.number(
                        label="Bathrooms",
                        min=0,
                        on_change=lambda x: self.user_inputs.update(bathrooms=x.value),
                    )
                    .classes("w-full")
                    .props("filled dark")
                )
                self.ui_components["toilets"] = (
                    ui.number(
                        label="Toilets",
                        min=0,
                        on_change=lambda x: self.user_inputs.update(toilets=x.value),
                    )
                    .classes("w-full")
                    .props("filled dark")
                )
                self.ui_components["number_of_frontages"] = (
                    ui.number(
                        label="Frontages",
                        min=1,
                        on_change=lambda x: self.user_inputs.update(
                            number_of_frontages=x.value
                        ),
                    )
                    .classes("w-full")
                    .props("filled dark")
                )
                self.ui_components["surface_of_the_plot"] = (
                    ui.number(
                        label="Total Land Area (m²)",
                        min=0,
                        format="%.1f",
                        on_change=lambda x: self.user_inputs.update(
                            surface_of_the_plot=x.value
                        ),
                    )
                    .classes("w-full")
                    .props("filled dark")
                )
                self.ui_components["living_area"] = (
                    ui.number(
                        label="Living Area (m²)",
                        min=0,
                        format="%.1f",
                        on_change=lambda x: self.user_inputs.update(
                            living_area=x.value
                        ),
                    )
                    .classes("w-full")
                    .props("filled dark")
                )
                self.ui_components["tenement_building"] = (
                    ui.select(
                        options=["Yes", "No"],
                        label="Tenement Building",
                        value=None,
                        on_change=lambda x: self.user_inputs.update(
                            tenement_building=x.value
                        ),
                    )
                    .classes("w-full")
                    .props('filled dark color="blue"')
                )

    def display_energy_panel(self):
        with ui.tab_panel("Energy").classes("w-full"):
            with ui.column().classes("space-y-6 w-full"):
                self.ui_components["primary_energy_consumption"] = (
                    ui.number(
                        label="Energy Consumption (kWh/m²)",
                        min=0,
                        format="%.1f",
                        on_change=lambda x: self.user_inputs.update(
                            primary_energy_consumption=x.value
                        ),
                    )
                    .classes("w-full")
                    .props("filled dark")
                )
                self.ui_components["energy_class"] = (
                    ui.select(
                        options=[
                            "F",
                            "B",
                            "C",
                            "A",
                            "D",
                            "E",
                            "G",
                            "Not specified",
                            "A+",
                            "A++",
                        ],
                        label="Energy Rating",
                        value=None,
                        on_change=lambda x: self.user_inputs.update(
                            energy_class=x.value
                        ),
                    )
                    .classes("w-full")
                    .props('filled dark color="green"')
                )
                self.ui_components["double_glazing"] = (
                    ui.select(
                        options=["Yes", "No"],
                        label="Double Glazing",
                        value=None,
                        on_change=lambda x: self.user_inputs.update(
                            double_glazing=x.value
                        ),
                    )
                    .classes("w-full")
                    .props('filled dark color="blue"')
                )
                self.ui_components["heating_type"] = (
                    ui.select(
                        options=[
                            "Gas",
                            "Electric",
                            "Fuel oil",
                            "missing",
                            "Solar",
                            "Pellet",
                            "Wood",
                            "Carbon",
                        ],
                        label="Heating Type",
                        value=None,
                        on_change=lambda x: self.user_inputs.update(
                            heating_type=x.value
                        ),
                    )
                    .classes("w-full")
                    .props('filled dark color="orange"')
                )
                self.ui_components["construction_year"] = (
                    ui.number(
                        label="Construction Year",
                        min=1500,
                        max=2024,
                        on_change=lambda x: self.user_inputs.update(
                            construction_year=x.value
                        ),
                    )
                    .classes("w-full")
                    .props("filled dark")
                )
                self.ui_components["building_condition"] = (
                    ui.select(
                        options=[
                            "To be done up",
                            "As new",
                            "Good",
                            "To renovate",
                            "Just renovated",
                            "To restore",
                        ],
                        label="Building Condition",
                        value=None,
                        on_change=lambda x: self.user_inputs.update(
                            building_condition=x.value
                        ),
                    )
                    .classes("w-full")
                    .props('filled dark color="purple"')
                )

    def reset_form(self):
        """Resets the form inputs to their default values."""
        self.user_inputs = {key: None for key in self.user_inputs}

        # Update UI components to reflect reset state
        for key, component in self.ui_components.items():
            if isinstance(component, ui.number) or isinstance(component, ui.select):
                component.value = None
                component.update()

    def get_prediction(self):
        """Handles the 'Calculate Price' button click."""
        try:

            user_input_df = pd.DataFrame(self.user_inputs, index=[0])
            log_y_pred, log_y_pis = self.model.predict(user_input_df, alpha=0.1)
            prediction = 10 ** log_y_pred[0]
            lower_CI = 10 ** log_y_pis.flatten()[0]
            upper_CI = 10 ** log_y_pis.flatten()[1]

            ui.notify(
                message=f"The property’s estimated price is €{prediction:,.0f}, with a 90% probability of ranging from €{lower_CI:,.0f} to €{upper_CI:,.0f}.",
                color="green",
            )
        except TypeError as e:
            ui.notify(
                message=("Please provide more input and try again."),
                color="red",
            )

    def make_api_prediction(self, inputs: dict) -> dict:
        """Handles prediction logic."""
        try:
            input_df = pd.DataFrame(inputs, index=[0])
            log_y_pred, log_y_pis = self.model.predict(input_df, alpha=0.1)
            prediction = 10 ** log_y_pred[0]
            lower_CI = 10 ** log_y_pis.flatten()[0]
            upper_CI = 10 ** log_y_pis.flatten()[1]

            return {
                "estimated_price": prediction,
                "confidence_interval": {
                    "lower_bound": lower_CI,
                    "upper_bound": upper_CI,
                },
            }
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error during prediction: {str(e)}",
            )
