# src/ui/api.py
from fastapi import HTTPException, FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from src.ui import prediction_tab

api = FastAPI(
    title="Belgian Housing Market API",
    description="API for predicting Belgian housing prices",
    version="1.0.0",
)


class ConfidenceInterval(BaseModel):
    lower_bound: float
    upper_bound: float


class PredictionResponse(BaseModel):
    estimated_price: float
    confidence_interval: ConfidenceInterval


prediction_instance = prediction_tab.PredictionTab()


@api.get("/predict", response_model=PredictionResponse)
async def predict(
    province: Optional[str] = None,
    zip_code: Optional[int] = None,
    bedrooms: Optional[int] = None,
    bathrooms: Optional[int] = None,
    toilets: Optional[int] = None,
    number_of_frontages: Optional[int] = None,
    surface_of_the_plot: Optional[int] = None,
    living_area: Optional[int] = None,
    tenement_building: Optional[str] = None,
    primary_energy_consumption: Optional[int] = None,
    energy_class: Optional[str] = None,
    double_glazing: Optional[str] = None,
    heating_type: Optional[str] = None,
    construction_year: Optional[int] = None,
    building_condition: Optional[str] = None,
):
    """
    Predict the price of a property based on input parameters.
    """
    user_inputs = {
        "province": province,
        "zip_code": zip_code,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "toilets": toilets,
        "number_of_frontages": number_of_frontages,
        "surface_of_the_plot": surface_of_the_plot,
        "living_area": living_area,
        "tenement_building": tenement_building,
        "primary_energy_consumption": primary_energy_consumption,
        "energy_class": energy_class,
        "double_glazing": double_glazing,
        "heating_type": heating_type,
        "construction_year": construction_year,
        "building_condition": building_condition,
    }

    user_input_df = pd.DataFrame(user_inputs, index=[0])

    try:
        result = prediction_instance.make_api_prediction(user_input_df)
        return PredictionResponse(
            estimated_price=result["estimated_price"],
            confidence_interval=ConfidenceInterval(**result["confidence_interval"]),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
