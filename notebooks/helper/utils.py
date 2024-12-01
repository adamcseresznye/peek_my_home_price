import os
import random

import numpy as np


class Configuration:

    target_col = "price"

    features_to_keep_v1 = [
        "bedrooms",
        "state",
        "kitchen_type",
        "number_of_frontages",
        "toilets",
        "street",
        "lng",
        "primary_energy_consumption",
        "bathrooms",
        "yearly_theoretical_total_energy_consumption",
        "surface_of_the_plot",
        "building_condition",
        "city",
        "lat",
        "cadastral_income",
        "living_area",
    ]
    features_to_keep_v2 = [
        "bedrooms",
        "state",
        "number_of_frontages",
        "street",
        "lng",
        "primary_energy_consumption",
        "bathrooms",
        "yearly_theoretical_total_energy_consumption",
        "surface_of_the_plot",
        "building_condition",
        "city",
        "lat",
        "cadastral_income",
        "living_area",
    ]
    seed = 3407
    n_folds = 10
    verbose = 0
    early_stopping_round = 20

    catboost_params = {
        #'iterations': 342,
        #'depth': 3,
        #'learning_rate': 0.3779980855781628,
        #'random_strength': 1.5478223057973914,
        #'bagging_temperature': 0.689173368569372,
        #'l2_leaf_reg': 16,
        #'border_count': 37,
        "thread_count": os.cpu_count(),
        "loss_function": "RMSE",
        "iterations": 100,
        "learning_rate": 0.2,
    }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
