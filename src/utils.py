from pathlib import Path


class DatabaseConfiguration:
    TRAINING_DATA = "training_data"
    MODEL_PERFORMANCE = "model_performance"
    FACTS = "facts"
    SHORT_TERM_ANALYTICS = "short_term_analytics"
    LONG_TERM_PRICE = "long_term_price"


class ModelConfiguration:

    MODEL = Path(__file__).parents[1].joinpath("models")

    TARGET_COLUMN = "price"
    RANDOM_SEED = 3407
    CATBOOST_ITERATIONS = 1000
    CATBOOST_EVAL_FRACTION = 0.2
    CATBOOST_EARLY_STOPPING_ROUNDS = 20
    RANDCV_ITERATIONS = 10
    CROSSVAL_FOLDS = 10
