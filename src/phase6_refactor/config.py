from pathlib import Path

RAW_DIR = Path("/home/kaue/Documents/IFSC/IA/predict-energy-behavior-of-prosumers")
OUT_DIR = Path("data/processed")

TRAIN_BLOCKS = (0, 600)
VAL_BLOCKS = (601, 633)
TEST_BLOCKS = (634, 637)

SPLIT_RATIO = (0.7, 0.15, 0.15)

ID_COLUMNS = ["prediction_unit_id"]
TIMESTAMP_COL = "datetime"
TARGET_COL = "target"
FREQ = "h"
CONTEXT_LENGTH = 512
FORECAST_LENGTH = 96

EXOGENOUS_COLS = [
    "temperature", "dewpoint", "rain", "cloudcover_total",
    "windspeed_10m", "shortwave_radiation",
    "temperature_forecast", "cloudcover_total_forecast",
    "price", "eic_count", "installed_capacity",
]

STATIC_COLS = ["county", "is_business", "product_type"]
