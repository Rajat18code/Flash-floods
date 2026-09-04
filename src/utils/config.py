import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Filepaths
RAW_WEATHER_DATA_PATH = DATA_RAW_DIR / "himachal_weather.csv"
PROCESSED_FEATURES_PATH = DATA_PROCESSED_DIR / "himachal_features.csv"
LABELED_DATA_PATH = DATA_PROCESSED_DIR / "himachal_flood_labeled.csv"
MODEL_PATH = MODELS_DIR / "flash_flood_model.joblib"

# Model Features
BASE_WEATHER_COLS = ["YEAR", "MO", "DY", "PRECTOTCORR", "T2M", "RH2M", "WS2M"]

ML_FEATURE_COLS = [
    "PRECTOTCORR",         # Daily precipitation (mm/day)
    "T2M",                 # 2m Temperature (°C)
    "RH2M",                # 2m Relative Humidity (%)
    "WS2M",                # 2m Wind Speed (m/s)
    "RAIN_3DAY",           # 3-day cumulative rainfall (mm)
    "RAIN_7DAY",           # 7-day cumulative rainfall (mm)
    "RAIN_14DAY",          # 14-day cumulative rainfall (mm)
    "RAIN_INTENSITY",      # Ratio of daily to 3-day rainfall
    "TEMP_HUMIDITY",       # Temperature-humidity interaction
    "WIND_RAIN",           # Wind-rainfall storm intensity
    "PRECTOTCORR_LAG1",    # 1-day lagged precipitation (mm)
    "PRECTOTCORR_LAG2",    # 2-day lagged precipitation (mm)
    "API_7DAY",            # Antecedent Precipitation Index (proxy for soil moisture)
]
# Target Column Definitions
DOCUMENTED_TARGET_COL = "DOCUMENTED_FLOOD_EVENT"
RULE_BASELINE_COL = "RULE_BASELINE_ALERT"
PROVENANCE_COL = "LABEL_PROVENANCE"
LEGACY_TARGET_COL = "FLASH_FLOOD_OCCURRED"

# Primary ML Target for future training (Documented historical disaster events only)
TARGET_COL = DOCUMENTED_TARGET_COL
# Chronological Train/Test Split (Verified Observational Period: 2015-2024)
TRAIN_START_YEAR = 2015  # 2015-2022 for Training (2,922 days)
TRAIN_END_YEAR = 2022
TEST_START_YEAR = 2023   # 2023-2024 for Testing (731 days, includes catastrophic 2023 & 2024 seasons)
TEST_END_YEAR = 2024

# Geographic & Basin Scope Definition
DATASET_COORDINATES = {"lat": 31.1048, "lon": 77.1734}
DATASET_CATCHMENT = "Sutlej River Basin (Shimla Hills & Upper Catchment)"
DATASET_ELEVATION_M = 2200

# Hydrological and IMD Guideline Thresholds for Mountainous Terrain
IMD_HEAVY_RAIN_MM = 64.5                 # IMD heavy rainfall threshold
IMD_MODERATE_RAIN_MM = 35.0              # IMD moderate-heavy rainfall
ANTECEDENT_3DAY_SATURATION_MM = 75.0     # 3-day catchment saturation threshold
ANTECEDENT_7DAY_SATURATION_MM = 120.0    # 7-day catchment saturation threshold
PROTRACTED_RAIN_MM = 25.0                # Protracted rain under saturated soil

# Confirmed Historical Flood/Cloudburst Disaster Event Dates in Himachal Pradesh (2015-2024)
# Exclusively sourced from published HPSDMA, NDMA Situation Reports, and IMD chronicles (52 Verified Events)
HISTORICAL_DISASTER_DATES = [
    # 2015 (HPSDMA & IMD: March unseasonal snowmelt; July Manikaran cloudburst)
    "2015-03-02", "2015-07-11", "2015-07-12", "2015-07-13",
    # 2016 (HPSDMA: Mandi & Kangra monsoon floods)
    "2016-08-01", "2016-08-06",
    # 2017 (HPSDMA/NDMA: Aug 13 Kotropi landslide disaster - 46 fatalities)
    "2017-07-28", "2017-08-13", "2017-08-21",
    # 2018 (HPSDMA/IMD: Late Sept catastrophic deluge, Beas river floodgate opening)
    "2018-07-03", "2018-08-13", "2018-09-23", "2018-09-24", "2018-09-25",
    # 2019 (HPSDMA: Aug 17-19 Shimla/Rohru/Solan deluge - 22 fatalities)
    "2019-07-15", "2019-08-17", "2019-08-18", "2019-08-19",
    # 2020 (HPSDMA: Mandi & Sirmaur seasonal floods)
    "2020-08-10", "2020-08-20",
    # 2021 (NDMA: July 12 Boh Valley mudslide & Kangra flash floods; July 27 Tozing Nullah)
    "2021-07-12", "2021-07-13", "2021-07-27", "2021-07-28",
    # 2022 (HPSDMA: July 6 Manikaran cloudburst; Aug 19-20 Chakki bridge & Kashan landslide)
    "2022-07-06", "2022-07-13", "2022-07-30", "2022-08-11", "2022-08-19", "2022-08-20", "2022-09-24",
    # 2023 (NDMA/HPSDMA/IMD: Catastrophic season; July 8-11 Beas deluge, Aug 13-15 Summer Hill landslides)
    "2023-06-25", "2023-07-08", "2023-07-09", "2023-07-10", "2023-07-11",
    "2023-07-14", "2023-07-20", "2023-07-22", "2023-07-25",
    "2023-08-13", "2023-08-14", "2023-08-15", "2023-08-22", "2023-08-23", "2023-08-24",
    # 2024 (NDMA/HPSDMA: July 31 Samej Khad hydel project burst - 36+ casualties)
    "2024-07-05", "2024-07-25", "2024-07-31", "2024-08-01", "2024-08-02", "2024-08-11",
]


# Risk Level Classification Thresholds
RISK_LEVEL_THRESHOLDS = {
    "LOW": (0.0, 0.30),
    "MODERATE": (0.30, 0.60),
    "HIGH": (0.60, 0.85),
    "SEVERE": (0.85, 1.01),
}
