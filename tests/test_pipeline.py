import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import (
    RAW_WEATHER_DATA_PATH,
    PROCESSED_FEATURES_PATH,
    LABELED_DATA_PATH,
    MODEL_PATH,
    ML_FEATURE_COLS,
    TARGET_COL,
)
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.data.create_labeled_dataset import FloodDatasetLabeler
from src.ml.predict import FloodPredictor


class TestFlashFloodPipeline(unittest.TestCase):

    def test_01_data_loader(self):
        """Test raw data loader properly skips NASA headers and parses columns."""
        loader = DataLoader(RAW_WEATHER_DATA_PATH)
        df = loader.load_csv()
        self.assertGreater(len(df), 3000)
        expected_cols = ["YEAR", "MO", "DY", "PRECTOTCORR", "T2M", "RH2M", "WS2M"]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_02_feature_engineering(self):
        """Test feature engineering generates required lag and hydrological features."""
        loader = DataLoader(RAW_WEATHER_DATA_PATH)
        df_raw = loader.load_csv()
        engineer = FeatureEngineer(df_raw.head(100))
        df_feat = engineer.create_features()

        for col in ML_FEATURE_COLS:
            self.assertIn(col, df_feat.columns)
            self.assertEqual(df_feat[col].isnull().sum(), 0, f"Column {col} contains NaNs")

        # Check API and Lag logic
        self.assertIn("API_7DAY", df_feat.columns)
        self.assertIn("PRECTOTCORR_LAG1", df_feat.columns)

    def test_03_labeling_logic(self):
        """Test ground truth labeling produces valid binary labels and catches known disaster dates."""
        df_feat = pd.read_csv(PROCESSED_FEATURES_PATH)
        labeler = FloodDatasetLabeler(df_feat)
        df_labeled = labeler.apply_labels()

        self.assertIn(TARGET_COL, df_labeled.columns)
        unique_labels = set(df_labeled[TARGET_COL].unique())
        self.assertTrue(unique_labels.issubset({0, 1}))

        # Peak 2023 disaster date must be labeled as flood (1)
        july_10_mask = (df_labeled["YEAR"] == 2023) & (df_labeled["MO"] == 7) & (df_labeled["DY"] == 10)
        self.assertTrue(july_10_mask.any())
        self.assertEqual(df_labeled.loc[july_10_mask, TARGET_COL].values[0], 1)

    def test_04_model_artifact_and_prediction(self):
        """Test FloodPredictor loads saved model and produces valid predictions and risk tiers."""
        predictor = FloodPredictor(MODEL_PATH)
        self.assertIsNotNone(predictor.model)
        self.assertGreaterEqual(predictor.metrics.get("accuracy", 0), 0.85)

        # 1. Extreme flood storm input
        extreme_weather = {
            "PRECTOTCORR": 115.0,
            "T2M": 20.0,
            "RH2M": 90.0,
            "WS2M": 3.0,
            "RAIN_3DAY": 250.0,
            "RAIN_7DAY": 320.0,
            "RAIN_14DAY": 380.0,
            "RAIN_INTENSITY": 0.46,
            "TEMP_HUMIDITY": 1800.0,
            "WIND_RAIN": 345.0,
            "PRECTOTCORR_LAG1": 60.0,
            "PRECTOTCORR_LAG2": 80.0,
            "API_7DAY": 150.0,
        }
        res_extreme = predictor.predict_single(extreme_weather)
        self.assertIn(res_extreme["risk_level"], ["HIGH", "SEVERE"])
        self.assertEqual(res_extreme["prediction"], 1)
        self.assertGreaterEqual(res_extreme["probability"], 0.60)
        self.assertGreater(len(res_extreme["top_contributing_factors"]), 0)

        # 2. Dry benign weather input
        dry_weather = {
            "PRECTOTCORR": 0.0,
            "T2M": 10.0,
            "RH2M": 35.0,
            "WS2M": 1.0,
            "RAIN_3DAY": 0.0,
            "RAIN_7DAY": 0.0,
            "RAIN_14DAY": 0.0,
            "RAIN_INTENSITY": 0.0,
            "TEMP_HUMIDITY": 350.0,
            "WIND_RAIN": 0.0,
            "PRECTOTCORR_LAG1": 0.0,
            "PRECTOTCORR_LAG2": 0.0,
            "API_7DAY": 0.0,
        }
        res_dry = predictor.predict_single(dry_weather)
        self.assertEqual(res_dry["risk_level"], "LOW")
        self.assertEqual(res_dry["prediction"], 0)
        self.assertLess(res_dry["probability"], 0.30)


if __name__ == "__main__":
    unittest.main()
