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
    DOCUMENTED_TARGET_COL,
    RULE_BASELINE_COL,
    PROVENANCE_COL,
    HISTORICAL_DISASTER_DATES,
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

    def test_03_labeling_logic_and_separation(self):
        """
        Verify scientific integrity of the flood labeling pipeline:
        1. DOCUMENTED_FLOOD_EVENT is binary.
        2. Documented event dates are correctly labeled.
        3. RULE_BASELINE_ALERT is binary.
        4. Rule-based alerts do not automatically become documented flood events.
        5. The two concepts remain strictly separate.
        """
        df_feat = pd.read_csv(PROCESSED_FEATURES_PATH)
        labeler = FloodDatasetLabeler(df_feat)
        df_labeled = labeler.apply_labels()

        # 1. DOCUMENTED_FLOOD_EVENT is binary
        self.assertIn(DOCUMENTED_TARGET_COL, df_labeled.columns)
        unique_doc = set(df_labeled[DOCUMENTED_TARGET_COL].unique())
        self.assertTrue(unique_doc.issubset({0, 1}))

        # 2. Documented event dates are correctly labeled
        df_labeled["DATE_STR"] = pd.to_datetime(df_labeled["DATE"]).dt.strftime("%Y-%m-%d")
        for date_str in HISTORICAL_DISASTER_DATES:
            row = df_labeled[df_labeled["DATE_STR"] == date_str]
            self.assertFalse(row.empty, f"Disaster date {date_str} missing from dataset")
            self.assertEqual(
                row[DOCUMENTED_TARGET_COL].values[0], 1,
                f"Documented disaster date {date_str} was not labeled 1 in {DOCUMENTED_TARGET_COL}"
            )

        # 3. RULE_BASELINE_ALERT is binary
        self.assertIn(RULE_BASELINE_COL, df_labeled.columns)
        unique_rule = set(df_labeled[RULE_BASELINE_COL].unique())
        self.assertTrue(unique_rule.issubset({0, 1}))

        # 4. Rule-based alerts do not automatically become documented flood events
        rule_only_mask = (df_labeled[RULE_BASELINE_COL] == 1) & (df_labeled[DOCUMENTED_TARGET_COL] == 0)
        self.assertGreater(rule_only_mask.sum(), 0, "Expected days where rule alert fired without documented disaster")
        
        # Verify that for all rule_only days, DOCUMENTED_FLOOD_EVENT remains 0
        rule_only_rows = df_labeled[rule_only_mask]
        self.assertTrue((rule_only_rows[DOCUMENTED_TARGET_COL] == 0).all())
        self.assertTrue((rule_only_rows[PROVENANCE_COL] == "rule_baseline_only").all())

        # 5. Provenance tags partition the dataset cleanly
        valid_provenances = {"documented_and_rule", "documented_event", "rule_baseline_only", "none"}
        self.assertTrue(set(df_labeled[PROVENANCE_COL].unique()).issubset(valid_provenances))

        # Ensure exact count separation
        doc_count = (df_labeled[DOCUMENTED_TARGET_COL] == 1).sum()
        self.assertEqual(doc_count, len(HISTORICAL_DISASTER_DATES))

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
