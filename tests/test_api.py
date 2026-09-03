import unittest
from pathlib import Path
import sys

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.api.main import (
    health_check,
    model_info,
    predict_flood_risk,
    predict_live_district,
    WeatherInput,
)


class TestFastAPIEndpoints(unittest.TestCase):

    def test_01_health_check(self):
        data = health_check()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])

    def test_02_model_info(self):
        data = model_info()
        self.assertIn("metrics", data)
        self.assertIn("accuracy", data["metrics"])
        self.assertIn("training_features", data)

    def test_03_predict_post(self):
        payload = WeatherInput(
            PRECTOTCORR=85.0,
            T2M=19.5,
            RH2M=89.0,
            WS2M=3.2,
            RAIN_3DAY=160.0,
            RAIN_7DAY=210.0,
            district="Kullu (Beas Basin)"
        )
        data = predict_flood_risk(payload)
        self.assertIn(data.risk_level, ["HIGH", "SEVERE"])
        self.assertIn(data.alert_code, ["ORANGE", "RED"])
        self.assertGreaterEqual(len(data.top_contributing_factors), 1)

    def test_04_predict_live(self):
        data = predict_live_district("Shimla")
        self.assertIn("Shimla", data.district)
        self.assertIsNotNone(data.risk_level)


if __name__ == "__main__":
    unittest.main()

