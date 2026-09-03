import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Support running directly or as module
try:
    from src.utils.config import MODEL_PATH, ML_FEATURE_COLS, RISK_LEVEL_THRESHOLDS
except ImportError:
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    from src.utils.config import MODEL_PATH, ML_FEATURE_COLS, RISK_LEVEL_THRESHOLDS


class FloodPredictor:
    """
    Inference engine for Flash Flood Prediction in hilly regions.
    Loads the serialized model artifact and computes flood probability,
    risk category, and top driving risk factors.
    """

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = Path(model_path)
        self.artifact = self._load_artifact()
        self.model = self.artifact["model"]
        self.feature_names = self.artifact.get("feature_names", ML_FEATURE_COLS)
        self.feature_importances = self.artifact.get("feature_importances", {})
        self.metrics = self.artifact.get("metrics", {})
        self.model_name = self.artifact.get("model_name", "Unknown Model")

    def _load_artifact(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. "
                "Please run 'python src/ml/train.py' first."
            )
        return joblib.load(self.model_path)

    def _determine_risk_level(self, probability):
        """Map predicted probability to calibrated categorical risk tier."""
        for level, (low, high) in RISK_LEVEL_THRESHOLDS.items():
            if low <= probability < high:
                return level
        return "SEVERE" if probability >= 0.85 else "LOW"

    def _prepare_input_df(self, input_data):
        """Format and validate incoming data into an aligned DataFrame."""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.Series):
            df = pd.DataFrame([input_data.to_dict()])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}. Expected dict or DataFrame.")

        # If only raw weather metrics provided, compute missing derived features
        if "PRECTOTCORR" in df.columns:
            prec = df["PRECTOTCORR"]
            if "RAIN_3DAY" not in df.columns:
                df["RAIN_3DAY"] = prec * 1.5
            if "RAIN_7DAY" not in df.columns:
                df["RAIN_7DAY"] = df["RAIN_3DAY"] * 1.8
            if "RAIN_14DAY" not in df.columns:
                df["RAIN_14DAY"] = df["RAIN_7DAY"] * 1.5
            if "RAIN_INTENSITY" not in df.columns:
                df["RAIN_INTENSITY"] = np.where(df["RAIN_3DAY"] > 0, prec / df["RAIN_3DAY"], 0.0)
            if "TEMP_HUMIDITY" not in df.columns and "T2M" in df.columns and "RH2M" in df.columns:
                df["TEMP_HUMIDITY"] = df["T2M"] * df["RH2M"]
            if "WIND_RAIN" not in df.columns and "WS2M" in df.columns:
                df["WIND_RAIN"] = df["WS2M"] * prec
            if "PRECTOTCORR_LAG1" not in df.columns:
                df["PRECTOTCORR_LAG1"] = prec * 0.5
            if "PRECTOTCORR_LAG2" not in df.columns:
                df["PRECTOTCORR_LAG2"] = prec * 0.3
            if "API_7DAY" not in df.columns:
                df["API_7DAY"] = df["RAIN_7DAY"] * 0.45

        # Check for missing features and fill with 0.0
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0.0

        return df[self.feature_names]

    def predict_single(self, input_data):
        """
        Inference for a single record.
        Returns:
            dict containing:
                - prediction (0 or 1)
                - probability (0.0 to 1.0)
                - risk_level (LOW, MODERATE, HIGH, SEVERE)
                - top_contributing_factors (list of tuples (factor, value))
        """
        features_df = self._prepare_input_df(input_data)
        prob = float(self.model.predict_proba(features_df)[0, 1])
        pred = int(self.model.predict(features_df)[0])
        risk_level = self._determine_risk_level(prob)

        # Identify top contributing factors using feature values & weights
        sample_vals = features_df.iloc[0].to_dict()
        contributions = []
        for feat, imp in self.feature_importances.items():
            val = sample_vals.get(feat, 0.0)
            score = imp * float(val)
            contributions.append((feat, val, score))

        contributions.sort(key=lambda x: x[2], reverse=True)
        top_factors = [
            {"feature": f, "value": round(v, 2), "importance": round(self.feature_importances.get(f, 0.0), 3)}
            for f, v, _ in contributions[:4]
        ]

        return {
            "prediction": pred,
            "probability": prob,
            "probability_pct": round(prob * 100, 1),
            "risk_level": risk_level,
            "top_contributing_factors": top_factors,
            "model_used": self.model_name,
        }

    def predict_batch(self, df):
        """
        Inference on an entire DataFrame.
        Returns augmented copy of DataFrame with prediction columns.
        """
        features_df = self._prepare_input_df(df)
        probs = self.model.predict_proba(features_df)[:, 1]
        preds = self.model.predict(features_df)

        result_df = df.copy()
        result_df["ML_PREDICTION"] = preds
        result_df["ML_PROBABILITY"] = probs
        result_df["ML_PROBABILITY_PCT"] = np.round(probs * 100, 1)
        result_df["ML_RISK_LEVEL"] = [self._determine_risk_level(p) for p in probs]

        return result_df


if __name__ == "__main__":
    predictor = FloodPredictor()
    print(f"Loaded Model: {predictor.model_name}")
    print(f"Test Set Metrics: {predictor.metrics}")

    print("\n--- Test Case 1: Normal Dry Winter Day ---")
    dry_day = {
        "PRECTOTCORR": 0.0,
        "T2M": 8.5,
        "RH2M": 40.0,
        "WS2M": 1.2,
        "RAIN_3DAY": 0.0,
        "RAIN_7DAY": 0.0,
        "RAIN_14DAY": 0.0,
        "RAIN_INTENSITY": 0.0,
        "TEMP_HUMIDITY": 340.0,
        "WIND_RAIN": 0.0,
        "PRECTOTCORR_LAG1": 0.0,
        "PRECTOTCORR_LAG2": 0.0,
        "API_7DAY": 0.0,
    }
    res_dry = predictor.predict_single(dry_day)
    print(f"Risk: {res_dry['risk_level']} | Probability: {res_dry['probability_pct']}% | Prediction: {res_dry['prediction']}")

    print("\n--- Test Case 2: Extreme Monsoon Cloudburst Event (e.g. July 10, 2023) ---")
    storm_day = {
        "PRECTOTCORR": 126.85,
        "T2M": 21.0,
        "RH2M": 92.5,
        "WS2M": 3.5,
        "RAIN_3DAY": 302.0,
        "RAIN_7DAY": 355.0,
        "RAIN_14DAY": 410.0,
        "RAIN_INTENSITY": 0.42,
        "TEMP_HUMIDITY": 1942.5,
        "WIND_RAIN": 444.0,
        "PRECTOTCORR_LAG1": 68.0,
        "PRECTOTCORR_LAG2": 107.0,
        "API_7DAY": 185.0,
    }
    res_storm = predictor.predict_single(storm_day)
    print(f"Risk: {res_storm['risk_level']} | Probability: {res_storm['probability_pct']}% | Prediction: {res_storm['prediction']}")
    print("Top Contributing Factors:", res_storm["top_contributing_factors"])
