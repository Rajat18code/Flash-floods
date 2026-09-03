import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Support running directly or as module
try:
    from src.utils.config import RAW_WEATHER_DATA_PATH, PROCESSED_FEATURES_PATH
except ImportError:
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    from src.utils.config import RAW_WEATHER_DATA_PATH, PROCESSED_FEATURES_PATH


class FeatureEngineer:
    """
    Creates hydrometeorological features for flash flood prediction in hilly terrain.
    """

    def __init__(self, df):
        self.df = df.copy()

    def create_features(self):
        """
        Create rainfall, antecedent moisture, lag, and atmospheric interaction features.
        """
        # Sort data chronologically if date columns are present
        if all(col in self.df.columns for col in ["YEAR", "MO", "DY"]):
            self.df = self.df.sort_values(
                by=["YEAR", "MO", "DY"]
            ).reset_index(drop=True)

            # Create proper date column
            self.df["DATE"] = pd.to_datetime(
                self.df[["YEAR", "MO", "DY"]].rename(
                    columns={"YEAR": "year", "MO": "month", "DY": "day"}
                )
            )

        # 1. Cumulative Rainfall Features (mm)
        self.df["RAIN_3DAY"] = (
            self.df["PRECTOTCORR"]
            .rolling(window=3, min_periods=1)
            .sum()
        )

        self.df["RAIN_7DAY"] = (
            self.df["PRECTOTCORR"]
            .rolling(window=7, min_periods=1)
            .sum()
        )

        self.df["RAIN_14DAY"] = (
            self.df["PRECTOTCORR"]
            .rolling(window=14, min_periods=1)
            .sum()
        )

        # 2. Rainfall Intensity Ratio (Current precipitation share of 3-day sum)
        self.df["RAIN_INTENSITY"] = np.where(
            self.df["RAIN_3DAY"] > 0,
            self.df["PRECTOTCORR"] / self.df["RAIN_3DAY"],
            0.0
        )

        # 3. Atmospheric interaction features
        self.df["TEMP_HUMIDITY"] = self.df["T2M"] * self.df["RH2M"]
        self.df["WIND_RAIN"] = self.df["WS2M"] * self.df["PRECTOTCORR"]

        # 4. Lagged Precipitation Features (mm)
        self.df["PRECTOTCORR_LAG1"] = (
            self.df["PRECTOTCORR"].shift(1).fillna(0.0)
        )
        self.df["PRECTOTCORR_LAG2"] = (
            self.df["PRECTOTCORR"].shift(2).fillna(0.0)
        )

        # 5. Antecedent Precipitation Index (API) - 7 Day Decay
        # Standard hydrological proxy for soil moisture saturation prior to current day:
        # API_t = sum_{i=1}^{7} (P_{t-i} * k^i), k = 0.85
        decay_factor = 0.85
        api = pd.Series(0.0, index=self.df.index)
        for i in range(1, 8):
            lagged_p = self.df["PRECTOTCORR"].shift(i).fillna(0.0)
            api += lagged_p * (decay_factor ** i)
        self.df["API_7DAY"] = api

        return self.df


if __name__ == "__main__":
    from src.data.data_loader import DataLoader

    print(f"Loading raw data from: {RAW_WEATHER_DATA_PATH}")
    loader = DataLoader(RAW_WEATHER_DATA_PATH)
    df_raw = loader.load_csv()

    engineer = FeatureEngineer(df_raw)
    df_features = engineer.create_features()

    print("\n--- Feature Engineered Data Preview ---")
    print(df_features[["DATE", "PRECTOTCORR", "RAIN_3DAY", "RAIN_7DAY", "API_7DAY"]].head())

    print("\nColumns:")
    print(df_features.columns.tolist())

    df_features.to_csv(PROCESSED_FEATURES_PATH, index=False)
    print(f"\nFeature-engineered dataset saved successfully to {PROCESSED_FEATURES_PATH}!")
 