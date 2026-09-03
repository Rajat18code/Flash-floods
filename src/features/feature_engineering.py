import pandas as pd


class FeatureEngineer:
    """
    Creates weather-based features for flash flood prediction.
    """

    def __init__(self, df):
        self.df = df.copy()

    def create_features(self):
        """
        Create rainfall and weather features.
        """

        # Sort data chronologically
        self.df = self.df.sort_values(
            by=["YEAR", "MO", "DY"]
        ).reset_index(drop=True)

        # Create proper date column
        self.df["DATE"] = pd.to_datetime(
            self.df[["YEAR", "MO", "DY"]]
            .rename(
                columns={
                    "YEAR": "year",
                    "MO": "month",
                    "DY": "day"
                }
            )
        )

        # Rainfall features
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

        # Recent rainfall intensity
        self.df["RAIN_INTENSITY"] = (
            self.df["PRECTOTCORR"]
            / self.df["RAIN_3DAY"].replace(0, 1)
        )

        # Temperature and humidity interaction
        self.df["TEMP_HUMIDITY"] = (
            self.df["T2M"] * self.df["RH2M"]
        )

        # Wind-related feature
        self.df["WIND_RAIN"] = (
            self.df["WS2M"] * self.df["PRECTOTCORR"]
        )

        return self.df

if __name__ == "__main__":
    from src.data.data_loader import DataLoader

    data_path = "data/raw/himachal_weather.csv"

    loader = DataLoader(data_path)
    df = loader.load_csv()

    engineer = FeatureEngineer(df)
    df_features = engineer.create_features()

    print("\n--- Feature Engineered Data ---")
    print(df_features.head())

    print("\nColumns:")
    print(df_features.columns.tolist())

    df_features.to_csv(
    "data/processed/himachal_features.csv",
    index=False
)

    print("\nFeature-engineered dataset saved successfully!") 