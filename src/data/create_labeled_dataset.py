import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Support running directly or as module
try:
    from src.utils.config import (
        PROCESSED_FEATURES_PATH,
        LABELED_DATA_PATH,
        TARGET_COL,
        HISTORICAL_DISASTER_DATES,
        IMD_HEAVY_RAIN_MM,
        IMD_MODERATE_RAIN_MM,
        ANTECEDENT_3DAY_SATURATION_MM,
        ANTECEDENT_7DAY_SATURATION_MM,
        PROTRACTED_RAIN_MM,
    )
except ImportError:
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    from src.utils.config import (
        PROCESSED_FEATURES_PATH,
        LABELED_DATA_PATH,
        TARGET_COL,
        HISTORICAL_DISASTER_DATES,
        IMD_HEAVY_RAIN_MM,
        IMD_MODERATE_RAIN_MM,
        ANTECEDENT_3DAY_SATURATION_MM,
        ANTECEDENT_7DAY_SATURATION_MM,
        PROTRACTED_RAIN_MM,
    )


class FloodDatasetLabeler:
    """
    Constructs a scientifically defensible ground-truth flash flood label
    combining verified historical disaster chronicles and IMD/CWC hydrological criteria.
    """

    def __init__(self, features_df):
        self.df = features_df.copy()

    def apply_labels(self):
        # Ensure DATE is string formatted YYYY-MM-DD
        self.df["DATE_STR"] = pd.to_datetime(self.df["DATE"]).dt.strftime("%Y-%m-%d")

        # 1. Historical Documented Disaster Dates
        historical_mask = self.df["DATE_STR"].isin(HISTORICAL_DISASTER_DATES)

        # 2. IMD Heavy Rainfall Threshold (>= 64.5 mm/day)
        imd_heavy_mask = self.df["PRECTOTCORR"] >= IMD_HEAVY_RAIN_MM

        # 3. Antecedent 3-Day Saturation + Intense Rain
        sat_3day_mask = (
            (self.df["PRECTOTCORR"] >= IMD_MODERATE_RAIN_MM) &
            (self.df["RAIN_3DAY"] >= ANTECEDENT_3DAY_SATURATION_MM)
        )

        # 4. Protracted 7-Day Cumulative Saturation + Significant Rain
        sat_7day_mask = (
            (self.df["PRECTOTCORR"] >= PROTRACTED_RAIN_MM) &
            (self.df["RAIN_7DAY"] >= ANTECEDENT_7DAY_SATURATION_MM)
        )

        # Combined Flood Occurrence Target
        combined_mask = historical_mask | imd_heavy_mask | sat_3day_mask | sat_7day_mask
        self.df[TARGET_COL] = combined_mask.astype(int)

        # Record reason for transparency & verification
        reasons = []
        for idx, row in self.df.iterrows():
            r = []
            if historical_mask.iloc[idx]:
                r.append("Historical Disaster Record")
            if imd_heavy_mask.iloc[idx]:
                r.append(f"IMD Heavy Rain ({row['PRECTOTCORR']:.1f}mm)")
            if sat_3day_mask.iloc[idx]:
                r.append(f"3-Day Saturated Runoff ({row['RAIN_3DAY']:.1f}mm)")
            if sat_7day_mask.iloc[idx]:
                r.append(f"7-Day Antecedent Saturation ({row['RAIN_7DAY']:.1f}mm)")
            reasons.append("; ".join(r) if r else "Normal / No Flood")

        self.df["LABEL_REASON"] = reasons
        self.df = self.df.drop(columns=["DATE_STR"])

        return self.df


def generate_labeled_dataset():
    print(f"Reading processed features from: {PROCESSED_FEATURES_PATH}")
    df_features = pd.read_csv(PROCESSED_FEATURES_PATH)

    labeler = FloodDatasetLabeler(df_features)
    df_labeled = labeler.apply_labels()

    total = len(df_labeled)
    positives = (df_labeled[TARGET_COL] == 1).sum()
    negatives = (df_labeled[TARGET_COL] == 0).sum()
    pct_pos = (positives / total) * 100

    print("\n=======================================================")
    print("      GROUND TRUTH FLOOD DATASET SUMMARY")
    print("=======================================================")
    print(f"Total Observations:        {total} days")
    print(f"Non-Flood Days (Class 0):  {negatives} ({100 - pct_pos:.2f}%)")
    print(f"Flood / Risk Days (Class 1): {positives} ({pct_pos:.2f}%)")
    print("=======================================================")

    print("\nSample Positive Ground Truth Flood Events:")
    sample_cols = ["DATE", "PRECTOTCORR", "RAIN_3DAY", "RAIN_7DAY", TARGET_COL, "LABEL_REASON"]
    print(df_labeled[df_labeled[TARGET_COL] == 1][sample_cols].head(12).to_string(index=False))

    df_labeled.to_csv(LABELED_DATA_PATH, index=False)
    print(f"\nSuccessfully generated and saved labeled dataset to:\n{LABELED_DATA_PATH}")
    return df_labeled


if __name__ == "__main__":
    generate_labeled_dataset()
