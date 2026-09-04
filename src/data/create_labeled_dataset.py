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
        DOCUMENTED_TARGET_COL,
        RULE_BASELINE_COL,
        PROVENANCE_COL,
        LEGACY_TARGET_COL,
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
        DOCUMENTED_TARGET_COL,
        RULE_BASELINE_COL,
        PROVENANCE_COL,
        LEGACY_TARGET_COL,
        HISTORICAL_DISASTER_DATES,
        IMD_HEAVY_RAIN_MM,
        IMD_MODERATE_RAIN_MM,
        ANTECEDENT_3DAY_SATURATION_MM,
        ANTECEDENT_7DAY_SATURATION_MM,
        PROTRACTED_RAIN_MM,
    )


class FloodDatasetLabeler:
    """
    Constructs a scientifically grounded labeling pipeline that cleanly separates:
    1. DOCUMENTED_FLOOD_EVENT: Strict ground truth from historical disaster records.
    2. RULE_BASELINE_ALERT: Hydrometeorological heuristic thresholds (IMD/CWC).
    3. LABEL_PROVENANCE: Explicit provenance taxonomy (documented_and_rule, documented_event, rule_baseline_only, none).
    """

    def __init__(self, features_df):
        self.df = features_df.copy()

    def apply_labels(self):
        # Ensure DATE is string formatted YYYY-MM-DD
        self.df["DATE_STR"] = pd.to_datetime(self.df["DATE"]).dt.strftime("%Y-%m-%d")

        # 1. Historical Documented Disaster Dates (Strict ML Ground Truth)
        historical_mask = self.df["DATE_STR"].isin(HISTORICAL_DISASTER_DATES)
        self.df[DOCUMENTED_TARGET_COL] = historical_mask.astype(int)

        # 2. IMD / Hydrometeorological Heuristic Alert Thresholds
        imd_heavy_mask = self.df["PRECTOTCORR"] >= IMD_HEAVY_RAIN_MM
        sat_3day_mask = (
            (self.df["PRECTOTCORR"] >= IMD_MODERATE_RAIN_MM) &
            (self.df["RAIN_3DAY"] >= ANTECEDENT_3DAY_SATURATION_MM)
        )
        sat_7day_mask = (
            (self.df["PRECTOTCORR"] >= PROTRACTED_RAIN_MM) &
            (self.df["RAIN_7DAY"] >= ANTECEDENT_7DAY_SATURATION_MM)
        )

        rule_mask = imd_heavy_mask | sat_3day_mask | sat_7day_mask
        self.df[RULE_BASELINE_COL] = rule_mask.astype(int)

        # RULE_BASELINE_ALERT must NEVER modify DOCUMENTED_FLOOD_EVENT
        # 3. Explicit Provenance Classification
        provenance = []
        reasons = []

        for idx, row in self.df.iterrows():
            is_doc = historical_mask.iloc[idx]
            is_rule = rule_mask.iloc[idx]

            rule_details = []
            if imd_heavy_mask.iloc[idx]:
                rule_details.append(f"IMD Heavy Rain ({row['PRECTOTCORR']:.1f}mm)")
            if sat_3day_mask.iloc[idx]:
                rule_details.append(f"3-Day Saturated Runoff ({row['RAIN_3DAY']:.1f}mm)")
            if sat_7day_mask.iloc[idx]:
                rule_details.append(f"7-Day Antecedent Saturation ({row['RAIN_7DAY']:.1f}mm)")

            if is_doc and is_rule:
                provenance.append("documented_and_rule")
                reasons.append(f"Documented Historical Disaster Record + Rule Trigger: {'; '.join(rule_details)}")
            elif is_doc and not is_rule:
                provenance.append("documented_event")
                reasons.append("Documented Historical Disaster Record (Localized cloudburst / dam overspill / geomorphic breach)")
            elif not is_doc and is_rule:
                provenance.append("rule_baseline_only")
                reasons.append(f"Hydrometeorological Rule Alert Only (Unconfirmed disaster): {'; '.join(rule_details)}")
            else:
                provenance.append("none")
                reasons.append("Normal / No Flood")

        self.df[PROVENANCE_COL] = provenance
        self.df["LABEL_REASON"] = reasons

        # Backward-safe alias for legacy code
        self.df[LEGACY_TARGET_COL] = self.df[DOCUMENTED_TARGET_COL]

        self.df = self.df.drop(columns=["DATE_STR"])
        return self.df


def generate_labeled_dataset():
    print(f"Reading processed features from: {PROCESSED_FEATURES_PATH}")
    df_features = pd.read_csv(PROCESSED_FEATURES_PATH)

    labeler = FloodDatasetLabeler(df_features)
    df_labeled = labeler.apply_labels()

    total = len(df_labeled)
    doc_positives = (df_labeled[DOCUMENTED_TARGET_COL] == 1).sum()
    doc_negatives = (df_labeled[DOCUMENTED_TARGET_COL] == 0).sum()
    rule_alerts = (df_labeled[RULE_BASELINE_COL] == 1).sum()
    both_count = (df_labeled[PROVENANCE_COL] == "documented_and_rule").sum()
    doc_only = (df_labeled[PROVENANCE_COL] == "documented_event").sum()
    rule_only = (df_labeled[PROVENANCE_COL] == "rule_baseline_only").sum()
    none_count = (df_labeled[PROVENANCE_COL] == "none").sum()

    print("\n=======================================================")
    print("      SCIENTIFIC AUDIT & FLOOD LABELING SUMMARY")
    print("=======================================================")
    print(f"Total Observations:                {total} days")
    print(f"Documented Flood Positives (ML):   {doc_positives} ({doc_positives/total*100:.2f}%)")
    print(f"Non-Documented Days:               {doc_negatives} ({doc_negatives/total*100:.2f}%)")
    print(f"Rule-Baseline Alerts:              {rule_alerts} ({rule_alerts/total*100:.2f}%)")
    print("-------------------------------------------------------")
    print(f"Provenance Breakdown:")
    print(f"  - Documented & Rule Trigger:     {both_count}")
    print(f"  - Documented Event Only:         {doc_only}")
    print(f"  - Rule-Baseline Only (Unconfirmed): {rule_only}")
    print(f"  - None:                          {none_count}")
    print("=======================================================")

    print("\nSample Rows with Differing Provenances:")
    sample_cols = ["DATE", "PRECTOTCORR", "RAIN_3DAY", DOCUMENTED_TARGET_COL, RULE_BASELINE_COL, PROVENANCE_COL]
    
    # Show samples of each category
    samples = pd.concat([
        df_labeled[df_labeled[PROVENANCE_COL] == "documented_and_rule"].head(3),
        df_labeled[df_labeled[PROVENANCE_COL] == "documented_event"].head(3),
        df_labeled[df_labeled[PROVENANCE_COL] == "rule_baseline_only"].head(3),
    ])[sample_cols]
    print(samples.to_string(index=False))

    df_labeled.to_csv(LABELED_DATA_PATH, index=False)
    print(f"\nSuccessfully regenerated and saved dataset to:\n{LABELED_DATA_PATH}")
    return df_labeled


if __name__ == "__main__":
    generate_labeled_dataset()

