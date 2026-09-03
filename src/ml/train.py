import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# Support running directly or as module
try:
    from src.utils.config import (
        LABELED_DATA_PATH,
        MODEL_PATH,
        ML_FEATURE_COLS,
        TARGET_COL,
        TRAIN_END_YEAR,
        TEST_START_YEAR,
        RISK_LEVEL_THRESHOLDS,
    )
except ImportError:
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    from src.utils.config import (
        LABELED_DATA_PATH,
        MODEL_PATH,
        ML_FEATURE_COLS,
        TARGET_COL,
        TRAIN_END_YEAR,
        TEST_START_YEAR,
        RISK_LEVEL_THRESHOLDS,
    )


class FloodModelTrainer:
    """
    Trains and evaluates machine learning models for Flash Flood Prediction
    using chronological time-series splitting and disaster-centric metrics.
    """

    def __init__(self, data_path=LABELED_DATA_PATH):
        self.data_path = Path(data_path)
        self.features = ML_FEATURE_COLS
        self.target = TARGET_COL

    def load_and_split_data(self):
        print(f"Loading labeled dataset from: {self.data_path}")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Labeled data not found at {self.data_path}. Run create_labeled_dataset.py first.")

        df = pd.read_csv(self.data_path)
        df["YEAR"] = df["YEAR"].astype(int)

        # Chronological Split
        train_mask = df["YEAR"] <= TRAIN_END_YEAR
        test_mask = df["YEAR"] >= TEST_START_YEAR

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        X_train = train_df[self.features]
        y_train = train_df[self.target]

        X_test = test_df[self.features]
        y_test = test_df[self.target]

        print("\n=== Chronological Dataset Partition ===")
        print(f"Training Period:  2015 - {TRAIN_END_YEAR} | Samples: {len(X_train)} | Floods: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
        print(f"Testing Period:   {TEST_START_YEAR} - 2025 | Samples: {len(X_test)} | Floods: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

        return X_train, y_train, X_test, y_test, train_df, test_df

    def train_and_evaluate_models(self, X_train, y_train, X_test, y_test):
        candidates = {
            "Baseline Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))
            ]),
            "Random Forest Classifier": RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1
            ),
            "HistGradientBoosting": HistGradientBoostingClassifier(
                max_depth=6,
                class_weight="balanced",
                random_state=42
            )
        }

        results = {}

        print("\n" + "="*70)
        print("          MODEL TRAINING & EVALUATION BENCHMARK")
        print("="*70)

        for name, model in candidates.items():
            print(f"\nTraining [{name}]...")
            model.fit(X_train, y_train)

            # Predict on Test Set
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            cm = confusion_matrix(y_test, y_pred)

            tn, fp, fn, tp = cm.ravel()

            results[name] = {
                "model": model,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "confusion_matrix": {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                },
                "y_pred": y_pred,
                "y_prob": y_prob
            }

            print(f"Results for {name}:")
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall:    {rec:.4f}  (Crucial: minimizes missed disasters)")
            print(f"  F1 Score:  {f1:.4f}")
            print(f"  ROC-AUC:   {roc_auc:.4f}")
            print(f"  PR-AUC:    {pr_auc:.4f}")
            print(f"  Confusion Matrix: [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")

        # Select Best Model based on F1-Score & Recall
        # In disaster early warning, Recall takes precedence
        best_name = max(results.keys(), key=lambda k: (results[k]["f1_score"] * 0.5 + results[k]["recall"] * 0.5))
        best_result = results[best_name]

        print("\n" + "="*70)
        print(f"🏆 BEST MODEL SELECTED: {best_name}")
        print("="*70)
        print(f"F1 Score:  {best_result['f1_score']:.4f}")
        print(f"Recall:    {best_result['recall']:.4f}")
        print(f"Precision: {best_result['precision']:.4f}")
        print(f"ROC-AUC:   {best_result['roc_auc']:.4f}")

        return best_name, best_result, results

    def extract_feature_importances(self, model, X_val, y_val, feature_names):
        """Extract feature importances using tree importances or permutation importance."""
        from sklearn.inspection import permutation_importance
        
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_imp = {feat: float(imp) for feat, imp in zip(feature_names, importances)}
        else:
            perm = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42, scoring="f1")
            importances = np.maximum(0, perm.importances_mean)
            total = np.sum(importances)
            if total > 0:
                importances = importances / total
            feat_imp = {feat: float(imp) for feat, imp in zip(feature_names, importances)}

        # Sort descending
        return dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))

    def save_model_artifact(self, best_name, best_result, X_train, y_train, X_test, y_test):
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        best_model = best_result["model"]

        feature_importances = self.extract_feature_importances(best_model, X_test, y_test, self.features)

        artifact = {
            "model_name": best_name,
            "model": best_model,
            "feature_names": self.features,
            "target_col": self.target,
            "metrics": {
                "accuracy": best_result["accuracy"],
                "precision": best_result["precision"],
                "recall": best_result["recall"],
                "f1_score": best_result["f1_score"],
                "roc_auc": best_result["roc_auc"],
                "pr_auc": best_result["pr_auc"],
            },
            "confusion_matrix": best_result["confusion_matrix"],
            "feature_importances": feature_importances,
            "train_period": f"2015-{TRAIN_END_YEAR}",
            "test_period": f"{TEST_START_YEAR}-2025",
            "risk_thresholds": RISK_LEVEL_THRESHOLDS,
        }

        joblib.dump(artifact, MODEL_PATH)
        print(f"\nModel artifact saved successfully to:\n{MODEL_PATH}")

        if feature_importances:
            print("\nTop 7 Predictive Features:")
            for rank, (feat, imp) in enumerate(list(feature_importances.items())[:7], 1):
                print(f"  {rank}. {feat:<20} Importance: {imp:.4f}")

        return artifact


def run_training_pipeline():
    trainer = FloodModelTrainer()
    X_train, y_train, X_test, y_test, train_df, test_df = trainer.load_and_split_data()
    best_name, best_result, all_results = trainer.train_and_evaluate_models(
        X_train, y_train, X_test, y_test
    )
    artifact = trainer.save_model_artifact(best_name, best_result, X_train, y_train, X_test, y_test)
    return artifact


if __name__ == "__main__":
    run_training_pipeline()
