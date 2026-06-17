"""
prediction.py — Hotspot Prediction Models (XGBoost + Random Forest).
Predicts: "Will a hotspot occur at a junction in the next hour?"
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Build feature matrix for hotspot prediction.

    Target: is_hotspot = 1 if the record belongs to a DBSCAN cluster (not noise).

    Features:
    - hour, weekday, is_weekend, month
    - vehicle_type (encoded)
    - junction_name (encoded)
    - historical_violation_count (per junction)
    - cluster_id
    """
    data = df.copy()

    # Target
    data["is_hotspot"] = (data["cluster_id"] != -1).astype(int)

    # Historical violation count per junction
    if "junction_name" in data.columns:
        junc_counts = data.groupby("junction_name")["id"].transform("count")
        data["historical_violation_count"] = junc_counts
    else:
        data["historical_violation_count"] = 0

    # Encode categoricals
    encoders = {}

    if "vehicle_type" in data.columns:
        le_vt = LabelEncoder()
        data["vehicle_type_enc"] = le_vt.fit_transform(data["vehicle_type"].astype(str))
        encoders["vehicle_type"] = le_vt
    else:
        data["vehicle_type_enc"] = 0

    if "junction_name" in data.columns:
        le_jn = LabelEncoder()
        data["junction_name_enc"] = le_jn.fit_transform(data["junction_name"].astype(str))
        encoders["junction_name"] = le_jn
    else:
        data["junction_name_enc"] = 0

    feature_cols = [
        "hour", "weekday", "is_weekend", "month",
        "vehicle_type_enc", "junction_name_enc",
        "historical_violation_count", "cluster_id",
    ]

    # Filter to only rows with required features
    available = [c for c in feature_cols if c in data.columns]
    X = data[available].fillna(0)
    y = data["is_hotspot"]

    return X, y, encoders


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train XGBoost and Random Forest, compare, and return results."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    results = {}

    # --- XGBoost ---
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

    results["xgboost"] = {
        "model": xgb_model,
        "accuracy": accuracy_score(y_test, xgb_pred),
        "precision": precision_score(y_test, xgb_pred, zero_division=0),
        "recall": recall_score(y_test, xgb_pred, zero_division=0),
        "f1": f1_score(y_test, xgb_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, xgb_prob),
        "y_test": y_test,
        "y_pred": xgb_pred,
        "y_prob": xgb_prob,
    }

    # --- Random Forest ---
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=random_state,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]

    results["random_forest"] = {
        "model": rf_model,
        "accuracy": accuracy_score(y_test, rf_pred),
        "precision": precision_score(y_test, rf_pred, zero_division=0),
        "recall": recall_score(y_test, rf_pred, zero_division=0),
        "f1": f1_score(y_test, rf_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, rf_prob),
        "y_test": y_test,
        "y_pred": rf_pred,
        "y_prob": rf_prob,
    }

    # Select best model by F1
    best_name = max(results, key=lambda k: results[k]["f1"])
    results["best_model_name"] = best_name
    results["best_model"] = results[best_name]["model"]
    results["feature_names"] = list(X.columns)
    results["X_test"] = X_test

    print(f"[INFO] Best model: {best_name} (F1={results[best_name]['f1']:.4f})")
    return results


def save_models(results: dict, encoders: dict, output_dir: str = "models"):
    """Persist trained models and encoders."""
    os.makedirs(output_dir, exist_ok=True)

    for name in ["xgboost", "random_forest"]:
        path = os.path.join(output_dir, f"{name}_model.pkl")
        joblib.dump(results[name]["model"], path)
        print(f"[INFO] Saved {name} model to {path}")

    enc_path = os.path.join(output_dir, "label_encoders.pkl")
    joblib.dump(encoders, enc_path)
    print(f"[INFO] Saved encoders to {enc_path}")

    # Save best model name
    meta_path = os.path.join(output_dir, "best_model.txt")
    with open(meta_path, "w") as f:
        f.write(results["best_model_name"])


def load_model(model_name: str = "best", model_dir: str = "models"):
    """Load a saved model."""
    if model_name == "best":
        meta_path = os.path.join(model_dir, "best_model.txt")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                model_name = f.read().strip()
        else:
            model_name = "xgboost"

    path = os.path.join(model_dir, f"{model_name}_model.pkl")
    if os.path.exists(path):
        return joblib.load(path), model_name
    raise FileNotFoundError(f"Model not found at {path}")


def predict_single(model, hour, weekday, is_weekend, month,
                   vehicle_type_enc, junction_name_enc,
                   historical_count, cluster_id):
    """Predict hotspot probability for a single input."""
    features = np.array([[
        hour, weekday, is_weekend, month,
        vehicle_type_enc, junction_name_enc,
        historical_count, cluster_id,
    ]])
    prob = model.predict_proba(features)[0][1]
    return prob
