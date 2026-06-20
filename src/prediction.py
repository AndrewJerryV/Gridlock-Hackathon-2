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


def prepare_features(df: pd.DataFrame, encoders: dict = None) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Build feature matrix for hotspot prediction.

    Target: is_hotspot = 1 if the record belongs to a DBSCAN cluster (not noise).

    Features:
    - hour, weekday, is_weekend, month
    - vehicle_type (encoded)
    - junction_name (encoded)
    - police_station (encoded)
    - violation_type_clean (encoded)
    - historical_violation_count (per junction)
    - junc_hour_count (violation count per junction and hour)
    - junc_weekday_count (violation count per junction and weekday)
    - junc_month_count (violation count per junction and month)
    - junc_veh_count (violation count per junction and vehicle type)
    - veh_hour_count (violation count per vehicle type and hour)
    - stat_hour_count (violation count per police station and hour)
    - junc_hour_veh_count (violation count per junction, hour, and vehicle type)
    - junc_hour_vtc_count (violation count per junction, hour, and violation type)
    """
    data = df.copy()

    # Target
    data["is_hotspot"] = (data["cluster_id"] != -1).astype(int)

    # Historical violation counts
    if "junction_name" in data.columns:
        junc_counts = data.groupby("junction_name")["id"].transform("count")
        data["historical_violation_count"] = junc_counts
        
        # Junction-hour violation count (frequency at this specific hour)
        junc_hour_counts = data.groupby(["junction_name", "hour"])["id"].transform("count")
        data["junc_hour_count"] = junc_hour_counts
        
        # Junction-weekday violation count
        junc_weekday_counts = data.groupby(["junction_name", "weekday"])["id"].transform("count")
        data["junc_weekday_count"] = junc_weekday_counts
        
        # Junction-month violation count
        junc_month_counts = data.groupby(["junction_name", "month"])["id"].transform("count")
        data["junc_month_count"] = junc_month_counts
    else:
        data["historical_violation_count"] = 0
        data["junc_hour_count"] = 0
        data["junc_weekday_count"] = 0
        data["junc_month_count"] = 0

    # Junction-vehicle type interaction
    if "junction_name" in data.columns and "vehicle_type" in data.columns:
        junc_veh_counts = data.groupby(["junction_name", "vehicle_type"])["id"].transform("count")
        data["junc_veh_count"] = junc_veh_counts
        
        # Junction-hour-vehicle type three-way interaction
        junc_hour_veh_counts = data.groupby(["junction_name", "hour", "vehicle_type"])["id"].transform("count")
        data["junc_hour_veh_count"] = junc_hour_veh_counts
    else:
        data["junc_veh_count"] = 0
        data["junc_hour_veh_count"] = 0
        
    # Junction-hour-violation type clean three-way interaction
    if "junction_name" in data.columns and "violation_type_clean" in data.columns and "hour" in data.columns:
        junc_hour_vtc_counts = data.groupby(["junction_name", "hour", "violation_type_clean"])["id"].transform("count")
        data["junc_hour_vtc_count"] = junc_hour_vtc_counts
    else:
        data["junc_hour_vtc_count"] = 0

    # Vehicle-hour interaction
    if "vehicle_type" in data.columns:
        veh_hour_counts = data.groupby(["vehicle_type", "hour"])["id"].transform("count")
        data["veh_hour_count"] = veh_hour_counts
    else:
        data["veh_hour_count"] = 0
        
    # Station-hour interaction
    if "police_station" in data.columns:
        stat_hour_counts = data.groupby(["police_station", "hour"])["id"].transform("count")
        data["stat_hour_count"] = stat_hour_counts
    else:
        data["stat_hour_count"] = 0

    # Extra features
    if "junction_name" in data.columns and "hour" in data.columns:
        data["junc_weekday_hour_count"] = data.groupby(["junction_name", "weekday", "hour"])["id"].transform("count")
        data["junc_month_hour_count"] = data.groupby(["junction_name", "month", "hour"])["id"].transform("count")
    else:
        data["junc_weekday_hour_count"] = 0
        data["junc_month_hour_count"] = 0

    if "junction_name" in data.columns and "vehicle_type" in data.columns and "violation_type_clean" in data.columns:
        data["junc_veh_vtc_count"] = data.groupby(["junction_name", "vehicle_type", "violation_type_clean"])["id"].transform("count")
    else:
        data["junc_veh_vtc_count"] = 0

    if "vehicle_type" in data.columns and "hour" in data.columns:
        data["veh_weekday_hour_count"] = data.groupby(["vehicle_type", "weekday", "hour"])["id"].transform("count")
    else:
        data["veh_weekday_hour_count"] = 0

    if "police_station" in data.columns and "hour" in data.columns:
        data["stat_weekday_hour_count"] = data.groupby(["police_station", "weekday", "hour"])["id"].transform("count")
    else:
        data["stat_weekday_hour_count"] = 0

    if "vehicle_type" in data.columns and "violation_type_clean" in data.columns:
        data["veh_vtc_count"] = data.groupby(["vehicle_type", "violation_type_clean"])["id"].transform("count")
    else:
        data["veh_vtc_count"] = 0

    # Encode categoricals
    out_encoders = {}

    if "vehicle_type" in data.columns:
        if encoders and "vehicle_type" in encoders:
            le_vt = encoders["vehicle_type"]
            classes = set(le_vt.classes_)
            data["vehicle_type_enc"] = data["vehicle_type"].astype(str).apply(
                lambda x: le_vt.transform([x])[0] if x in classes else 0
            )
            out_encoders["vehicle_type"] = le_vt
        else:
            le_vt = LabelEncoder()
            data["vehicle_type_enc"] = le_vt.fit_transform(data["vehicle_type"].astype(str))
            out_encoders["vehicle_type"] = le_vt
    else:
        data["vehicle_type_enc"] = 0

    if "junction_name" in data.columns:
        if encoders and "junction_name" in encoders:
            le_jn = encoders["junction_name"]
            classes = set(le_jn.classes_)
            data["junction_name_enc"] = data["junction_name"].astype(str).apply(
                lambda x: le_jn.transform([x])[0] if x in classes else 0
            )
            out_encoders["junction_name"] = le_jn
        else:
            le_jn = LabelEncoder()
            data["junction_name_enc"] = le_jn.fit_transform(data["junction_name"].astype(str))
            out_encoders["junction_name"] = le_jn
    else:
        data["junction_name_enc"] = 0

    if "police_station" in data.columns:
        if encoders and "police_station" in encoders:
            le_ps = encoders["police_station"]
            classes = set(le_ps.classes_)
            data["police_station_enc"] = data["police_station"].astype(str).apply(
                lambda x: le_ps.transform([x])[0] if x in classes else 0
            )
            out_encoders["police_station"] = le_ps
        else:
            le_ps = LabelEncoder()
            data["police_station_enc"] = le_ps.fit_transform(data["police_station"].astype(str))
            out_encoders["police_station"] = le_ps
    else:
        data["police_station_enc"] = 0

    if "violation_type_clean" in data.columns:
        if encoders and "violation_type_clean" in encoders:
            le_vtc = encoders["violation_type_clean"]
            classes = set(le_vtc.classes_)
            data["violation_type_clean_enc"] = data["violation_type_clean"].astype(str).apply(
                lambda x: le_vtc.transform([x])[0] if x in classes else 0
            )
            out_encoders["violation_type_clean"] = le_vtc
        else:
            le_vtc = LabelEncoder()
            data["violation_type_clean_enc"] = le_vtc.fit_transform(data["violation_type_clean"].astype(str))
            out_encoders["violation_type_clean"] = le_vtc
    else:
        data["violation_type_clean_enc"] = 0

    feature_cols = [
        "hour", "weekday", "is_weekend", "month",
        "vehicle_type_enc", "junction_name_enc", "police_station_enc", "violation_type_clean_enc",
        "historical_violation_count", "junc_hour_count", "junc_weekday_count", "junc_month_count",
        "junc_veh_count", "veh_hour_count", "stat_hour_count",
        "junc_hour_veh_count", "junc_hour_vtc_count",
        "junc_weekday_hour_count", "junc_veh_vtc_count", "veh_weekday_hour_count", "stat_weekday_hour_count",
        "veh_vtc_count", "junc_month_hour_count",
        "latitude", "longitude"
    ]

    # Filter to only rows with required features
    available = [c for c in feature_cols if c in data.columns]
    X = data[available].fillna(0)
    y = data["is_hotspot"]

    return X, y, out_encoders


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
        n_estimators=300,
        max_depth=14,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.2,
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
        n_estimators=300,
        max_depth=None,
        min_samples_split=10,
        class_weight="balanced",
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

    # Save precomputed evaluation metrics
    metrics_data = []
    for name in ["xgboost", "random_forest"]:
        metrics_data.append({
            "Model": name.replace("_", " ").title(),
            "Accuracy": results[name]["accuracy"],
            "Precision": results[name]["precision"],
            "Recall": results[name]["recall"],
            "F1 Score": results[name]["f1"],
            "ROC-AUC": results[name]["roc_auc"],
        })
    metrics_path = os.path.join(output_dir, "metrics.pkl")
    joblib.dump(metrics_data, metrics_path)
    print(f"[INFO] Saved precomputed metrics to {metrics_path}")


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
                   historical_count, cluster_id=None):
    """Predict hotspot probability for a single input."""
    features = np.array([[
        hour, weekday, is_weekend, month,
        vehicle_type_enc, junction_name_enc,
        historical_count,
    ]])
    prob = model.predict_proba(features)[0][1]
    return prob
