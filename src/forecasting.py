"""
forecasting.py — Time-series forecasting and anomaly detection.
Implements Prophet for violation forecasting and Isolation Forest
for anomaly detection in parking activity.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Forecasting with Prophet
# ─────────────────────────────────────────────────────────────

def prepare_prophet_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily violation counts for Prophet."""
    if "date" not in df.columns:
        return pd.DataFrame()

    daily = df.groupby("date")["id"].count().reset_index()
    daily.columns = ["ds", "y"]
    daily["ds"] = pd.to_datetime(daily["ds"])
    
    # Shift dates to current year (2026) to match hackathon timeline
    if len(daily) > 0:
        max_date = daily["ds"].max()
        target_end_date = pd.to_datetime("2026-06-19")
        offset = target_end_date - max_date
        daily["ds"] = daily["ds"] + offset

    daily = daily.sort_values("ds").reset_index(drop=True)
    return daily


def run_prophet_forecast(daily: pd.DataFrame, periods: int = 30):
    """Run Prophet forecasting for future violation counts."""
    try:
        from prophet import Prophet
    except ImportError:
        print("[WARN] Prophet not installed. Skipping forecast.")
        return None, None

    if len(daily) < 10:
        return None, None

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05,
    )
    model.fit(daily)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    return model, forecast


# ─────────────────────────────────────────────────────────────
# Anomaly Detection with Isolation Forest
# ─────────────────────────────────────────────────────────────

def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.05,
) -> pd.DataFrame:
    """
    Detect unusual parking activity using Isolation Forest.

    Features used: hour, weekday, latitude, longitude, historical count
    """
    feature_cols = []
    data = df.copy()

    for col in ["hour", "weekday", "latitude", "longitude"]:
        if col in data.columns:
            feature_cols.append(col)

    if "junction_name" in data.columns:
        junc_counts = data.groupby("junction_name")["id"].transform("count")
        data["junction_violation_count"] = junc_counts
        feature_cols.append("junction_violation_count")

    if len(feature_cols) == 0:
        data["is_anomaly"] = False
        return data

    X = data[feature_cols].fillna(0)

    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    preds = iso.fit_predict(X)

    # -1 = anomaly, 1 = normal
    data["anomaly_label"] = preds
    data["is_anomaly"] = preds == -1
    data["anomaly_score"] = iso.decision_function(X)

    n_anomalies = data["is_anomaly"].sum()
    print(f"[INFO] Isolation Forest detected {n_anomalies:,} anomalies "
          f"({n_anomalies / len(data) * 100:.1f}% of records).")

    return data


def get_anomaly_summary(df: pd.DataFrame) -> dict:
    """Summarise detected anomalies."""
    if "is_anomaly" not in df.columns:
        return {}

    anomalies = df[df["is_anomaly"]]
    summary = {
        "total_anomalies": len(anomalies),
        "anomaly_rate": round(len(anomalies) / len(df) * 100, 2),
    }

    if "hour" in anomalies.columns and len(anomalies) > 0:
        summary["peak_anomaly_hours"] = anomalies["hour"].value_counts().head(5).to_dict()

    if "junction_name" in anomalies.columns and len(anomalies) > 0:
        junc = anomalies[anomalies["junction_name"] != "No Junction"]["junction_name"]
        summary["top_anomaly_junctions"] = junc.value_counts().head(5).to_dict()

    if "weekday_name" in anomalies.columns and len(anomalies) > 0:
        summary["anomaly_by_day"] = anomalies["weekday_name"].value_counts().to_dict()

    return summary


# ─────────────────────────────────────────────────────────────
# Temporal Analysis Helpers
# ─────────────────────────────────────────────────────────────

def get_peak_hours(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Identify peak violation hours."""
    if "hour" not in df.columns:
        return pd.DataFrame()
    hourly = df.groupby("hour")["id"].count().reset_index()
    hourly.columns = ["hour", "violations"]
    hourly = hourly.sort_values("violations", ascending=False)
    return hourly.head(top_n)


def get_peak_days(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Identify peak violation days of the week."""
    if "weekday_name" not in df.columns:
        return pd.DataFrame()
    daily = df.groupby("weekday_name")["id"].count().reset_index()
    daily.columns = ["day", "violations"]
    daily = daily.sort_values("violations", ascending=False)
    return daily.head(top_n)
