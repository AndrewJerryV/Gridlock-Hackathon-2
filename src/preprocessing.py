"""
preprocessing.py — Data ingestion, cleaning, and feature engineering.
Handles automatic CSV detection, missing values, duplicate removal,
datetime parsing, and temporal feature extraction.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path


def find_csv(project_root: str = ".") -> str:
    """Auto-detect CSV files in the project folder and subfolders."""
    patterns = [
        os.path.join(project_root, "*.csv"),
        os.path.join(project_root, "data", "*.csv"),
    ]
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            # Return the largest CSV (most likely the dataset)
            return max(files, key=os.path.getsize)
    raise FileNotFoundError("No CSV dataset found in the project folder.")


def load_data(filepath: str | None = None, project_root: str = ".") -> pd.DataFrame:
    """Load the parking violations dataset."""
    if filepath is None:
        filepath = find_csv(project_root)
    print(f"[INFO] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"[INFO] Loaded {len(df):,} records with {len(df.columns)} columns.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and remove duplicates."""
    initial_rows = len(df)

    # Remove exact duplicates
    df = df.drop_duplicates()
    dupes_removed = initial_rows - len(df)
    print(f"[INFO] Removed {dupes_removed:,} duplicate rows.")

    # Fill missing text fields
    text_cols = ["location", "vehicle_type", "description", "violation_type",
                 "junction_name", "police_station", "center_code",
                 "updated_vehicle_number", "updated_vehicle_type",
                 "validation_status"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            # Replace NULL strings
            df[col] = df[col].replace("NULL", "Unknown")
            df[col] = df[col].replace("null", "Unknown")

    # Fill missing numeric fields
    if "latitude" in df.columns:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    if "longitude" in df.columns:
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Drop rows without valid coordinates
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    print(f"[INFO] Dropped {before - len(df):,} rows with missing coordinates.")

    # Filter to Bengaluru bounding box (approx)
    df = df[
        (df["latitude"] > 12.7) & (df["latitude"] < 13.2) &
        (df["longitude"] > 77.4) & (df["longitude"] < 77.8)
    ]
    print(f"[INFO] {len(df):,} records within Bengaluru bounds.")

    return df.reset_index(drop=True)


DATETIME_COLS = [
    "created_datetime", "closed_datetime", "modified_datetime",
    "action_taken_timestamp", "data_sent_to_scita_timestamp",
    "validation_timestamp"
]


def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all datetime columns to proper datetime types."""
    for col in DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            # Convert to IST
            df[col] = df[col].dt.tz_convert("Asia/Kolkata")
    return df


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, weekday, month, weekend flag from created_datetime."""
    if "created_datetime" not in df.columns:
        return df

    dt = df["created_datetime"]
    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.dayofweek          # 0=Monday
    df["weekday_name"] = dt.dt.day_name()
    df["month"] = dt.dt.month
    df["month_name"] = dt.dt.month_name()
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["date"] = dt.dt.date
    df["week"] = dt.dt.isocalendar().week.fillna(0).astype(int)

    return df


def parse_violation_types(df: pd.DataFrame) -> pd.DataFrame:
    """Parse violation_type JSON-like list into a cleaner format."""
    import ast

    def _clean(val):
        if pd.isna(val) or val in ("Unknown", "NULL"):
            return "Unknown"
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return ", ".join(str(v).strip() for v in parsed)
        except (ValueError, SyntaxError):
            pass
        return str(val).strip('"[]')

    if "violation_type" in df.columns:
        df["violation_type_clean"] = df["violation_type"].apply(_clean)
    return df


def generate_summary(df: pd.DataFrame) -> dict:
    """Generate summary statistics for the dashboard."""
    summary = {
        "total_records": len(df),
        "unique_vehicles": df["vehicle_number"].nunique() if "vehicle_number" in df.columns else 0,
        "unique_junctions": df["junction_name"].nunique() if "junction_name" in df.columns else 0,
        "unique_locations": df["location"].nunique() if "location" in df.columns else 0,
        "date_range": None,
        "vehicle_type_dist": {},
        "top_violations": {},
        "top_junctions": {},
        "top_police_stations": {},
    }
    if "created_datetime" in df.columns:
        valid = df["created_datetime"].dropna()
        if len(valid) > 0:
            summary["date_range"] = (valid.min(), valid.max())

    if "vehicle_type" in df.columns:
        summary["vehicle_type_dist"] = df["vehicle_type"].value_counts().head(10).to_dict()

    if "violation_type_clean" in df.columns:
        summary["top_violations"] = df["violation_type_clean"].value_counts().head(10).to_dict()
    elif "violation_type" in df.columns:
        summary["top_violations"] = df["violation_type"].value_counts().head(10).to_dict()

    if "junction_name" in df.columns:
        junc = df[df["junction_name"] != "No Junction"]["junction_name"]
        summary["top_junctions"] = junc.value_counts().head(10).to_dict()

    if "police_station" in df.columns:
        summary["top_police_stations"] = df["police_station"].value_counts().head(10).to_dict()

    return summary


def run_preprocessing(project_root: str = ".") -> tuple[pd.DataFrame, dict]:
    """Full preprocessing pipeline."""
    df = load_data(project_root=project_root)
    df = clean_data(df)
    df = parse_datetimes(df)
    df = extract_temporal_features(df)
    df = parse_violation_types(df)
    summary = generate_summary(df)
    return df, summary


if __name__ == "__main__":
    df, summary = run_preprocessing()
    print("\n=== Dataset Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
