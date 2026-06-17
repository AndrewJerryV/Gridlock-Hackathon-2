"""
pcri.py — Parking Congestion Risk Index (PCRI) computation.
Custom score combining violation density, peak hour weight,
junction frequency, and repeat violations. Normalised 0-100.
"""

import pandas as pd
import numpy as np


# Peak hours for Bengaluru traffic
PEAK_HOURS = {8, 9, 10, 17, 18, 19, 20}


def compute_pcri(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute PCRI for each junction and area (police_station).

    Returns
    -------
    junction_pcri : DataFrame with PCRI per junction.
    area_pcri     : DataFrame with PCRI per police_station / area.
    """
    # --- Junction-level PCRI ---
    junctions = df[df["junction_name"] != "No Junction"].copy() if "junction_name" in df.columns else pd.DataFrame()

    if len(junctions) > 0:
        junction_pcri = _compute_for_group(junctions, group_col="junction_name")
    else:
        junction_pcri = pd.DataFrame()

    # --- Area-level PCRI (police_station) ---
    if "police_station" in df.columns:
        area_pcri = _compute_for_group(df, group_col="police_station")
    else:
        area_pcri = pd.DataFrame()

    return junction_pcri, area_pcri


def _compute_for_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute PCRI components for a given grouping column."""
    groups = df.groupby(group_col)

    # 1. Violation Density
    violation_density = groups["id"].count().rename("violation_density")

    # 2. Peak Hour Weight (fraction of violations during peak hours)
    if "hour" in df.columns:
        peak_weight = groups.apply(
            lambda g: (g["hour"].isin(PEAK_HOURS)).mean(), include_groups=False
        ).rename("peak_hour_weight")
    else:
        peak_weight = pd.Series(0.5, index=violation_density.index, name="peak_hour_weight")

    # 3. Junction Frequency (unique dates with violations)
    if "date" in df.columns:
        freq = groups["date"].nunique().rename("junction_frequency")
    else:
        freq = pd.Series(1, index=violation_density.index, name="junction_frequency")

    # 4. Repeat Violations (vehicles seen more than once)
    if "vehicle_number" in df.columns:
        repeat = groups.apply(
            lambda g: (g["vehicle_number"].value_counts() > 1).sum() / max(g["vehicle_number"].nunique(), 1),
            include_groups=False,
        ).rename("repeat_violation_rate")
    else:
        repeat = pd.Series(0, index=violation_density.index, name="repeat_violation_rate")

    # Combine
    pcri_df = pd.concat([violation_density, peak_weight, freq, repeat], axis=1).reset_index()
    pcri_df.rename(columns={"index": group_col}, inplace=True)

    # Normalise each component 0-1
    for col in ["violation_density", "peak_hour_weight", "junction_frequency", "repeat_violation_rate"]:
        if col in pcri_df.columns:
            cmin, cmax = pcri_df[col].min(), pcri_df[col].max()
            if cmax > cmin:
                pcri_df[f"{col}_norm"] = (pcri_df[col] - cmin) / (cmax - cmin)
            else:
                pcri_df[f"{col}_norm"] = 0.5

    # Weighted PCRI
    w = {"violation_density_norm": 0.35,
         "peak_hour_weight_norm": 0.25,
         "junction_frequency_norm": 0.25,
         "repeat_violation_rate_norm": 0.15}

    pcri_df["pcri_raw"] = sum(pcri_df[k] * v for k, v in w.items() if k in pcri_df.columns)

    # Scale to 0-100
    pmin, pmax = pcri_df["pcri_raw"].min(), pcri_df["pcri_raw"].max()
    if pmax > pmin:
        pcri_df["pcri"] = ((pcri_df["pcri_raw"] - pmin) / (pmax - pmin) * 100).round(1)
    else:
        pcri_df["pcri"] = 50.0

    # Risk category
    pcri_df["risk_level"] = pd.cut(
        pcri_df["pcri"],
        bins=[-1, 33, 66, 101],
        labels=["Low", "Medium", "High"],
    )

    pcri_df = pcri_df.sort_values("pcri", ascending=False).reset_index(drop=True)
    return pcri_df


def get_top_risky(pcri_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top N highest-PCRI entries."""
    return pcri_df.head(n)
