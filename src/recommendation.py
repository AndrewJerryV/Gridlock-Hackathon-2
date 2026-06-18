"""
recommendation.py — Enforcement Recommendation Engine.
Ranks junctions by PCRI and prediction probability to generate
patrol deployment recommendations.
"""

import pandas as pd
import numpy as np


def generate_recommendations(
    junction_pcri: pd.DataFrame,
    df: pd.DataFrame,
    model=None,
    encoders: dict = None,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Generate ranked enforcement recommendations per junction.

    Combines:
    - PCRI score (from pcri.py)
    - Hotspot prediction probability (from prediction model)
    - Historical violation count
    """
    if junction_pcri is None or len(junction_pcri) == 0:
        return pd.DataFrame()

    recs = junction_pcri.copy()
    group_col = "junction_name" if "junction_name" in recs.columns else recs.columns[0]

    # Compute prediction probabilities per junction if model is available
    if model is not None and encoders is not None:
        probs = []
        for _, row in recs.iterrows():
            junc = row[group_col]
            try:
                junc_data = df[df["junction_name"] == junc]
                if len(junc_data) == 0:
                    probs.append(0.5)
                    continue

                # Use most common hour and weekday for the junction
                common_hour = int(junc_data["hour"].mode().iloc[0]) if "hour" in junc_data.columns else 12
                common_weekday = int(junc_data["weekday"].mode().iloc[0]) if "weekday" in junc_data.columns else 2
                is_weekend = 1 if common_weekday >= 5 else 0
                common_month = int(junc_data["month"].mode().iloc[0]) if "month" in junc_data.columns else 1

                # Encode vehicle type and junction name
                vt_enc = 0
                if "vehicle_type" in encoders:
                    common_vt = junc_data["vehicle_type"].mode().iloc[0] if "vehicle_type" in junc_data.columns else "Unknown"
                    try:
                        vt_enc = int(encoders["vehicle_type"].transform([common_vt])[0])
                    except ValueError:
                        vt_enc = 0

                jn_enc = 0
                if "junction_name" in encoders:
                    try:
                        jn_enc = int(encoders["junction_name"].transform([junc])[0])
                    except ValueError:
                        jn_enc = 0

                hist_count = len(junc_data)

                features = np.array([[
                    common_hour, common_weekday, is_weekend, common_month,
                    vt_enc, jn_enc, hist_count,
                ]])
                prob = float(model.predict_proba(features)[0][1])
                probs.append(prob)
            except Exception:
                probs.append(0.5)

        recs["hotspot_probability"] = probs
    else:
        # Estimate from PCRI score
        recs["hotspot_probability"] = (recs["pcri"] / 100).clip(0, 1)

    recs["hotspot_probability_pct"] = (recs["hotspot_probability"] * 100).round(1)

    # Combined enforcement priority score
    recs["enforcement_priority"] = (
        0.6 * recs["pcri"] + 0.4 * recs["hotspot_probability_pct"]
    ).round(1)

    recs = recs.sort_values("enforcement_priority", ascending=False).reset_index(drop=True)
    recs["rank"] = range(1, len(recs) + 1)

    # Priority zone classification
    recs["priority_zone"] = pd.cut(
        recs["enforcement_priority"],
        bins=[-1, 33, 66, 101],
        labels=["Routine", "Elevated", "Critical"],
    )

    # Generate recommendations text
    recs["recommendation"] = recs.apply(
        lambda r: _format_recommendation(r, group_col), axis=1
    )

    return recs.head(top_n)


def _format_recommendation(row, group_col):
    """Format a human-readable recommendation."""
    junc = row[group_col]
    pcri = row["pcri"]
    prob = row["hotspot_probability_pct"]
    zone = row["priority_zone"]

    if zone == "Critical":
        action = "🚨 IMMEDIATE DEPLOYMENT REQUIRED"
    elif zone == "Elevated":
        action = "⚠️ Schedule priority patrol"
    else:
        action = "📋 Include in routine patrol"

    return (
        f"{action}\n"
        f"Deploy Patrol Unit → {junc}\n"
        f"PCRI: {pcri} | Hotspot Probability: {prob}%\n"
        f"Zone: {zone}"
    )


def get_priority_zones(recs: pd.DataFrame) -> dict:
    """Group recommendations by priority zone."""
    if len(recs) == 0:
        return {"Critical": [], "Elevated": [], "Routine": []}

    group_col = "junction_name" if "junction_name" in recs.columns else recs.columns[0]
    zones = {}
    for zone in ["Critical", "Elevated", "Routine"]:
        subset = recs[recs["priority_zone"] == zone]
        zones[zone] = subset[group_col].tolist()
    return zones
