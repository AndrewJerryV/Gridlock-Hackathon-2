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


# ─── Feature 1: Resource-Constrained Dispatch Engine ──────────

def optimize_dispatch(
    recs: pd.DataFrame,
    available_units: int = 5,
) -> pd.DataFrame:
    """
    Greedy knapsack-style optimizer: given a limited number of patrol/tow
    units, select the combination of junctions that maximises total
    congestion relief (sum of enforcement_priority).

    Each junction is assumed to require 1 unit. The algorithm simply
    picks the top-N by enforcement_priority, which is optimal for a
    unit-weight knapsack.

    Returns a DataFrame of selected junctions with cumulative stats.
    """
    if len(recs) == 0 or available_units <= 0:
        return pd.DataFrame()

    group_col = "junction_name" if "junction_name" in recs.columns else recs.columns[0]

    # Sort by enforcement_priority descending (already sorted, but be safe)
    ranked = recs.sort_values("enforcement_priority", ascending=False).head(available_units).copy()
    ranked = ranked.reset_index(drop=True)
    ranked["dispatch_order"] = range(1, len(ranked) + 1)

    # Cumulative PCRI relief
    ranked["cumulative_pcri_relief"] = ranked["pcri"].cumsum().round(1)

    # Estimated congestion reduction %  (heuristic: each cleared junction
    # removes its share of the total PCRI from the network)
    total_pcri = recs["pcri"].sum()
    if total_pcri > 0:
        ranked["est_congestion_reduction_pct"] = (
            ranked["cumulative_pcri_relief"] / total_pcri * 100
        ).round(1)
    else:
        ranked["est_congestion_reduction_pct"] = 0.0

    return ranked


# ─── Feature 2: What-If Impact Simulator ─────────────────────

def simulate_clearance(
    junction_pcri: pd.DataFrame,
    cleared_junctions: list[str],
) -> pd.DataFrame:
    """
    Simulate clearing enforcement at selected junctions and recalculate
    network-wide impact.

    Returns a copy of junction_pcri with:
      - cleared junctions PCRI set to 0
      - a new 'pcri_delta' column showing the change
      - updated network-level stats
    """
    if junction_pcri is None or len(junction_pcri) == 0:
        return junction_pcri

    group_col = "junction_name" if "junction_name" in junction_pcri.columns else junction_pcri.columns[0]

    simulated = junction_pcri.copy()
    simulated["original_pcri"] = simulated["pcri"]
    simulated["is_cleared"] = simulated[group_col].isin(cleared_junctions)

    # Cleared junctions drop to 0 PCRI
    simulated.loc[simulated["is_cleared"], "pcri"] = 0.0

    # Recalculate risk_level
    simulated["risk_level"] = pd.cut(
        simulated["pcri"],
        bins=[-1, 33, 66, 101],
        labels=["Low", "Medium", "High"],
    )

    # Delta
    simulated["pcri_delta"] = simulated["pcri"] - simulated["original_pcri"]

    return simulated


def get_clearance_impact(
    original_pcri: pd.DataFrame,
    simulated_pcri: pd.DataFrame,
) -> dict:
    """Compute before/after network statistics for the what-if simulation."""
    orig_total = original_pcri["pcri"].sum()
    sim_total = simulated_pcri["pcri"].sum()

    orig_high = len(original_pcri[original_pcri["risk_level"] == "High"])
    sim_high = len(simulated_pcri[simulated_pcri["risk_level"] == "High"])

    orig_mean = original_pcri["pcri"].mean()
    sim_mean = simulated_pcri["pcri"].mean()

    reduction_pct = ((orig_total - sim_total) / orig_total * 100) if orig_total > 0 else 0

    return {
        "original_total_pcri": round(orig_total, 1),
        "simulated_total_pcri": round(sim_total, 1),
        "pcri_reduction_pct": round(reduction_pct, 1),
        "original_high_risk": orig_high,
        "simulated_high_risk": sim_high,
        "high_risk_eliminated": orig_high - sim_high,
        "original_mean_pcri": round(orig_mean, 1),
        "simulated_mean_pcri": round(sim_mean, 1),
    }


# ─── Feature 3: Dispatch Briefing Generator ──────────────────

def generate_dispatch_briefing(
    recs: pd.DataFrame,
    dispatch_plan: pd.DataFrame = None,
    impact: dict = None,
) -> str:
    """
    Generate a formatted text dispatch briefing for download by
    control room operators and field officers.

    Parameters
    ----------
    recs : Full recommendation DataFrame
    dispatch_plan : Output of optimize_dispatch (constrained plan)
    impact : Output of get_clearance_impact (what-if stats)

    Returns
    -------
    Formatted string ready for download as .txt
    """
    from datetime import datetime

    now = datetime.now().strftime("%d %B %Y, %I:%M %p IST")
    group_col = "junction_name" if "junction_name" in recs.columns else recs.columns[0]

    lines = []
    lines.append("=" * 64)
    lines.append("  PARKIQ — DAILY ENFORCEMENT DISPATCH BRIEFING")
    lines.append("  Bengaluru Traffic Police — AI Parking Intelligence")
    lines.append(f"  Generated: {now}")
    lines.append("=" * 64)
    lines.append("")

    # ── Section 1: Optimised Dispatch Plan ──
    if dispatch_plan is not None and len(dispatch_plan) > 0:
        lines.append("─" * 64)
        lines.append(f"  OPTIMISED DISPATCH PLAN  ({len(dispatch_plan)} Units Deployed)")
        lines.append("─" * 64)
        lines.append("")
        lines.append(f"  {'#':<4} {'Junction':<35} {'Priority':>8}  {'Zone':>10}")
        lines.append(f"  {'─'*4} {'─'*35} {'─'*8}  {'─'*10}")

        for _, row in dispatch_plan.iterrows():
            order = int(row["dispatch_order"])
            junc = str(row[group_col])[:35]
            priority = row["enforcement_priority"]
            zone = row.get("priority_zone", "—")
            lines.append(f"  {order:<4} {junc:<35} {priority:>8.1f}  {zone:>10}")

        lines.append("")
        total_relief = dispatch_plan["pcri"].sum()
        est_pct = dispatch_plan["est_congestion_reduction_pct"].iloc[-1] if len(dispatch_plan) > 0 else 0
        lines.append(f"  Total PCRI Relief:           {total_relief:.1f}")
        lines.append(f"  Est. Congestion Reduction:   {est_pct:.1f}%")
        lines.append("")

    # ── Section 2: Impact Projection ──
    if impact is not None:
        lines.append("─" * 64)
        lines.append("  PROJECTED IMPACT (What-If Analysis)")
        lines.append("─" * 64)
        lines.append("")
        lines.append(f"  Network PCRI Before:    {impact['original_total_pcri']}")
        lines.append(f"  Network PCRI After:     {impact['simulated_total_pcri']}")
        lines.append(f"  Reduction:              {impact['pcri_reduction_pct']}%")
        lines.append(f"  High-Risk Before:       {impact['original_high_risk']} junctions")
        lines.append(f"  High-Risk After:        {impact['simulated_high_risk']} junctions")
        lines.append(f"  High-Risk Eliminated:   {impact['high_risk_eliminated']} junctions")
        lines.append("")

    # ── Section 3: Full Priority List ──
    lines.append("─" * 64)
    lines.append("  FULL JUNCTION PRIORITY LIST")
    lines.append("─" * 64)
    lines.append("")
    lines.append(f"  {'Rank':<5} {'Junction':<30} {'PCRI':>6} {'Prob%':>6} {'Priority':>8}  {'Zone':>10}")
    lines.append(f"  {'─'*5} {'─'*30} {'─'*6} {'─'*6} {'─'*8}  {'─'*10}")

    for _, row in recs.iterrows():
        rank = int(row.get("rank", 0))
        junc = str(row[group_col])[:30]
        pcri = row.get("pcri", 0)
        prob = row.get("hotspot_probability_pct", 0)
        priority = row.get("enforcement_priority", 0)
        zone = row.get("priority_zone", "—")
        lines.append(f"  {rank:<5} {junc:<30} {pcri:>6.1f} {prob:>6.1f} {priority:>8.1f}  {zone:>10}")

    lines.append("")
    lines.append("─" * 64)
    lines.append("  INSTRUCTIONS FOR FIELD OFFICERS")
    lines.append("─" * 64)
    lines.append("")
    lines.append("  1. Deploy units in dispatch order (#1 first, then #2, etc.)")
    lines.append("  2. CRITICAL zones: Immediate towing + challan issuance")
    lines.append("  3. ELEVATED zones: Patrol + warning, escalate if persistent")
    lines.append("  4. ROUTINE zones: Include in regular patrol cycle")
    lines.append("  5. Report clearance status back to control room via radio")
    lines.append("")
    lines.append("=" * 64)
    lines.append("  END OF BRIEFING — ParkIQ AI Parking Intelligence")
    lines.append("=" * 64)

    return "\n".join(lines)
