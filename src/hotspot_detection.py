"""
hotspot_detection.py — DBSCAN-based illegal parking hotspot detection.
Clusters violations by geo-coordinates and assigns severity levels.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def detect_hotspots(
    df: pd.DataFrame,
    eps_km: float = 0.3,
    min_samples: int = 15,
    max_samples: int = 40000,
) -> pd.DataFrame:
    """
    Run DBSCAN on lat/lon to find violation hotspot clusters.
    Samples data if too large to avoid memory errors, then assigns
    all records to nearest cluster center.

    Parameters
    ----------
    df : DataFrame with 'latitude' and 'longitude' columns.
    eps_km : Neighborhood radius in kilometres (~0.3 km default).
    min_samples : Minimum points to form a cluster.
    max_samples : Maximum sample size for DBSCAN fitting.

    Returns
    -------
    DataFrame with 'cluster_id' column added.
    """
    coords = df[["latitude", "longitude"]].values
    n = len(coords)

    # Convert eps from km to radians for haversine
    eps_rad = eps_km / 6371.0

    if n > max_samples:
        # Sample for DBSCAN fitting
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n, size=max_samples, replace=False)
        sample_coords = coords[sample_idx]
    else:
        sample_idx = np.arange(n)
        sample_coords = coords

    db = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    )
    sample_rad = np.radians(sample_coords)
    sample_labels = db.fit_predict(sample_rad)

    if n > max_samples:
        # Compute cluster centers from the sample
        from scipy.spatial import cKDTree

        unique_labels = set(sample_labels)
        unique_labels.discard(-1)

        if len(unique_labels) == 0:
            df = df.copy()
            df["cluster_id"] = -1
            return df

        centers = []
        center_labels = []
        for lbl in sorted(unique_labels):
            mask = sample_labels == lbl
            center = sample_coords[mask].mean(axis=0)
            centers.append(center)
            center_labels.append(lbl)
        centers = np.array(centers)

        # Assign all points to nearest cluster center (within eps_km)
        tree = cKDTree(centers)
        dists, indices = tree.query(coords, k=1)

        labels = np.full(n, -1, dtype=int)
        for i in range(n):
            if dists[i] < eps_km / 111.0:  # rough degree conversion
                labels[i] = center_labels[indices[i]]
    else:
        labels = sample_labels

    df = df.copy()
    df["cluster_id"] = labels

    return df


def compute_cluster_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-cluster statistics."""
    clustered = df[df["cluster_id"] != -1].copy()

    if len(clustered) == 0:
        return pd.DataFrame()

    stats = clustered.groupby("cluster_id").agg(
        violation_count=("id", "count"),
        center_lat=("latitude", "mean"),
        center_lon=("longitude", "mean"),
        unique_vehicles=("vehicle_number", "nunique"),
        lat_std=("latitude", "std"),
        lon_std=("longitude", "std"),
    ).reset_index()

    # Junctions in the cluster
    if "junction_name" in clustered.columns:
        junc_agg = (
            clustered[clustered["junction_name"] != "No Junction"]
            .groupby("cluster_id")["junction_name"]
            .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else "Unknown")
        )
        stats = stats.merge(
            junc_agg.rename("primary_junction"),
            on="cluster_id", how="left"
        )
        stats["primary_junction"] = stats["primary_junction"].fillna("Unknown")

    # Location names
    if "location" in clustered.columns:
        loc_agg = (
            clustered.groupby("cluster_id")["location"]
            .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else "Unknown")
        )
        stats = stats.merge(
            loc_agg.rename("primary_location"),
            on="cluster_id", how="left"
        )

    return stats


def assign_severity(stats: pd.DataFrame) -> pd.DataFrame:
    """Assign Low / Medium / High severity based on violation density percentiles."""
    if len(stats) == 0:
        return stats

    q33 = stats["violation_count"].quantile(0.33)
    q66 = stats["violation_count"].quantile(0.66)

    def _severity(count):
        if count <= q33:
            return "Low"
        elif count <= q66:
            return "Medium"
        else:
            return "High"

    stats["severity"] = stats["violation_count"].apply(_severity)

    # Colour mapping
    colour_map = {"Low": "green", "Medium": "orange", "High": "red"}
    stats["colour"] = stats["severity"].map(colour_map)

    # Severity score (0-100)
    max_v = stats["violation_count"].max()
    if max_v > 0:
        stats["severity_score"] = (stats["violation_count"] / max_v * 100).round(1)
    else:
        stats["severity_score"] = 0

    return stats


def run_hotspot_detection(df: pd.DataFrame, eps_km=0.3, min_samples=15):
    """Full hotspot detection pipeline."""
    df = detect_hotspots(df, eps_km=eps_km, min_samples=min_samples)
    stats = compute_cluster_stats(df)
    stats = assign_severity(stats)

    n_clusters = df["cluster_id"].nunique() - (1 if -1 in df["cluster_id"].values else 0)
    noise = (df["cluster_id"] == -1).sum()
    print(f"[INFO] DBSCAN found {n_clusters} hotspot clusters, {noise:,} noise points.")

    return df, stats
