"""
explainability.py — SHAP-based model explainability.
Generates SHAP summary plots, feature importance, and top reasons.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def compute_shap_values(model, X_sample: pd.DataFrame, model_type: str = "xgboost"):
    """Compute SHAP values for the given model and sample data."""
    if model_type in ("xgboost", "random_forest"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X_sample.iloc[:100])

    shap_values = explainer.shap_values(X_sample)

    # For binary classification, pick positive class
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    return explainer, shap_values


def plot_shap_summary(shap_values, X_sample: pd.DataFrame, output_dir: str = "outputs"):
    """Generate and save SHAP summary beeswarm plot."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False, plot_size=(10, 6))
    path = os.path.join(output_dir, "shap_summary.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[INFO] SHAP summary plot saved to {path}")
    return path


def plot_feature_importance(shap_values, feature_names: list, output_dir: str = "outputs"):
    """Generate bar chart of mean |SHAP| values."""
    os.makedirs(output_dir, exist_ok=True)

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_abs,
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(importance)))
    ax.barh(importance["feature"], importance["importance"], color=colors)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (SHAP)")
    plt.tight_layout()

    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[INFO] Feature importance chart saved to {path}")
    return path, importance


def get_top_reasons(shap_values, feature_names: list, top_n: int = 5) -> list[dict]:
    """Get top reasons (features) driving hotspot predictions."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(mean_abs)[::-1][:top_n]

    reasons = []
    for idx in indices:
        reasons.append({
            "feature": feature_names[idx],
            "mean_shap": round(float(mean_abs[idx]), 4),
            "direction": "increases" if np.mean(shap_values[:, idx]) > 0 else "decreases",
        })
    return reasons
