"""
train.py - Training pipeline for the AI Parking Intelligence Platform.
Runs preprocessing, hotspot detection, PCRI computation, model training,
SHAP analysis, and forecasting. Saves all artifacts.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import run_preprocessing
from src.hotspot_detection import run_hotspot_detection
from src.pcri import compute_pcri
from src.prediction import prepare_features, train_models, save_models
from src.explainability import compute_shap_values, plot_shap_summary, plot_feature_importance, get_top_reasons
from src.forecasting import (
    prepare_prophet_data, run_prophet_forecast,
    detect_anomalies, get_anomaly_summary
)

import pandas as pd
import joblib


def main():
    print("=" * 60)
    print("  AI Parking Intelligence Platform - Training Pipeline")
    print("=" * 60)

    # Step 1: Preprocessing
    print("\n>> Step 1: Data Preprocessing")
    df, summary = run_preprocessing(project_root=PROJECT_ROOT)
    print(f"  Total records after cleaning: {len(df):,}")

    # Step 2: Hotspot Detection
    print("\n>> Step 2: DBSCAN Hotspot Detection")
    df, hotspot_stats = run_hotspot_detection(df, eps_km=0.3, min_samples=15)
    print(f"  Hotspot clusters found: {len(hotspot_stats)}")

    # Step 3: PCRI
    print("\n>> Step 3: Computing PCRI Scores")
    junction_pcri, area_pcri = compute_pcri(df)
    print(f"  Junctions scored: {len(junction_pcri)}")
    print(f"  Areas scored: {len(area_pcri)}")

    # Step 4: Model Training
    print("\n>> Step 4: Training Prediction Models")
    X, y, encoders = prepare_features(df)
    results = train_models(X, y)

    for name in ["xgboost", "random_forest"]:
        r = results[name]
        print(f"\n  {name.upper()}")
        print(f"    Accuracy:  {r['accuracy']:.4f}")
        print(f"    Precision: {r['precision']:.4f}")
        print(f"    Recall:    {r['recall']:.4f}")
        print(f"    F1 Score:  {r['f1']:.4f}")
        print(f"    ROC-AUC:   {r['roc_auc']:.4f}")

    print(f"\n  * Best Model: {results['best_model_name']}")

    save_models(results, encoders, output_dir=os.path.join(PROJECT_ROOT, "models"))

    # Step 5: SHAP Explainability
    print("\n>> Step 5: SHAP Explainability Analysis")
    X_sample = results["X_test"].sample(min(1000, len(results["X_test"])), random_state=42)
    best_model = results["best_model"]
    best_name = results["best_model_name"]

    explainer, shap_values = compute_shap_values(best_model, X_sample, model_type=best_name)
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    plot_shap_summary(shap_values, X_sample, output_dir=output_dir)
    _, importance_df = plot_feature_importance(shap_values, results["feature_names"], output_dir=output_dir)
    top_reasons = get_top_reasons(shap_values, results["feature_names"])
    print("  Top reasons for hotspot formation:")
    for r in top_reasons:
        print(f"    - {r['feature']}: {r['direction']} risk (SHAP={r['mean_shap']})")

    # Step 6: Anomaly Detection
    print("\n>> Step 6: Anomaly Detection (Isolation Forest)")
    df = detect_anomalies(df, contamination=0.05)
    anomaly_summary = get_anomaly_summary(df)
    print(f"  Anomalies detected: {anomaly_summary.get('total_anomalies', 0)}")

    # Step 7: Save processed data
    print("\n>> Step 7: Saving Processed Data")
    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)

    # Ensure mixed-type columns are strings for parquet compatibility
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str)
    df.to_parquet(os.path.join(PROJECT_ROOT, "data", "processed_data.parquet"), index=False)
    hotspot_stats.to_csv(os.path.join(PROJECT_ROOT, "data", "hotspot_stats.csv"), index=False)
    junction_pcri.to_csv(os.path.join(PROJECT_ROOT, "data", "junction_pcri.csv"), index=False)
    area_pcri.to_csv(os.path.join(PROJECT_ROOT, "data", "area_pcri.csv"), index=False)

    # Save summary
    joblib.dump(summary, os.path.join(PROJECT_ROOT, "data", "summary.pkl"))
    joblib.dump(anomaly_summary, os.path.join(PROJECT_ROOT, "data", "anomaly_summary.pkl"))
    joblib.dump(top_reasons, os.path.join(PROJECT_ROOT, "data", "shap_reasons.pkl"))

    # Save feature names and encoders reference
    joblib.dump(results["feature_names"], os.path.join(PROJECT_ROOT, "data", "feature_names.pkl"))

    print("\n" + "=" * 60)
    print("  [OK] Training Pipeline Complete!")
    print("  Run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
