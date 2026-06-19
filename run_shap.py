import os
import sys
import pandas as pd
import joblib

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.explainability import compute_shap_values, plot_shap_summary, plot_feature_importance, get_top_reasons
from src.prediction import prepare_features
from src.forecasting import detect_anomalies, get_anomaly_summary

def main():
    print("Recovering from SHAP failure...")
    # Load data
    df = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "processed_data.parquet"))
    hotspot_stats = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "hotspot_stats.csv"))
    
    # We need X_test for SHAP. We can just use prepare_features and then sample.
    X, y, encoders = prepare_features(df)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    best_name = "random_forest"
    best_model = joblib.load(os.path.join(PROJECT_ROOT, "models", f"{best_name}_model.pkl"))
    feature_names = list(X.columns)
    
    print("\n>> Step 5: SHAP Explainability Analysis")
    X_sample = X_test.sample(min(150, len(X_test)), random_state=42)
    
    explainer, shap_values = compute_shap_values(best_model, X_sample, model_type=best_name)
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    plot_shap_summary(shap_values, X_sample, output_dir=output_dir)
    _, importance_df = plot_feature_importance(shap_values, feature_names, output_dir=output_dir)
    top_reasons = get_top_reasons(shap_values, feature_names)
    print("  Top reasons for hotspot formation:")
    for r in top_reasons:
        print(f"    - {r['feature']}: {r['direction']} risk (SHAP={r['mean_shap']})")
        
    print("\n>> Step 6: Anomaly Detection (Isolation Forest)")
    df = detect_anomalies(df, contamination=0.05)
    anomaly_summary = get_anomaly_summary(df)
    print(f"  Anomalies detected: {anomaly_summary.get('total_anomalies', 0)}")
    
    print("\n>> Step 7: Saving Processed Data")
    df.to_parquet(os.path.join(PROJECT_ROOT, "data", "processed_data.parquet"), index=False)
    
    joblib.dump(anomaly_summary, os.path.join(PROJECT_ROOT, "data", "anomaly_summary.pkl"))
    joblib.dump(top_reasons, os.path.join(PROJECT_ROOT, "data", "shap_reasons.pkl"))
    joblib.dump(feature_names, os.path.join(PROJECT_ROOT, "data", "feature_names.pkl"))
    
    print("Recovery complete!")

if __name__ == "__main__":
    main()
