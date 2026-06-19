import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(PROJECT_ROOT, "models")
data_dir = os.path.join(PROJECT_ROOT, "data")

parquet_path = os.path.join(data_dir, "processed_data.parquet")
df = pd.read_parquet(parquet_path)

encoders = joblib.load(os.path.join(models_dir, "label_encoders.pkl"))

from src.prediction import prepare_features
X, y, _ = prepare_features(df, encoders=encoders)

np.random.seed(42)
indices = np.random.choice(len(df), min(10000, len(df)), replace=False)
X_sample = X.iloc[indices]
y_sample = y.iloc[indices]

for mname in ["xgboost", "random_forest"]:
    mpath = os.path.join(models_dir, f"{mname}_model.pkl")
    if os.path.exists(mpath):
        m = joblib.load(mpath)
        y_pred = m.predict(X_sample)
        y_prob = m.predict_proba(X_sample)[:, 1]
        print(f"Model: {mname}")
        print(f"  Accuracy:  {accuracy_score(y_sample, y_pred):.4f}")
        print(f"  Precision: {precision_score(y_sample, y_pred, zero_division=0):.4f}")
        print(f"  Recall:    {recall_score(y_sample, y_pred, zero_division=0):.4f}")
        print(f"  F1 Score:  {f1_score(y_sample, y_pred, zero_division=0):.4f}")
        print(f"  ROC-AUC:   {roc_auc_score(y_sample, y_prob):.4f}")
