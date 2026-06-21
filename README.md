# ParkIQ - AI Parking Intelligence Platform

> **Flipkart Gridlock 2026 - Stage 2 Submission**
>
> *How can AI-driven parking intelligence detect illegal parking hotspots and quantify their impact on traffic flow to enable targeted enforcement?*

---

## Architecture

```
project/
│
├── data/                        # Processed data & artifacts
├── models/                      # Trained ML models
├── outputs/                     # SHAP plots & reports
├── src/
│   ├── preprocessing.py         # Data ingestion & cleaning
│   ├── hotspot_detection.py     # DBSCAN hotspot clustering
│   ├── pcri.py                  # Parking Congestion Risk Index
│   ├── prediction.py            # XGBoost + Random Forest models
│   ├── explainability.py        # SHAP explanations
│   ├── recommendation.py        # Enforcement recommendation engine
│   └── forecasting.py           # Prophet + Isolation Forest
│
├── app.py                       # Streamlit dashboard
├── train.py                     # End-to-end training pipeline
├── requirements.txt             # Dependencies
└── README.md
```

## Quick Start

### 1. Place the Dataset
Place the CSV dataset in the project root folder. Ensure the raw violation file is named `jan to may police violation_anonymized791b166.csv` or similar.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Models
Run the training script to run the end-to-end preprocessing, spatial clustering, risk scoring, XGBoost modeling, explainability, and anomaly detection pipelines:
```bash
python train.py
```

### 4. Launch Dashboard
```bash
streamlit run app.py
```

---

## Core Capabilities

### 1. DBSCAN Hotspot Detection
- Clusters parking violations by geographic coordinates using haversine distance.
- Identifies high-density illegal parking zones.
- Severity classification: Low, Medium, High.

### 2. Parking Congestion Risk Index (PCRI)
Custom 0-100 composite score:
| Component | Weight | Description |
|---|---|---|
| Violation Density | 35% | Total violations per zone |
| Peak Hour Weight | 25% | Fraction during traffic peaks |
| Junction Frequency | 25% | Days with violations |
| Repeat Violations | 15% | Recurring offender rate |

### 3. ML Prediction Engine
Predicts: *"Will a hotspot form at this junction in the next hour?"*

**Models:** XGBoost, Random Forest (auto-selects best by F1 metric)

**Features:** hour, weekday, vehicle_type, junction_name, historical_violation_count, cluster_id

### 4. SHAP Explainability
- Feature importance rankings.
- SHAP beeswarm plots.
- Top reasons for hotspot formation.

### 5. Enforcement Recommendation Engine
- Combines PCRI + prediction probability.
- Ranks all junctions by enforcement priority.
- Generates patrol deployment recommendations.
- Critical, Elevated, and Routine zone classification.

### 6. Resource-Constrained Dispatch Optimizer
- Greedy knapsack-style algorithm for limited patrol/tow units.
- Selects optimal junction combination to maximise congestion relief.
- Cumulative PCRI relief tracking with estimated congestion reduction percentage.
- Visual deployment route with numbered dispatch order.

### 7. What-If Impact Simulator
- Interactive junction selection for simulated enforcement clearance.
- Before vs. After network-wide PCRI comparison.
- Projected high-risk zone elimination metrics.
- Grouped bar chart visualization of simulated impact.

### 8. Export Dispatch Briefing
- One-click downloadable text dispatch briefing for control room operators.
- Includes optimised dispatch plan, projected impact, and full priority list.
- Field officer instructions for immediate operational deployment.

### 9. Advanced Analytics
- **Temporal Analysis:** Peak hours, days, monthly trends.
- **Forecasting:** Prophet-based 30-day violation forecast.
- **Anomaly Detection:** Isolation Forest for unusual activity.
- **Cross-analysis:** Vehicle type x hour heatmaps.

---

## Dashboard Pages

| Page | Description |
|---|---|
| **Overview** | KPIs, trends, severity breakdown, SHAP insights |
| **Heatmap** | Interactive Folium maps with hotspot clusters & PCRI overlay |
| **Prediction** | Live hotspot prediction + model performance comparison |
| **Enforcement** | Ranked priorities, dispatch optimizer, what-if simulator, briefing export |
| **Analytics** | Deep temporal, vehicle, station & anomaly analysis |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| ML | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualization | Plotly, Folium |
| Forecasting | Prophet |
| Data | Pandas, NumPy |

---

## License

Built for Flipkart Gridlock 2026 Hackathon.
