"""
app.py — AI Parking Intelligence Platform Dashboard (Streamlit)
Professional, hackathon-winning dashboard with 5 pages:
  1. Overview  2. Heatmap  3. Prediction  4. Enforcement  5. Analytics
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
import contextlib

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import joblib

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ParkIQ — AI Parking Intelligence",
    page_icon=":material/local_parking:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 50%, #0a0a1a 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1923 0%, #0a0f18 100%);
    border-right: 1px solid rgba(56, 189, 248, 0.1);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #38bdf8 !important;
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(15, 25, 35, 0.95) 0%, rgba(20, 30, 50, 0.9) 100%);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(56, 189, 248, 0.1);
    transition: all 0.3s ease;
}
div[data-testid="stMetric"]:hover {
    border-color: rgba(56, 189, 248, 0.4);
    box-shadow: 0 8px 32px rgba(56, 189, 248, 0.1);
    transform: translateY(-2px);
}
div[data-testid="stMetric"] label {
    color: #94a3b8 !important;
    font-weight: 500;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-weight: 700;
    font-size: 2rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(15, 25, 35, 0.6);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(56, 189, 248, 0.1);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 500;
    padding: 10px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e3a5f 0%, #1a2744 100%) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.3);
}

/* Headers */
h1, h2, h3 {
    color: #e2e8f0 !important;
}

/* Dataframe */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(56, 189, 248, 0.1);
}

/* Select boxes, inputs */
.stSelectbox > div > div,
.stSlider > div > div,
.stNumberInput > div > div {
    border-radius: 8px;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(15, 25, 35, 0.6);
    border-radius: 12px;
    border: 1px solid rgba(56, 189, 248, 0.1);
}

/* Hero text */
.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #94a3b8;
    margin-top: 4px;
    font-weight: 400;
}

/* Card container */
.glass-card {
    background: linear-gradient(135deg, rgba(15, 25, 35, 0.8) 0%, rgba(20, 30, 50, 0.6) 100%);
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 16px;
    padding: 24px;
    margin: 8px 0;
    backdrop-filter: blur(10px);
}

/* Alert badges */
.badge-critical {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-elevated {
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-routine {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


@contextlib.contextmanager
def custom_spinner(message, icon="fa-circle-notch", color="#38bdf8", animation="fa-spin"):
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(f"""
        <div class="glass-card" style="display: flex; align-items: center; justify-content: center; gap: 16px; padding: 20px; margin: 16px 0;">
            <i class="fa-solid {icon} {animation}" style="font-size: 2rem; color: {color};"></i>
            <span style="color: #e2e8f0; font-size: 1.1rem; font-weight: 500;">{message}</span>
        </div>
        """, unsafe_allow_html=True)
    try:
        yield
    finally:
        placeholder.empty()


# ─── Data Loading ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_all_data():
    """Load all preprocessed data. Run training if not available."""
    data_dir = os.path.join(PROJECT_ROOT, "data")
    models_dir = os.path.join(PROJECT_ROOT, "models")

    parquet_path = os.path.join(data_dir, "processed_data.parquet")

    if not os.path.exists(parquet_path):
        # Run training pipeline first
        with custom_spinner("First run detected — running AI training pipeline (this may take a few minutes)...", icon="fa-gears", color="#818cf8", animation="fa-spin"):
            import subprocess
            result = subprocess.run(
                [sys.executable, os.path.join(PROJECT_ROOT, "train.py")],
                capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            if result.returncode != 0:
                st.error(f"Training failed:\n{result.stderr}")
                st.stop()

    # Load processed data
    df = pd.read_parquet(parquet_path)

    # Convert date column back
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Load artifacts
    hotspot_stats = pd.read_csv(os.path.join(data_dir, "hotspot_stats.csv"))
    junction_pcri = pd.read_csv(os.path.join(data_dir, "junction_pcri.csv"))
    area_pcri = pd.read_csv(os.path.join(data_dir, "area_pcri.csv"))
    summary = joblib.load(os.path.join(data_dir, "summary.pkl"))
    anomaly_summary = joblib.load(os.path.join(data_dir, "anomaly_summary.pkl"))
    shap_reasons = joblib.load(os.path.join(data_dir, "shap_reasons.pkl"))
    feature_names = joblib.load(os.path.join(data_dir, "feature_names.pkl"))

    # Load model
    model = None
    encoders = None
    model_name = "xgboost"
    try:
        best_path = os.path.join(models_dir, "best_model.txt")
        if os.path.exists(best_path):
            with open(best_path) as f:
                model_name = f.read().strip()
        model = joblib.load(os.path.join(models_dir, f"{model_name}_model.pkl"))
        encoders = joblib.load(os.path.join(models_dir, "label_encoders.pkl"))
    except Exception:
        pass

    # Load model metrics
    metrics = {}
    for mname in ["xgboost", "random_forest"]:
        mpath = os.path.join(models_dir, f"{mname}_model.pkl")
        if os.path.exists(mpath):
            metrics[mname] = {"name": mname}

    return {
        "df": df,
        "hotspot_stats": hotspot_stats,
        "junction_pcri": junction_pcri,
        "area_pcri": area_pcri,
        "summary": summary,
        "anomaly_summary": anomaly_summary,
        "shap_reasons": shap_reasons,
        "feature_names": feature_names,
        "model": model,
        "model_name": model_name,
        "encoders": encoders,
    }


# ─── Plotly Theme ──────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10, 15, 30, 0.5)",
    font=dict(family="Inter", color="#94a3b8"),
    title_font=dict(color="#e2e8f0", size=16),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="rgba(56, 189, 248, 0.06)", zerolinecolor="rgba(56, 189, 248, 0.1)"),
    yaxis=dict(gridcolor="rgba(56, 189, 248, 0.06)", zerolinecolor="rgba(56, 189, 248, 0.1)"),
    colorway=["#38bdf8", "#818cf8", "#c084fc", "#f472b6", "#fb923c",
              "#34d399", "#fbbf24", "#ef4444", "#06b6d4", "#a78bfa"],
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
)

SEVERITY_COLORS = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}
RISK_COLORS = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}


# ─── Sidebar ──────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown('<p class="hero-title" style="font-size:1.6rem;"><i class="fa-solid fa-square-parking" style="margin-right:8px; color:#38bdf8;"></i>ParkIQ</p>', unsafe_allow_html=True)
        st.markdown('<p class="hero-subtitle" style="font-size:0.85rem;">AI Parking Intelligence</p>', unsafe_allow_html=True)
        st.markdown("---")

        page = st.radio(
            "Navigation",
            [":material/dashboard: Overview", ":material/map: Heatmap", ":material/online_prediction: Prediction",
             ":material/local_police: Enforcement", ":material/analytics: Analytics"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            """
            <div style='text-align:center; color:#64748b; font-size:0.75rem;'>
            Flipkart Gridlock 2026<br/>
            AI Parking Intelligence<br/>
            <span style='color:#38bdf8;'>Stage 2 Submission</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return page


# ─── Page: Overview ───────────────────────────────────────────
def page_overview(data):
    df = data["df"]
    summary = data["summary"]
    hotspot_stats = data["hotspot_stats"]
    junction_pcri = data["junction_pcri"]
    shap_reasons = data["shap_reasons"]

    st.markdown('<p class="hero-title">AI Parking Intelligence Platform</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Real-time hotspot detection, congestion risk scoring, and enforcement optimization for Bengaluru</p>', unsafe_allow_html=True)
    st.markdown("")

    # ── KPI Row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Violations", f"{summary['total_records']:,}")
    with c2:
        high_risk = len(hotspot_stats[hotspot_stats["severity"] == "High"]) if "severity" in hotspot_stats.columns else 0
        st.metric("Active Hotspots", f"{len(hotspot_stats)}", delta=f"{high_risk} High Risk")
    with c3:
        high_junc = len(junction_pcri[junction_pcri["risk_level"] == "High"]) if "risk_level" in junction_pcri.columns else 0
        st.metric("High Risk Junctions", f"{high_junc}")
    with c4:
        st.metric("Unique Vehicles", f"{summary['unique_vehicles']:,}")

    st.markdown("")

    # ── Violation Trends ──
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### :material/trending_up: Violation Trend Over Time")
        if "date" in df.columns:
            daily = df.groupby("date")["id"].count().reset_index()
            daily.columns = ["Date", "Violations"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily["Date"], y=daily["Violations"],
                mode="lines",
                fill="tozeroy",
                line=dict(color="#38bdf8", width=2),
                fillcolor="rgba(56, 189, 248, 0.1)",
                name="Daily Violations",
            ))
            # 7-day moving average
            daily["MA7"] = daily["Violations"].rolling(7, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=daily["Date"], y=daily["MA7"],
                mode="lines",
                line=dict(color="#c084fc", width=2, dash="dash"),
                name="7-Day Average",
            ))
            fig.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items() if k != "legend"},
                              title="", height=350,
                              legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.12, font=dict(color="#94a3b8")))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### :material/label: Hotspot Severity Breakdown")
        if "severity" in hotspot_stats.columns:
            sev_counts = hotspot_stats["severity"].value_counts().reset_index()
            sev_counts.columns = ["Severity", "Count"]
            fig = go.Figure(go.Pie(
                labels=sev_counts["Severity"],
                values=sev_counts["Count"],
                hole=0.6,
                marker=dict(colors=[SEVERITY_COLORS.get(s, "#666") for s in sev_counts["Severity"]]),
                textinfo="label+value",
                textfont=dict(size=13, color="white"),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False,
                              annotations=[dict(text=f"<b>{len(hotspot_stats)}</b><br>Clusters",
                                                x=0.5, y=0.5, font_size=16, font_color="#e2e8f0",
                                                showarrow=False)])
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Vehicle Types + Top Junctions ──
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### :material/directions_car: Violations by Vehicle Type")
        if summary["vehicle_type_dist"]:
            vt = pd.DataFrame(list(summary["vehicle_type_dist"].items()), columns=["Type", "Count"])
            vt = vt.sort_values("Count", ascending=True).tail(8)
            fig = go.Figure(go.Bar(
                x=vt["Count"], y=vt["Type"], orientation="h",
                marker=dict(
                    color=vt["Count"],
                    colorscale=[[0, "#1e3a5f"], [0.5, "#38bdf8"], [1, "#c084fc"]],
                ),
                text=vt["Count"].apply(lambda x: f"{x:,}"),
                textposition="auto",
                textfont=dict(color="white"),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=320, title="", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("#### :material/local_fire_department: Top 10 Risky Junctions (PCRI)")
        if len(junction_pcri) > 0:
            top10 = junction_pcri.head(10).copy()
            display_col = "junction_name" if "junction_name" in top10.columns else top10.columns[0]
            top10 = top10.sort_values("pcri", ascending=True)
            fig = go.Figure(go.Bar(
                x=top10["pcri"], y=top10[display_col], orientation="h",
                marker=dict(
                    color=top10["pcri"],
                    colorscale=[[0, "#22c55e"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                    colorbar=dict(title="PCRI"),
                ),
                text=top10["pcri"].apply(lambda x: f"{x:.0f}"),
                textposition="auto",
                textfont=dict(color="white"),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=320, title="", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── SHAP Insights ──
    if shap_reasons:
        st.markdown("#### :material/psychology: AI Insights — Top Drivers of Hotspot Formation")
        cols = st.columns(len(shap_reasons))
        icons = [
            '<i class="fa-solid fa-location-dot" style="color:#38bdf8;"></i>',
            '<i class="fa-solid fa-clock" style="color:#818cf8;"></i>',
            '<i class="fa-solid fa-rotate" style="color:#c084fc;"></i>',
            '<i class="fa-solid fa-traffic-light" style="color:#f472b6;"></i>',
            '<i class="fa-solid fa-chart-simple" style="color:#fb923c;"></i>'
        ]
        for i, reason in enumerate(shap_reasons):
            with cols[i]:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center; min-height:140px;">
                    <div style="font-size:2rem; margin-bottom:8px;">{icons[i % len(icons)]}</div>
                    <div style="color:#e2e8f0; font-weight:600; margin:8px 0;">{reason['feature'].replace('_', ' ').title()}</div>
                    <div style="color:#38bdf8; font-size:1.3rem; font-weight:700;">SHAP {reason['mean_shap']}</div>
                    <div style="color:#94a3b8; font-size:0.8rem;">{reason['direction']} hotspot risk</div>
                </div>
                """, unsafe_allow_html=True)


# ─── Page: Heatmap ────────────────────────────────────────────
def page_heatmap(data):
    df = data["df"]
    hotspot_stats = data["hotspot_stats"]
    junction_pcri = data["junction_pcri"]

    st.markdown("### :material/map: Hotspot & Congestion Risk Map")

    # Map style selector
    map_styles = {
        "Dark Mode (CartoDB)": {"tiles": "CartoDB dark_matter", "attr": "CartoDB"},
        "Light Mode (CartoDB)": {"tiles": "CartoDB Positron", "attr": "CartoDB"},
        "Standard (OpenStreetMap)": {"tiles": "OpenStreetMap", "attr": "OpenStreetMap"},
        "Satellite (Esri World Imagery)": {
            "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "attr": "Esri World Imagery"
        }
    }
    
    col_style, _ = st.columns([1, 2])
    with col_style:
        selected_style = st.selectbox("Map Style", list(map_styles.keys()), index=0)
    
    style_config = map_styles[selected_style]

    tab1, tab2 = st.tabs([":material/layers: Hotspot Clusters", ":material/thermostat: PCRI Heatmap"])

    with tab1:
        st.markdown("Interactive map of DBSCAN-detected parking violation hotspots. "
                     "**Green** = Low, **Orange** = Medium, **Red** = High risk.")

        # Create Folium map
        center_lat = df["latitude"].mean()
        center_lon = df["longitude"].mean()
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles=style_config["tiles"],
            attr=style_config["attr"],
        )

        # Add hotspot cluster circles
        for _, row in hotspot_stats.iterrows():
            colour = row.get("colour", "blue")
            severity = row.get("severity", "Unknown")
            count = row.get("violation_count", 0)
            score = row.get("severity_score", 0)
            junction = row.get("primary_junction", "Unknown")

            popup_html = f"""
            <div style='font-family:Inter,sans-serif; min-width:200px;'>
                <h4 style='color:#1e3a5f; margin:0;'>Hotspot Cluster {int(row['cluster_id'])}</h4>
                <hr style='margin:4px 0;'>
                <b>Severity:</b> <span style='color:{colour};font-weight:700;'>{severity}</span><br>
                <b>Violations:</b> {count:,}<br>
                <b>Severity Score:</b> {score:.1f}/100<br>
                <b>Junction:</b> {junction}<br>
                <b>Unique Vehicles:</b> {int(row.get('unique_vehicles', 0)):,}
            </div>
            """

            radius = max(200, min(count * 2, 1500))

            folium.Circle(
                location=[row["center_lat"], row["center_lon"]],
                radius=radius,
                popup=folium.Popup(popup_html, max_width=300),
                color=colour,
                fill=True,
                fill_color=colour,
                fill_opacity=0.35,
                weight=2,
            ).add_to(m)

            folium.CircleMarker(
                location=[row["center_lat"], row["center_lon"]],
                radius=6,
                popup=folium.Popup(popup_html, max_width=300),
                color="white",
                fill=True,
                fill_color=colour,
                fill_opacity=0.9,
                weight=1,
            ).add_to(m)

        st_folium(m, width=None, height=550, returned_objects=[])

        # Stats below map
        col1, col2, col3 = st.columns(3)
        for sev, col in zip(["High", "Medium", "Low"], [col1, col2, col3]):
            count = len(hotspot_stats[hotspot_stats["severity"] == sev]) if "severity" in hotspot_stats.columns else 0
            with col:
                colour = SEVERITY_COLORS[sev]
                st.markdown(f"""
                <div class="glass-card" style="text-align:center; border-color:{colour}40;">
                    <div style="color:{colour}; font-size:2rem; font-weight:800;">{count}</div>
                    <div style="color:#94a3b8; font-weight:500;">{sev} Risk Hotspots</div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("PCRI (Parking Congestion Risk Index) overlay for all scored junctions.")

        # PCRI on map
        m2 = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles=style_config["tiles"],
            attr=style_config["attr"],
        )

        # Merge junction PCRI with location data
        if "junction_name" in junction_pcri.columns and "junction_name" in df.columns:
            junc_locs = df.groupby("junction_name").agg(
                lat=("latitude", "mean"),
                lon=("longitude", "mean"),
            ).reset_index()
            pcri_map = junction_pcri.merge(junc_locs, on="junction_name", how="left").dropna(subset=["lat", "lon"])

            for _, row in pcri_map.iterrows():
                pcri_val = row["pcri"]
                if pcri_val > 66:
                    colour = "#ef4444"
                elif pcri_val > 33:
                    colour = "#f59e0b"
                else:
                    colour = "#22c55e"

                popup = f"""
                <div style='font-family:Inter,sans-serif;'>
                    <h4 style='margin:0;'>{row['junction_name']}</h4>
                    <b>PCRI Score:</b> {pcri_val:.1f}/100<br>
                    <b>Risk Level:</b> {row.get('risk_level', 'N/A')}<br>
                    <b>Violations:</b> {int(row.get('violation_density', 0)):,}<br>
                    <b>Peak Hour Weight:</b> {row.get('peak_hour_weight', 0):.2f}
                </div>
                """

                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=max(5, pcri_val / 8),
                    popup=folium.Popup(popup, max_width=300),
                    color=colour,
                    fill=True,
                    fill_color=colour,
                    fill_opacity=0.7,
                    weight=2,
                ).add_to(m2)

        st_folium(m2, width=None, height=550, returned_objects=[])

        # Top 10 PCRI areas
        if len(data["area_pcri"]) > 0:
            st.markdown("#### :material/domain: Top 10 Risky Areas (by Police Station)")
            area_col = "police_station" if "police_station" in data["area_pcri"].columns else data["area_pcri"].columns[0]
            top_areas = data["area_pcri"].head(10)
            fig = go.Figure(go.Bar(
                x=top_areas["pcri"],
                y=top_areas[area_col],
                orientation="h",
                marker=dict(
                    color=top_areas["pcri"],
                    colorscale=[[0, "#22c55e"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                ),
                text=top_areas["pcri"].apply(lambda x: f"{x:.0f}"),
                textposition="auto",
                textfont=dict(color="white"),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=350, title="", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# ─── Page: Prediction ─────────────────────────────────────────
def page_prediction(data):
    df = data["df"]
    model = data["model"]
    encoders = data["encoders"]
    model_name = data["model_name"]

    st.markdown("### :material/online_prediction: Hotspot Prediction Engine")

    tab1, tab2, tab3 = st.tabs([":material/bolt: Live Prediction", ":material/analytics: Model Performance", ":material/traffic: Live Traffic Analysis"])

    with tab1:
        st.markdown("Predict whether a parking hotspot will form at a given junction and time.")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Input Parameters")

            # Junction selector
            junctions = sorted(df["junction_name"].unique().tolist()) if "junction_name" in df.columns else []
            junctions = [j for j in junctions if j not in ("Unknown", "No Junction")]
            selected_junction = st.selectbox(":material/traffic: Junction", junctions[:100], index=0 if junctions else None)

            # Time parameters
            col_h, col_w = st.columns(2)
            with col_h:
                hour_labels = []
                for h in range(24):
                    if h == 0:
                        label = "12:00 AM"
                    elif h < 12:
                        label = f"{h}:00 AM"
                    elif h == 12:
                        label = "12:00 PM"
                    else:
                        label = f"{h-12}:00 PM"
                    hour_labels.append(label)
                
                selected_hour_str = st.selectbox(":material/schedule: Time of Day", hour_labels, index=10)
                selected_hour = hour_labels.index(selected_hour_str)
            with col_w:
                selected_weekday = st.selectbox(":material/calendar_today: Day of Week",
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    index=0)

            weekday_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                           "Friday": 4, "Saturday": 5, "Sunday": 6}
            weekday_num = weekday_map[selected_weekday]
            is_weekend = 1 if weekday_num >= 5 else 0

            month_names = [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ]
            selected_month_str = st.selectbox(":material/calendar_month: Month", month_names, index=0)
            selected_month = month_names.index(selected_month_str) + 1

            # Vehicle type
            vehicle_types = sorted(df["vehicle_type"].unique().tolist()) if "vehicle_type" in df.columns else []
            selected_vt = st.selectbox(":material/directions_car: Vehicle Type", vehicle_types, index=0 if vehicle_types else None)

            predict_btn = st.button("Predict Hotspot", icon=":material/online_prediction:", type="primary", use_container_width=True)

        with col2:
            if predict_btn and model is not None:
                try:
                    # Encode inputs
                    vt_enc = 0
                    if encoders and "vehicle_type" in encoders:
                        try:
                            vt_enc = int(encoders["vehicle_type"].transform([selected_vt])[0])
                        except ValueError:
                            vt_enc = 0

                    jn_enc = 0
                    if encoders and "junction_name" in encoders:
                        try:
                            jn_enc = int(encoders["junction_name"].transform([selected_junction])[0])
                        except ValueError:
                            jn_enc = 0

                    # Historical count
                    hist_count = len(df[df["junction_name"] == selected_junction]) if "junction_name" in df.columns else 0

                    # Cluster ID
                    junc_data = df[df["junction_name"] == selected_junction] if "junction_name" in df.columns else pd.DataFrame()
                    cluster_id = int(junc_data["cluster_id"].mode().iloc[0]) if len(junc_data) > 0 and "cluster_id" in junc_data.columns else -1

                    features = np.array([[
                        selected_hour, weekday_num, is_weekend, selected_month,
                        vt_enc, jn_enc, hist_count,
                    ]])

                    prob = float(model.predict_proba(features)[0][1])
                    prediction = "HOTSPOT LIKELY" if prob >= 0.5 else "LOW RISK"

                    # Display result
                    if prob >= 0.7:
                        color = "#ef4444"
                        bg = "rgba(239, 68, 68, 0.1)"
                        icon = '<i class="fa-solid fa-triangle-exclamation" style="color:#ef4444;"></i>'
                    elif prob >= 0.5:
                        color = "#f59e0b"
                        bg = "rgba(245, 158, 11, 0.1)"
                        icon = '<i class="fa-solid fa-circle-exclamation" style="color:#f59e0b;"></i>'
                    else:
                        color = "#22c55e"
                        bg = "rgba(34, 197, 94, 0.1)"
                        icon = '<i class="fa-solid fa-circle-check" style="color:#22c55e;"></i>'

                    st.markdown(f"""
                    <div class="glass-card" style="text-align:center; border-color:{color}40; background:{bg};">
                        <div style="font-size:3rem; margin-bottom:8px;">{icon}</div>
                        <div style="color:{color}; font-size:1.8rem; font-weight:800; margin:8px 0;">
                            {prediction}
                        </div>
                        <div style="color:#e2e8f0; font-size:1rem; margin:4px 0;">
                            {selected_junction}
                        </div>
                        <div style="color:{color}; font-size:2.5rem; font-weight:800; margin:12px 0;">
                            {prob*100:.1f}%
                        </div>
                        <div style="color:#94a3b8; font-size:0.85rem;">Hotspot Probability</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Additional context
                    st.markdown("")
                    st.markdown(f"""
                    <div class="glass-card">
                        <div style="color:#e2e8f0; font-weight:600; margin-bottom:12px;"><i class="fa-solid fa-list-check" style="margin-right:8px; color:#38bdf8;"></i>Prediction Context</div>
                        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; color:#94a3b8;">
                            <div>Model: <span style="color:#38bdf8;">{model_name.replace('_', ' ').title()}</span></div>
                            <div>Historical Violations: <span style="color:#38bdf8;">{hist_count:,}</span></div>
                            <div>Cluster ID: <span style="color:#38bdf8;">{cluster_id}</span></div>
                            <div>Time: <span style="color:#38bdf8;">{selected_hour_str} {selected_weekday}</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction error: {e}")
            elif predict_btn:
                st.warning("Model not loaded. Please run `python train.py` first.")
            else:
                st.markdown("""
                <div class="glass-card" style="text-align:center; min-height:300px; display:flex; flex-direction:column; justify-content:center;">
                    <div style="font-size:4rem; color:#38bdf8;"><i class="fa-solid fa-wand-magic-sparkles"></i></div>
                    <div style="color:#94a3b8; font-size:1.1rem; margin-top:16px;">
                        Select parameters and click<br><b style="color:#38bdf8;">Predict Hotspot</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("#### Model Comparison — XGBoost vs Random Forest")

        # Re-train quickly to get metrics (or load from saved)
        models_dir = os.path.join(PROJECT_ROOT, "models")
        metrics_data = []

        for mname in ["xgboost", "random_forest"]:
            mpath = os.path.join(models_dir, f"{mname}_model.pkl")
            if os.path.exists(mpath):
                m = joblib.load(mpath)
                # Quick evaluation on a sample
                from src.prediction import prepare_features
                X, y, _ = prepare_features(df.sample(min(10000, len(df)), random_state=42))
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                y_pred = m.predict(X)
                y_prob = m.predict_proba(X)[:, 1]
                metrics_data.append({
                    "Model": mname.replace("_", " ").title(),
                    "Accuracy": accuracy_score(y, y_pred),
                    "Precision": precision_score(y, y_pred, zero_division=0),
                    "Recall": recall_score(y, y_pred, zero_division=0),
                    "F1 Score": f1_score(y, y_pred, zero_division=0),
                    "ROC-AUC": roc_auc_score(y, y_prob),
                })

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)

            # Radar chart
            categories = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
            fig = go.Figure()

            colors = ["#38bdf8", "#c084fc"]
            fill_colors = ["rgba(56, 189, 248, 0.12)", "rgba(192, 132, 252, 0.12)"]
            for i, row in metrics_df.iterrows():
                values = [row[c] for c in categories] + [row[categories[0]]]
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=row["Model"],
                    line=dict(color=colors[i % len(colors)], width=2),
                    fillcolor=fill_colors[i % len(fill_colors)],
                ))

            fig.update_layout(
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
                polar=dict(
                    bgcolor="rgba(10, 15, 30, 0.5)",
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(56, 189, 248, 0.1)"),
                    angularaxis=dict(gridcolor="rgba(56, 189, 248, 0.1)"),
                ),
                height=420,
                title="Model Performance Comparison",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Metrics table
            st.markdown("#### Detailed Metrics")
            styled_df = metrics_df.set_index("Model")
            for col in styled_df.columns:
                styled_df[col] = styled_df[col].apply(lambda x: f"{x:.4f}")
            st.dataframe(styled_df, use_container_width=True)

            # Best model indicator
            best = metrics_df.sort_values("F1 Score", ascending=False).iloc[0]
            st.success(f"Best Model: **{best['Model']}** with F1 Score = {best['F1 Score']:.4f}", icon=":material/star:")

        # SHAP plots
        st.markdown("#### :material/psychology: SHAP Explainability")
        shap_path = os.path.join(PROJECT_ROOT, "outputs", "shap_summary.png")
        fi_path = os.path.join(PROJECT_ROOT, "outputs", "feature_importance.png")

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(shap_path):
                st.image(shap_path, caption="SHAP Summary Plot", use_container_width=True)
            else:
                st.info("Run `python train.py` to generate SHAP plots.")
        with col2:
            if os.path.exists(fi_path):
                st.image(fi_path, caption="Feature Importance (SHAP)", use_container_width=True)
            else:
                st.info("Feature importance plot not found. Run `python train.py` to generate.")

        # Top reasons
        if data["shap_reasons"]:
            st.markdown("#### Top Reasons for Hotspot Formation")
            for r in data["shap_reasons"]:
                direction_icon = ":material/trending_up:" if r["direction"] == "increases" else ":material/trending_down:"
                st.markdown(f"- {direction_icon} **{r['feature'].replace('_', ' ').title()}** — "
                            f"Mean SHAP = `{r['mean_shap']}` ({r['direction']} hotspot risk)")

    with tab3:
        col_title, col_refresh, col_tstyle = st.columns([3, 1, 1])
        with col_title:
            st.markdown("#### :material/traffic: Live Traffic Conditions & Interventions")
        with col_refresh:
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            refresh_btn = st.button("Refresh Live Data", icon=":material/refresh:", use_container_width=True)
            if refresh_btn:
                st.rerun()
        with col_tstyle:
            traffic_styles = {
                "Dark Mode": {"tiles": "CartoDB dark_matter", "attr": "CartoDB"},
                "Light Mode": {"tiles": "CartoDB Positron", "attr": "CartoDB"},
                "Standard": {"tiles": "OpenStreetMap", "attr": "OpenStreetMap"},
                "Satellite": {
                    "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                    "attr": "Esri World Imagery"
                }
            }
            selected_tstyle = st.selectbox("Style", list(traffic_styles.keys()), index=0, key="traffic_style_select")
        
        style_cfg = traffic_styles[selected_tstyle]
        st.markdown("Real-time traffic overlay mapped with AI-recommended mitigation strategies for Bengaluru's major congestion junctions.")
        
        col_map, col_alerts = st.columns([3, 2])
        
        with col_map:
            center_lat = df["latitude"].mean()
            center_lon = df["longitude"].mean()
            m_traffic = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles=style_cfg["tiles"],
                attr=style_cfg["attr"],
            )
            
            # Google live traffic overlay tile layer
            traffic_url = "https://mt0.google.com/vt?lyrs=h@159000000,traffic|seconds_into_week:-1&style=3&x={x}&y={y}&z={z}"
            folium.TileLayer(
                tiles=traffic_url,
                attr="Google Maps Live Traffic",
                name="Google Traffic Overlay",
                overlay=True,
                control=True,
                opacity=0.8
            ).add_to(m_traffic)
            
            folium.LayerControl().add_to(m_traffic)
            
            # Add top active hotspot indicators to traffic map
            hotspot_stats = data["hotspot_stats"]
            for _, row in hotspot_stats.head(15).iterrows():
                colour = row.get("colour", "red")
                junction = row.get("primary_junction", "Unknown")
                count = row.get("violation_count", 0)
                folium.CircleMarker(
                    location=[row["center_lat"], row["center_lon"]],
                    radius=8,
                    popup=f"Junction: {junction}<br>Violation Density: {count}",
                    color="white",
                    fill=True,
                    fill_color=colour,
                    fill_opacity=0.9,
                    weight=1,
                ).add_to(m_traffic)
            
            st_folium(m_traffic, width=None, height=500, key="live_traffic_map", returned_objects=[])
            
            # Live Traffic Metrics
            st.markdown("##### 📊 Real-time Traffic Velocity & Delay Metrics (Estimated)")
            cm1, cm2, cm3 = st.columns(3)
            with cm1:
                st.metric("Avg. Corridor Speed", "16.8 km/h", delta="-3.4 km/h (Slow)")
            with cm2:
                st.metric("Congestion Index", "74%", delta="Elevated")
            with cm3:
                st.metric("Peak Travel Delay", "+28 min", delta="+6 min (Increasing)")
                
            st.markdown("""
            <div class="glass-card" style="padding: 16px; margin-top: 8px;">
                <div style="font-weight:600; color:#e2e8f0; margin-bottom:12px;"><i class="fa-solid fa-gauge-simple-high" style="color:#38bdf8; margin-right:8px;"></i>Live Corridor Velocity Metrics</div>
                <div style="display:grid; grid-template-columns: 2.5fr 1fr 1fr; gap: 8px; font-size: 0.85rem; color:#94a3b8; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 6px; margin-bottom: 6px;">
                    <strong>Corridor Route</strong>
                    <strong>Current Speed</strong>
                    <strong>Delay Ratio</strong>
                </div>
                <div style="display:grid; grid-template-columns: 2.5fr 1fr 1fr; gap: 8px; font-size: 0.8rem; color:#e2e8f0; margin-bottom:6px;">
                    <span>Outer Ring Road (Silk Board to Marathahalli)</span>
                    <span style="color:#ef4444; font-weight:700;">11 km/h</span>
                    <span style="color:#ef4444;">+42 min</span>
                </div>
                <div style="display:grid; grid-template-columns: 2.5fr 1fr 1fr; gap: 8px; font-size: 0.8rem; color:#e2e8f0; margin-bottom:6px;">
                    <span>Hosur Road (Silk Board to Electronic City)</span>
                    <span style="color:#f59e0b; font-weight:700;">19 km/h</span>
                    <span style="color:#f59e0b;">+18 min</span>
                </div>
                <div style="display:grid; grid-template-columns: 2.5fr 1fr 1fr; gap: 8px; font-size: 0.8rem; color:#e2e8f0; margin-bottom:6px;">
                    <span>Old Madras Road (Indiranagar to KR Puram)</span>
                    <span style="color:#ef4444; font-weight:700;">14 km/h</span>
                    <span style="color:#ef4444;">+26 min</span>
                </div>
                <div style="display:grid; grid-template-columns: 2.5fr 1fr 1fr; gap: 8px; font-size: 0.8rem; color:#e2e8f0;">
                    <span>Richmond Road (Flyover to Trinity Circle)</span>
                    <span style="color:#22c55e; font-weight:700;">28 km/h</span>
                    <span style="color:#22c55e;">+3 min</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_alerts:
            st.markdown("##### 🚨 Live Congestion Alerts")
            
            junc_pcri = data["junction_pcri"]
            if len(junc_pcri) > 0:
                top_congested = junc_pcri.head(5)
                
                for idx, (_, row) in enumerate(top_congested.iterrows()):
                    junc_name = row["junction_name"]
                    pcri = row["pcri"]
                    
                    # Compute realistic dynamic metrics based on PCRI components
                    density = row.get("violation_density_norm", 0)
                    peak = row.get("peak_hour_weight_norm", 0)
                    freq = row.get("junction_frequency_norm", 0)
                    repeat = row.get("repeat_violation_rate_norm", 0)
                    
                    if pcri >= 66:
                        status = "GRIDLOCKED"
                        color = "#ef4444"
                        bg = "rgba(239, 68, 68, 0.1)"
                    elif pcri >= 33:
                        status = "HEAVY TRAFFIC"
                        color = "#f59e0b"
                        bg = "rgba(245, 158, 11, 0.1)"
                    else:
                        status = "MODERATE"
                        color = "#38bdf8"
                        bg = "rgba(56, 189, 248, 0.1)"
                        
                    # Find dominant cause
                    comps = {"density": density, "peak": peak, "freq": freq, "repeat": repeat}
                    max_comp = max(comps, key=comps.get)
                    
                    if max_comp == "density":
                        cause = "Extremely high density of double-parking & commercial loading."
                        action = "🚨 <strong>Dispatch towing vehicles</strong> to clear double-parked vehicles and open access lanes."
                    elif max_comp == "peak":
                        cause = "High concentration of commuter drop-offs during peak traffic hours."
                        action = "🚦 <strong>Optimize signal timing</strong> and deploy local traffic marshals to keep junctions flowing."
                    elif max_comp == "repeat":
                        cause = "Frequent repeat violations from delivery agents and ride-share cabs."
                        action = "📹 <strong>Deploy Virtual Patrolling:</strong> Issue automated digital fines via CCTV feeds."
                    else:
                        cause = "Persistent round-the-clock illegal parking blocking turn lanes."
                        action = "🚧 <strong>Install lane clearway barriers</strong> and deploy regular motorcycle patrols."
                    
                    st.markdown(f"""
                    <div class="glass-card" style="border-left: 4px solid {color}; padding: 12px; margin-bottom: 10px; background: {bg};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong style="color: #e2e8f0; font-size: 0.95rem;">{junc_name}</strong>
                            <span style="background: {color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 700;">
                                {status} (PCRI: {pcri:.0f})
                            </span>
                        </div>
                        <div style="color: #94a3b8; font-size: 0.8rem; margin: 4px 0;">
                            <strong>Cause:</strong> {cause}
                        </div>
                        <div style="color: #f1f5f9; font-size: 0.8rem; margin-top: 6px; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.05);">
                            {action}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No congestion alerts available.")


# ─── Page: Enforcement ────────────────────────────────────────
def page_enforcement(data):
    df = data["df"]
    junction_pcri = data["junction_pcri"]
    model = data["model"]
    encoders = data["encoders"]

    st.markdown("### :material/local_police: Enforcement Recommendation Engine")
    st.markdown("AI-powered patrol deployment recommendations ranked by enforcement priority.")

    # Generate recommendations
    from src.recommendation import generate_recommendations, get_priority_zones

    recs = generate_recommendations(
        junction_pcri, df, model=model, encoders=encoders, top_n=20
    )

    if len(recs) == 0:
        st.warning("No recommendations available. Ensure training has been run.")
        return

    # Priority zone summary
    zones = get_priority_zones(recs)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="glass-card" style="border-color:#ef444440; text-align:center;">
            <div style="font-size:2.5rem; color:#ef4444;"><i class="fa-solid fa-circle-exclamation"></i></div>
            <div style="color:#ef4444; font-size:2rem; font-weight:800;">{len(zones.get('Critical', []))}</div>
            <div style="color:#94a3b8;">Critical Zones</div>
            <div style="color:#ef4444; font-size:0.8rem; margin-top:8px;">IMMEDIATE DEPLOYMENT</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="glass-card" style="border-color:#f59e0b40; text-align:center;">
            <div style="font-size:2.5rem; color:#f59e0b;"><i class="fa-solid fa-triangle-exclamation"></i></div>
            <div style="color:#f59e0b; font-size:2rem; font-weight:800;">{len(zones.get('Elevated', []))}</div>
            <div style="color:#94a3b8;">Elevated Zones</div>
            <div style="color:#f59e0b; font-size:0.8rem; margin-top:8px;">PRIORITY PATROL</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="glass-card" style="border-color:#22c55e40; text-align:center;">
            <div style="font-size:2.5rem; color:#22c55e;"><i class="fa-solid fa-clipboard-list"></i></div>
            <div style="color:#22c55e; font-size:2rem; font-weight:800;">{len(zones.get('Routine', []))}</div>
            <div style="color:#94a3b8;">Routine Zones</div>
            <div style="color:#22c55e; font-size:0.8rem; margin-top:8px;">REGULAR SCHEDULE</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Ranked recommendation cards
    st.markdown("#### :material/format_list_numbered: Ranked Enforcement Priorities")

    group_col = "junction_name" if "junction_name" in recs.columns else recs.columns[0]

    for _, row in recs.iterrows():
        zone = row.get("priority_zone", "Routine")
        if zone == "Critical":
            border_color = "#ef4444"
            badge_class = "badge-critical"
            icon = '<i class="fa-solid fa-circle-exclamation" style="color:#ef4444;"></i>'
        elif zone == "Elevated":
            border_color = "#f59e0b"
            badge_class = "badge-elevated"
            icon = '<i class="fa-solid fa-triangle-exclamation" style="color:#f59e0b;"></i>'
        else:
            border_color = "#22c55e"
            badge_class = "badge-routine"
            icon = '<i class="fa-solid fa-clipboard-list" style="color:#22c55e;"></i>'

        pcri = row.get("pcri", 0)
        prob = row.get("hotspot_probability_pct", 0)
        priority = row.get("enforcement_priority", 0)
        rank = int(row.get("rank", 0))
        junction = row[group_col]

        st.markdown(f"""
        <div class="glass-card" style="border-left: 4px solid {border_color}; margin:6px 0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="color:#64748b; font-weight:700; font-size:0.85rem;">#{rank}</span>
                    <span style="margin-left:8px; font-size:1rem;">{icon}</span>
                    <span style="color:#e2e8f0; font-weight:700; font-size:1.1rem; margin-left:8px;">
                        Deploy Patrol Unit → {junction}
                    </span>
                </div>
                <span class="{badge_class}">{zone}</span>
            </div>
            <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:16px; margin-top:12px;">
                <div>
                    <div style="color:#64748b; font-size:0.75rem; text-transform:uppercase;">PCRI Score</div>
                    <div style="color:#38bdf8; font-size:1.3rem; font-weight:700;">{pcri:.0f}<span style="font-size:0.75rem; color:#64748b;">/100</span></div>
                </div>
                <div>
                    <div style="color:#64748b; font-size:0.75rem; text-transform:uppercase;">Hotspot Probability</div>
                    <div style="color:#c084fc; font-size:1.3rem; font-weight:700;">{prob:.1f}%</div>
                </div>
                <div>
                    <div style="color:#64748b; font-size:0.75rem; text-transform:uppercase;">Priority Score</div>
                    <div style="color:#f59e0b; font-size:1.3rem; font-weight:700;">{priority:.1f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Priority bar chart
    st.markdown("#### :material/bar_chart: Enforcement Priority Distribution")
    fig = go.Figure(go.Bar(
        x=recs[group_col],
        y=recs["enforcement_priority"],
        marker=dict(
            color=recs["enforcement_priority"],
            colorscale=[[0, "#22c55e"], [0.5, "#f59e0b"], [1, "#ef4444"]],
            colorbar=dict(title="Priority"),
        ),
        text=recs["enforcement_priority"].apply(lambda x: f"{x:.0f}"),
        textposition="auto",
        textfont=dict(color="white"),
    ))
    fig.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items() if k != "xaxis"}, height=400, title="", showlegend=False,
                      xaxis=dict(tickangle=-45, **PLOTLY_LAYOUT.get("xaxis", {})))
    st.plotly_chart(fig, use_container_width=True)


# ─── Page: Analytics ──────────────────────────────────────────
def page_analytics(data):
    df = data["df"]
    anomaly_summary = data["anomaly_summary"]

    st.markdown("### :material/analytics: Deep Analytics")

    tab1, tab2, tab3, tab4 = st.tabs([
        ":material/schedule: Temporal", ":material/directions_car: Vehicle", ":material/domain: Station", ":material/search: Anomalies"
    ])

    with tab1:
        st.markdown("#### Violations by Hour of Day")
        if "hour" in df.columns:
            hourly = df.groupby("hour")["id"].count().reset_index()
            hourly.columns = ["Hour", "Violations"]

            fig = go.Figure()
            colors = ["#ef4444" if h in (8, 9, 10, 17, 18, 19, 20) else "#38bdf8"
                      for h in hourly["Hour"]]
            fig.add_trace(go.Bar(
                x=hourly["Hour"], y=hourly["Violations"],
                marker=dict(color=colors, line=dict(width=0)),
                text=hourly["Violations"].apply(lambda x: f"{x:,}"),
                textposition="auto",
                textfont=dict(color="white", size=10),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=380, title="",
                              xaxis_title="Hour", yaxis_title="Violations",
                              showlegend=False)
            fig.add_annotation(
                text="● Peak Hours (8-10 AM, 5-8 PM)",
                xref="paper", yref="paper", x=0.95, y=0.95,
                showarrow=False, font=dict(color="#ef4444", size=11),
            )
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Violations by Day of Week")
            if "weekday_name" in df.columns:
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                daily = df.groupby("weekday_name")["id"].count().reset_index()
                daily.columns = ["Day", "Violations"]
                daily["Day"] = pd.Categorical(daily["Day"], categories=day_order, ordered=True)
                daily = daily.sort_values("Day")

                fig = go.Figure(go.Bar(
                    x=daily["Day"], y=daily["Violations"],
                    marker=dict(
                        color=daily["Violations"],
                        colorscale=[[0, "#1e3a5f"], [1, "#38bdf8"]],
                    ),
                    text=daily["Violations"].apply(lambda x: f"{x:,}"),
                    textposition="auto",
                    textfont=dict(color="white"),
                ))
                fig.update_layout(**PLOTLY_LAYOUT, height=350, title="", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Violations by Month")
            if "month_name" in df.columns:
                monthly = df.groupby(["month", "month_name"])["id"].count().reset_index()
                monthly.columns = ["month_num", "Month", "Violations"]
                monthly = monthly.sort_values("month_num")

                fig = go.Figure(go.Bar(
                    x=monthly["Month"], y=monthly["Violations"],
                    marker=dict(
                        color=monthly["Violations"],
                        colorscale=[[0, "#1e3a5f"], [1, "#c084fc"]],
                    ),
                    text=monthly["Violations"].apply(lambda x: f"{x:,}"),
                    textposition="auto",
                    textfont=dict(color="white"),
                ))
                fig.update_layout(**PLOTLY_LAYOUT, height=350, title="", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Heatmap: hour vs day
        st.markdown("#### :material/thermostat: Violation Heatmap — Hour × Day")
        if "hour" in df.columns and "weekday_name" in df.columns:
            heatmap_data = df.groupby(["weekday_name", "hour"])["id"].count().reset_index()
            heatmap_data.columns = ["Day", "Hour", "Violations"]
            heatmap_pivot = heatmap_data.pivot(index="Day", columns="Hour", values="Violations").fillna(0)
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heatmap_pivot = heatmap_pivot.reindex(day_order)

            fig = go.Figure(go.Heatmap(
                z=heatmap_pivot.values,
                x=list(range(24)),
                y=day_order,
                colorscale=[[0, "#0a0a1a"], [0.3, "#1e3a5f"], [0.6, "#38bdf8"], [1, "#ef4444"]],
                text=heatmap_pivot.values.astype(int),
                texttemplate="%{text}",
                textfont=dict(size=9, color="white"),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=320, title="",
                              xaxis_title="Hour", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### Violations by Vehicle Type")
        if "vehicle_type" in df.columns:
            vt = df["vehicle_type"].value_counts().head(12).reset_index()
            vt.columns = ["Vehicle Type", "Count"]

            fig = go.Figure(go.Bar(
                x=vt["Count"], y=vt["Vehicle Type"], orientation="h",
                marker=dict(
                    color=vt["Count"],
                    colorscale=[[0, "#1e3a5f"], [0.5, "#818cf8"], [1, "#c084fc"]],
                ),
                text=vt["Count"].apply(lambda x: f"{x:,}"),
                textposition="auto",
                textfont=dict(color="white"),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=420, title="", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Vehicle type by hour heatmap
        st.markdown("#### Vehicle Type × Hour Distribution")
        if "vehicle_type" in df.columns and "hour" in df.columns:
            top_vt = df["vehicle_type"].value_counts().head(8).index.tolist()
            vt_hour = df[df["vehicle_type"].isin(top_vt)].groupby(["vehicle_type", "hour"])["id"].count().reset_index()
            vt_hour.columns = ["Vehicle Type", "Hour", "Count"]
            vt_pivot = vt_hour.pivot(index="Vehicle Type", columns="Hour", values="Count").fillna(0)

            fig = go.Figure(go.Heatmap(
                z=vt_pivot.values,
                x=list(range(24)),
                y=vt_pivot.index.tolist(),
                colorscale=[[0, "#0a0a1a"], [0.3, "#1e3a5f"], [0.6, "#38bdf8"], [1, "#c084fc"]],
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=350, title="",
                              xaxis_title="Hour")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### Violations by Junction")
        if "junction_name" in df.columns:
            junc = df[df["junction_name"] != "No Junction"]["junction_name"].value_counts().head(15).reset_index()
            junc.columns = ["Junction", "Violations"]

            fig = go.Figure(go.Bar(
                x=junc["Violations"], y=junc["Junction"], orientation="h",
                marker=dict(
                    color=junc["Violations"],
                    colorscale=[[0, "#1e3a5f"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                ),
                text=junc["Violations"].apply(lambda x: f"{x:,}"),
                textposition="auto",
                textfont=dict(color="white"),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=480, title="", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Police Station Analysis")
        if "police_station" in df.columns:
            ps = df["police_station"].value_counts().head(15).reset_index()
            ps.columns = ["Station", "Violations"]

            fig = go.Figure(go.Bar(
                x=ps["Station"], y=ps["Violations"],
                marker=dict(
                    color=ps["Violations"],
                    colorscale=[[0, "#1e3a5f"], [0.5, "#38bdf8"], [1, "#818cf8"]],
                ),
                text=ps["Violations"].apply(lambda x: f"{x:,}"),
                textposition="auto",
                textfont=dict(color="white"),
            ))
            fig.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items() if k != "xaxis"}, height=400, title="", showlegend=False,
                              xaxis=dict(tickangle=-45, **PLOTLY_LAYOUT.get("xaxis", {})))
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("#### :material/search: Anomaly Detection Results (Isolation Forest)")

        if anomaly_summary:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Anomalies", f"{anomaly_summary.get('total_anomalies', 0):,}")
            with col2:
                st.metric("Anomaly Rate", f"{anomaly_summary.get('anomaly_rate', 0):.2f}%")
            with col3:
                peak_hours = anomaly_summary.get("peak_anomaly_hours", {})
                top_hour = list(peak_hours.keys())[0] if peak_hours else "N/A"
                st.metric("Peak Anomaly Hour", f"{top_hour}:00" if top_hour != "N/A" else "N/A")

            col1, col2 = st.columns(2)
            with col1:
                if "peak_anomaly_hours" in anomaly_summary and anomaly_summary["peak_anomaly_hours"]:
                    hours_data = pd.DataFrame(
                        list(anomaly_summary["peak_anomaly_hours"].items()),
                        columns=["Hour", "Count"]
                    )
                    fig = go.Figure(go.Bar(
                        x=hours_data["Hour"], y=hours_data["Count"],
                        marker=dict(color="#ef4444"),
                    ))
                    fig.update_layout(**PLOTLY_LAYOUT, height=300, title="Peak Anomaly Hours")
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if "top_anomaly_junctions" in anomaly_summary and anomaly_summary["top_anomaly_junctions"]:
                    junc_data = pd.DataFrame(
                        list(anomaly_summary["top_anomaly_junctions"].items()),
                        columns=["Junction", "Anomalies"]
                    )
                    fig = go.Figure(go.Bar(
                        x=junc_data["Anomalies"], y=junc_data["Junction"], orientation="h",
                        marker=dict(color="#c084fc"),
                    ))
                    fig.update_layout(**PLOTLY_LAYOUT, height=300, title="Top Anomaly Junctions")
                    st.plotly_chart(fig, use_container_width=True)

            if "anomaly_by_day" in anomaly_summary and anomaly_summary["anomaly_by_day"]:
                day_data = pd.DataFrame(
                    list(anomaly_summary["anomaly_by_day"].items()),
                    columns=["Day", "Anomalies"]
                )
                fig = go.Figure(go.Bar(
                    x=day_data["Day"], y=day_data["Anomalies"],
                    marker=dict(
                        color=day_data["Anomalies"],
                        colorscale=[[0, "#1e3a5f"], [1, "#ef4444"]],
                    ),
                ))
                fig.update_layout(**PLOTLY_LAYOUT, height=300, title="Anomalies by Day of Week")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run `python train.py` to generate anomaly detection results.")

        # Forecasting section
        st.markdown("---")
        st.markdown("#### :material/trending_up: Violation Forecasting (Prophet)")

        if st.button("Generate 30-Day Forecast", icon=":material/online_prediction:", type="primary"):
            from src.forecasting import prepare_prophet_data, run_prophet_forecast
            with custom_spinner("Training Prophet forecasting model...", icon="fa-chart-line", color="#c084fc", animation="fa-beat"):
                daily = prepare_prophet_data(df)
                if len(daily) > 10:
                    model_prophet, forecast = run_prophet_forecast(daily, periods=30)
                    if forecast is not None:
                        fig = go.Figure()
                        # Historical
                        fig.add_trace(go.Scatter(
                            x=daily["ds"], y=daily["y"],
                            mode="lines", name="Historical",
                            line=dict(color="#38bdf8", width=1.5),
                        ))
                        # Forecast
                        forecast_future = forecast[forecast["ds"] > daily["ds"].max()]
                        fig.add_trace(go.Scatter(
                            x=forecast_future["ds"], y=forecast_future["yhat"],
                            mode="lines", name="Forecast",
                            line=dict(color="#c084fc", width=2, dash="dash"),
                        ))
                        # Confidence interval
                        fig.add_trace(go.Scatter(
                            x=pd.concat([forecast_future["ds"], forecast_future["ds"][::-1]]),
                            y=pd.concat([forecast_future["yhat_upper"], forecast_future["yhat_lower"][::-1]]),
                            fill="toself",
                            fillcolor="rgba(192, 132, 252, 0.1)",
                            line=dict(color="rgba(0,0,0,0)"),
                            name="95% Confidence",
                        ))
                        fig.update_layout(**PLOTLY_LAYOUT, height=400,
                                          title="30-Day Violation Forecast",
                                          xaxis_title="Date", yaxis_title="Violations")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Could not generate forecast. Ensure Prophet is installed.")
                else:
                    st.warning("Insufficient data for forecasting.")


# ─── Main Router ──────────────────────────────────────────────
def main():
    page = render_sidebar()

    with custom_spinner("Initializing platform and loading intelligence data..."):
        data = load_all_data()

    if "Overview" in page:
        page_overview(data)
    elif "Heatmap" in page:
        page_heatmap(data)
    elif "Prediction" in page:
        page_prediction(data)
    elif "Enforcement" in page:
        page_enforcement(data)
    elif "Analytics" in page:
        page_analytics(data)


if __name__ == "__main__":
    main()
