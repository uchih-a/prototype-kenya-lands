import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import plotly.express as px
import plotly.graph_objects as go
from math import radians, sin, cos, sqrt, atan2
import os

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Kenya Land Valuation",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f2027, #203a43, #2c5364);
}
[data-testid="stSidebar"] * {
    color: #e8f4f8 !important;
}
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stSelectbox > label {
    color: #a8d8ea !important;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a3a4a 0%, #0d2233 100%);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(100,200,255,0.15);
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    text-align: center;
    margin-bottom: 1rem;
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #7ecbde;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #ffffff;
    line-height: 1.1;
}
.metric-sub {
    font-size: 0.75rem;
    color: #a8c8d8;
    margin-top: 0.3rem;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(100,200,255,0.1);
}
.hero h1 {
    color: #ffffff;
    font-size: 2.4rem;
    margin: 0 0 0.5rem 0;
}
.hero p {
    color: #a8d8ea;
    font-size: 1rem;
    margin: 0;
}

/* Score badge */
.score-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-good { background: rgba(34,197,94,0.2); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.badge-med  { background: rgba(251,191,36,0.2); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.badge-low  { background: rgba(239,68,68,0.2);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }

.stButton > button {
    background: linear-gradient(135deg, #1e7fa8, #0e4d6a);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #28a0d0, #1e6a8a);
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(30,127,168,0.4);
}
</style>
""", unsafe_allow_html=True)


# ── MLP Architecture (must match training) ─────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# ── Haversine ──────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ── County reference data ──────────────────────────────────────
COUNTY_TOWNS = {
    "Nairobi": (-1.2921, 36.8219), "Mombasa": (-4.0435, 39.6682),
    "Kwale": (-4.1816, 39.4600), "Kilifi": (-3.5107, 39.9093),
    "Tana River": (-1.4000, 40.0000), "Lamu": (-2.2686, 40.9020),
    "Taita Taveta": (-3.4000, 38.5000), "Garissa": (-0.4532, 39.6461),
    "Meru": (0.0470, 37.6490), "Embu": (-0.5300, 37.4500),
    "Machakos": (-1.5177, 37.2634), "Kitui": (-1.3667, 38.0167),
    "Makueni": (-2.2558, 37.8942), "Nyandarua": (-0.1000, 36.3700),
    "Nyeri": (-0.4167, 36.9500), "Kirinyaga": (-0.6500, 37.3000),
    "Murang'a": (-0.7167, 37.1500), "Kiambu": (-1.0317, 36.8300),
    "Turkana": (3.1167, 35.5959), "West Pokot": (1.6204, 35.1173),
    "Samburu": (1.0740, 36.7000), "Trans Nzoia": (1.0170, 35.0023),
    "Uasin Gishu": (0.5143, 35.2699), "Elgeyo Marakwet": (0.9000, 35.5000),
    "Nandi": (0.1833, 35.1167), "Baringo": (0.4667, 35.9667),
    "Laikipia": (0.3606, 36.7819), "Nakuru": (-0.3031, 36.0800),
    "Narok": (-1.0833, 35.8667), "Kajiado": (-1.8500, 36.7833),
    "Kericho": (-0.3690, 35.2863), "Bomet": (-0.7833, 35.3333),
    "Kakamega": (0.2827, 34.7519), "Vihiga": (0.0667, 34.7167),
    "Bungoma": (0.5635, 34.5606), "Busia": (0.4667, 34.1167),
    "Siaya": (-0.0613, 34.2884), "Kisumu": (-0.1022, 34.7617),
    "Homa Bay": (-0.5167, 34.4500), "Migori": (-1.0634, 34.4731),
    "Kisii": (-0.6817, 34.7667), "Nyamira": (-0.5667, 34.9333),
    "Marsabit": (2.3284, 37.9899), "Isiolo": (0.3540, 37.5820),
    "Wajir": (1.7471, 40.0573), "Mandera": (3.9366, 41.8670),
    "Tharaka Nithi": (-0.3000, 37.8800),
}
NAIROBI_CBD = (-1.2921, 36.8219)
NAIROBI_ADJACENT = {"Kiambu", "Kajiado", "Machakos", "Murang'a"}
WATER_BODY_COUNTIES = {
    "Mombasa": (-4.0435, 39.6682), "Kwale": (-4.1816, 39.4600),
    "Kilifi": (-3.5107, 39.9093), "Lamu": (-2.2686, 40.9020),
    "Kisumu": (-0.1022, 34.7617), "Homa Bay": (-0.5167, 34.4500),
    "Siaya": (-0.0613, 34.2884), "Migori": (-1.0634, 34.4731),
    "Nakuru": (-0.3031, 36.0800),
}


# ── Load model artifacts ───────────────────────────────────────
@st.cache_resource
def load_model():
    """Load MLP model, scaler, and feature list from the models/ folder."""
    model_dir = "models"
    required = ["mlp_model.pt", "mlp_scaler.pkl", "mlp_feature_list.pkl"]
    missing = [f for f in required if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        return None, None, None, missing

    features = joblib.load(os.path.join(model_dir, "mlp_feature_list.pkl"))
    scaler   = joblib.load(os.path.join(model_dir, "mlp_scaler.pkl"))
    model    = MLP(input_dim=len(features))
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "mlp_model.pt"), map_location="cpu")
    )
    model.eval()
    return model, scaler, features, []


model, scaler, MLP_FEATURES, missing_files = load_model()


# ── Prediction helper ──────────────────────────────────────────
def predict_price(feature_dict):
    """Run one prediction. Returns (log_pred, ksh_pred_per_acre)."""
    vec = np.array([[feature_dict[f] for f in MLP_FEATURES]], dtype=np.float32)
    vec_sc = scaler.transform(vec)
    with torch.no_grad():
        log_pred = model(torch.FloatTensor(vec_sc)).item()
    return log_pred, np.exp(log_pred)


def compute_features(county, size_acres, lat, lon,
                      amenities_score, accessibility_score, infrastructure_score,
                      geocode_confidence):
    """Compute all derived proximity features."""
    dist_nairobi = haversine_km(lat, lon, *NAIROBI_CBD)

    town_coords = COUNTY_TOWNS.get(county)
    dist_town = haversine_km(lat, lon, *town_coords) if town_coords else dist_nairobi

    if county in NAIROBI_ADJACENT and dist_nairobi < dist_town:
        ref_dist = dist_nairobi
    else:
        ref_dist = dist_town

    water_coords = WATER_BODY_COUNTIES.get(county)
    dist_water = haversine_km(lat, lon, *water_coords) if water_coords else np.nan

    return {
        "log_size_acres":         np.log(size_acres) if size_acres > 0 else 0,
        "dist_to_nairobi_km":     round(dist_nairobi, 2),
        "dist_to_county_town_km": round(dist_town, 2),
        "dist_to_water_body_km":  round(dist_water, 2) if not np.isnan(dist_water) else 0,
        "reference_city_dist_km": round(ref_dist, 2),
        "geocode_confidence":     geocode_confidence,
        "amenities_score":        amenities_score,
        "accessibility_score":    accessibility_score,
        "infrastructure_score":   infrastructure_score,
    }


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR — inputs
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌍 Kenya Land Valuation")
    st.markdown("---")

    st.markdown("### 📍 Location")
    county = st.selectbox("County", sorted(COUNTY_TOWNS.keys()), index=list(sorted(COUNTY_TOWNS.keys())).index("Nairobi"))

    default_lat, default_lon = COUNTY_TOWNS[county]
    lat = st.number_input("Latitude", value=round(default_lat, 4), format="%.4f", step=0.001)
    lon = st.number_input("Longitude", value=round(default_lon, 4), format="%.4f", step=0.001)

    st.markdown("### 📐 Plot Details")
    size_acres = st.number_input("Plot Size (acres)", min_value=0.05, max_value=50000.0, value=0.5, step=0.1)
    geocode_confidence = st.slider("Location Confidence", 0.2, 1.0, 0.8, 0.1,
                                   help="How precisely is the location known? (1.0 = GPS, 0.2 = county centroid)")

    st.markdown("### 🏗️ Pillar Scores")
    st.caption("Leave as-is to use county averages, or adjust for a specific property.")
    amenities_score     = st.slider("Amenities Score",      0, 100, 50, help="Proximity to hospitals, schools, markets, banks")
    accessibility_score = st.slider("Accessibility Score",  0, 100, 50, help="Road quality + city proximity decay")
    infrastructure_score= st.slider("Infrastructure Score", 0, 100,  5, help="On-site infrastructure: borehole, power, fencing")

    st.markdown("---")
    predict_btn = st.button("🔮 Estimate Value")


# ═══════════════════════════════════════════════════════════════
#  MAIN — hero + results
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>🌍 Kenya Land Valuation</h1>
  <p>MLP-powered price estimation for Kenyan land listings — enter property details in the sidebar and click Estimate Value.</p>
</div>
""", unsafe_allow_html=True)

# ── Model status ───────────────────────────────────────────────
if missing_files:
    st.error(f"⚠️ **Model files not found.** Please add these to a `models/` folder:\n" +
             "\n".join(f"- `{f}`" for f in missing_files))
    st.info("📂 Upload `mlp_model.pt`, `mlp_scaler.pkl`, and `mlp_feature_list.pkl` from your Google Drive outputs folder.")
    st.stop()

# ── Run prediction ─────────────────────────────────────────────
features = compute_features(
    county, size_acres, lat, lon,
    amenities_score, accessibility_score, infrastructure_score, geocode_confidence
)

if predict_btn or True:   # show preview even before button press
    try:
        log_pred, price_per_acre = predict_price(features)
        total_price = price_per_acre * size_acres

        # ── Key metrics ────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Price per Acre</div>
              <div class="metric-value">KSh {price_per_acre:,.0f}</div>
              <div class="metric-sub">Estimated market rate</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Total Plot Value</div>
              <div class="metric-value">KSh {total_price:,.0f}</div>
              <div class="metric-sub">{size_acres} acre{'s' if size_acres != 1 else ''}</div>
            </div>""", unsafe_allow_html=True)

        with col3:
            dist_n = features["dist_to_nairobi_km"]
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Distance to Nairobi</div>
              <div class="metric-value">{dist_n:.1f} km</div>
              <div class="metric-sub">From listed coordinates</div>
            </div>""", unsafe_allow_html=True)

        with col4:
            ref_d = features["reference_city_dist_km"]
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">Ref. City Distance</div>
              <div class="metric-value">{ref_d:.1f} km</div>
              <div class="metric-sub">{county} reference</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Two-column layout ──────────────────────────────────
        left, right = st.columns([1.2, 1])

        with left:
            st.subheader("📊 Feature Breakdown")

            # Gauge chart for pillar scores
            fig_gauge = go.Figure()
            scores = {
                "Amenities":      amenities_score,
                "Accessibility":  accessibility_score,
                "Infrastructure": infrastructure_score,
            }
            colors = ["#1e88e5", "#43a047", "#fb8c00"]
            for i, (name, val) in enumerate(scores.items()):
                fig_gauge.add_trace(go.Bar(
                    x=[val],
                    y=[name],
                    orientation="h",
                    marker_color=colors[i],
                    text=f"{val}",
                    textposition="outside",
                    width=0.5,
                ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.04)",
                font=dict(color="#e0e0e0", family="DM Sans"),
                xaxis=dict(range=[0, 115], showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False),
                margin=dict(l=0, r=30, t=10, b=10),
                height=160,
                showlegend=False,
                bargap=0.35,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Feature importance table
            feat_df = pd.DataFrame({
                "Feature": [
                    "Log Size (acres)",
                    "Distance to Nairobi",
                    "Dist to County Town",
                    "Dist to Water Body",
                    "Ref. City Distance",
                    "Geocode Confidence",
                    "Amenities Score",
                    "Accessibility Score",
                    "Infrastructure Score",
                ],
                "Value": [
                    f"{features['log_size_acres']:.3f}",
                    f"{features['dist_to_nairobi_km']:.1f} km",
                    f"{features['dist_to_county_town_km']:.1f} km",
                    f"{features['dist_to_water_body_km']:.1f} km",
                    f"{features['reference_city_dist_km']:.1f} km",
                    f"{features['geocode_confidence']:.2f}",
                    f"{amenities_score}",
                    f"{accessibility_score}",
                    f"{infrastructure_score}",
                ],
            })
            st.dataframe(feat_df, hide_index=True, use_container_width=True)

        with right:
            st.subheader("🗺️ Property Location")

            map_df = pd.DataFrame({
                "lat": [lat],
                "lon": [lon],
                "label": [f"{county}\nKSh {price_per_acre:,.0f}/acre"],
            })
            fig_map = px.scatter_mapbox(
                map_df, lat="lat", lon="lon",
                hover_data={"label": True, "lat": False, "lon": False},
                color_discrete_sequence=["#1e88e5"],
                zoom=9,
                height=340,
            )
            fig_map.update_traces(marker=dict(size=16))
            fig_map.update_layout(
                mapbox_style="carto-darkmatter",
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_map, use_container_width=True)

            # Price bands context
            st.markdown("#### Price Context")
            price_bands = {
                "Budget  (< KSh 1M/acre)":       1_000_000,
                "Mid-range (1M–10M/acre)":       10_000_000,
                "Premium (10M–50M/acre)":        50_000_000,
                "Luxury  (> KSh 50M/acre)":    999_999_999,
            }
            tier = next(
                (label for label, cap in price_bands.items() if price_per_acre < cap),
                "Luxury"
            )
            badge_class = (
                "badge-good" if price_per_acre < 1_000_000
                else "badge-med" if price_per_acre < 10_000_000
                else "badge-low"
            )
            st.markdown(f'<span class="score-badge {badge_class}">{tier}</span>', unsafe_allow_html=True)
            st.caption(f"Log price per acre: {log_pred:.4f} (model raw output)")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.caption("Ensure model files are in the `models/` directory.")

# ── About section ──────────────────────────────────────────────
with st.expander("ℹ️ About this model"):
    st.markdown("""
    **Model:** Multi-Layer Perceptron (4 hidden layers: 128 → 64 → 32 → 1)

    **Target:** `log(price_per_acre)` in KSh — predictions are back-transformed with `exp()`

    **Features used:**
    - Plot size (log-transformed acres)
    - Distance to Nairobi CBD (km)
    - Distance to county town (km)
    - Distance to nearest water body (km, where applicable)
    - Reference city distance (km)
    - Geocode confidence (0.2 – 1.0)
    - Amenities Score (OSM gravity model, 0–100)
    - Accessibility Score (road + urban decay, 0–100)
    - Infrastructure Score (NLP from descriptions, 0–100)

    **Training split:** 70% train / 15% val / 15% test | StandardScaler applied on train set only

    **Data source:** Property24 Kenya listings — scraped and cleaned through a 9-phase pipeline
    """)
