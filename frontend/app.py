import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import numpy as np
import pandas as pd
import streamlit as st
import requests

# IMPORT PLOTTING UTILS
from utils.plot import plot_success_gauge, plot_impact_bar, plot_treemap

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Clinical Trial Success Predictor",
    page_icon="🧪",
    layout="wide",
)

# ==========================
# 1. SETUP & PATHS
# ==========================
CURRENT_DIR = Path(__file__).resolve().parent
DATA_PATH = CURRENT_DIR / "data" / "search_registry.csv"

# *** SMART API URL ***
# Defaults to local if API_URL env var is not set (e.g. on GCloud)
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

ID_COL = "nct_id"

# ==========================
# GLOBAL STYLES
# ==========================
st.markdown(
    """
    <style>
        .main > div { max-width: 1200px; margin: 0 auto; }
        .stButton > button {
            background: #1f2a38; color: white; padding: 0.5rem 1.2rem;
            border-radius: 6px; font-size: 0.9rem; border: none; font-weight: 600;
        }
        h1 { font-size: 2.1rem !important; font-weight: 800 !important; color: #1f2a38; margin-bottom: 0.5rem; }
        h5 { color: #475569; font-weight: 700; margin-top: 1rem; }
        .risk-box { padding: 15px; border-radius: 8px; margin: 10px 0; border: 1px solid rgba(0,0,0,0.05); }
        .risk-title { font-size: 1.2rem; font-weight: 800; margin-bottom: 4px; }
        .risk-desc { font-size: 0.95rem; opacity: 0.8; }
        div[data-testid="stExpander"] { border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================
# LOADING DATA
# ==========================
@st.cache_data
def load_predict_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Search registry not found at {DATA_PATH}")
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)

X = load_predict_data()

# ==========================
# UI HELPER FUNCTIONS
# ==========================
def get_risk_tier(score: float):
    if score >= 75:
        return "Robust", "Trial exhibits strong success patterns across clinical pillars.", "#f0fdf4", "#166534"
    elif score >= 50:
        return "Good", "Trial aligns with historical success benchmarks.", "#eff6ff", "#1e40af"
    elif score >= 25:
        return "Watchlist", "Mixed signals detected; several risk drivers require mitigation.", "#fff7ed", "#9a3412"
    else:
        return "High Risk", "Significant attrition patterns identified. High vulnerability.", "#fde8e8", "#991b1b"

# ==========================
# UI: MAIN APPLICATION
# ==========================
st.markdown("# 🧪 ClinTrialPredict <span style='font-size: 16px; font-weight: normal; color: #64748b;'>v01 Production</span>", unsafe_allow_html=True)

if X.empty:
    st.stop()

# Dropdown
X["short_label"] = X[ID_COL].astype(str) + " — " + X["brief_title"].astype(str)
selected_label = st.selectbox("Search Trial Portfolio", X["short_label"].tolist(), index=None, placeholder="Enter NCT ID or Trial Title...", key="trial_select")

if selected_label:
    trial_id = selected_label.split(" — ")[0]
    row = X[X[ID_COL] == trial_id].iloc[0]

    st.markdown(f"### {row.get('ui_title', row['brief_title'])}")
    
    # ORGANIZED IDENTITY CARD (4 PILLARS)
    id_c1, id_c2, id_c3, id_c4 = st.columns(4)
    
    with id_c1:
        with st.expander("🌍 Therapeutic Context", expanded=True):
            st.write(f"**Area:** {row['therapeutic_area']}")
            st.write(f"**Indication:** {row.get('therapeutic_subgroup_name', 'N/A')}")
            st.write(f"**Agent:** {row.get('agent_category', 'N/A')}")

    with id_c2:
        with st.expander("🔬 Scientific Design", expanded=True):
            st.write(f"**Phase:** {row['phase']}")
            st.write(f"**Arms:** {row.get('number_of_arms', 'N/A')}")
            st.write(f"**Purpose:** {row.get('primary_purpose', 'N/A')}")
            with st.expander("View Title/Summary"):
                st.write(f"**Title:** {row.get('ui_title', 'N/A')}")
                st.write(f"**Summary:** {row.get('ui_summary', 'N/A')}")

    with id_c3:
        with st.expander("⚙️ Execution Framework", expanded=True):
            st.write(f"**Sponsor:** {row.get('lead_sponsor', 'N/A')}")
            st.write(f"**Sponsor Tier:** {row.get('sponsor_tier', 'N/A')}")
            st.write(f"**DMC:** {'Yes' if row.get('has_dmc') else 'No'}")
            st.write(f"**Sites:** {row.get('number_of_facilities', 'N/A')}")

    with id_c4:
        with st.expander("👥 Patient Profile", expanded=True):
            st.write(f"**Enrollment:** {row.get('enrollment', 'N/A')}")
            st.write(f"**Severity:** {'Severe' if row.get('is_severe') else 'Standard'}")
            with st.expander("View Eligibility Criteria"):
                st.write(row.get('ui_criteria', 'No criteria text available.'))

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Generate Success Forecast"):
        row_dict = row.replace({np.nan: None}).to_dict()

        with st.spinner("Analyzing clinical signals..."):
            try:
                response = requests.post(API_URL, json=row_dict)
                if response.status_code == 200:
                    result = response.json()
                    if "error" in result:
                        st.error(f"API Error: {result['error']}"); st.stop()

                    score = result.get('score', 0)
                    pillar_impacts = result.get('pillar_impacts', [])
                    subcat_impacts = result.get('subcat_impacts', [])

                    tier, desc, bg_color, t_color = get_risk_tier(score)

                    # DASHBOARD LAYOUT
                    st.markdown("---")
                    col_left, col_right = st.columns([1.0, 1.4])

                    with col_left:
                        st.markdown("##### Clinical Success Score")
                        st.plotly_chart(plot_success_gauge(score), use_container_width=True, config={'displayModeBar': False})

                        # Small Pastel Risk Box
                        st.markdown(f"""
                        <div class="risk-box" style="background-color: {bg_color}; color: {t_color}; border-color: {t_color}22;">
                            <div class="risk-title">{tier}</div>
                            <div class="risk-desc">{desc}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        if pillar_impacts:
                            st.markdown("##### Strategic Pillar Impact")
                            df_pillars = pd.DataFrame(pillar_impacts)
                            st.plotly_chart(plot_impact_bar(df_pillars), use_container_width=True, config={'displayModeBar': False})

                    with col_right:
                        if subcat_impacts:
                            st.markdown("##### Clinical Driver Decomposition")
                            st.plotly_chart(plot_treemap(subcat_impacts, pillar_impacts), use_container_width=True, config={'displayModeBar': False})

                else: st.error(f"API Connection Error: {response.status_code}")
            except Exception as e: st.error(f"System Error: {e}")