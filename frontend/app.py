import sys
from pathlib import Path
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
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "search_registry.csv"

# *** API URL ***
API_URL = "https://clintrialpredict-835962039082.europe-west1.run.app/predict"

ID_COL = "nct_id"

# ==========================
# GLOBAL STYLES
# ==========================
st.markdown(
    """
    <style>
        .main > div { max-width: 1200px; margin: 0 auto; }
        .stButton > button {
            background: #1f2a38; color: white; padding: 0.55rem 1.2rem;
            border-radius: 8px; font-size: 0.9rem; border: none; font-weight: 500;
        }
        .stButton > button:hover {
            background: #2f3e50; transform: translateY(-1px);
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.2);
        }
        h1 { font-size: 2.3rem !important; font-weight: 800 !important; letter-spacing: -0.03em; color: #1f2a38; }
        h4 { color: #1f2a38; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================
# LOADING DATA (UI ONLY)
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
        return "Robust", "Trial exhibits strong success patterns across most clinical pillars.", "#1C5699"
    elif score >= 50:
        return "Good", "Trial aligns with historical success benchmarks, showing manageable risk.", "#9ACBE8"
    elif score >= 25:
        return "Watchlist", "Mixed signals detected; several risk drivers require mitigation.", "#F0A3A3"
    else:
        return "High Risk", "Significant attrition patterns identified. High operational vulnerability.", "#A83232"

# ==========================
# UI: MAIN APPLICATION
# ==========================
st.markdown("# 🧪 ClinTrialPredict <span style='font-size: 18px; font-weight: normal; color: #666;'>v01 Production</span>", unsafe_allow_html=True)

if X.empty:
    st.stop()

# Build display labels
X["short_label"] = X[ID_COL].astype(str) + " — " + X["brief_title"].astype(str)
all_labels = X["short_label"].tolist()
label_to_nct = dict(zip(X["short_label"], X[ID_COL]))

# --- Dropdown ---
selected_label = st.selectbox("Search Trial Portfolio (NCT ID or Title)", all_labels, index=None, placeholder="Select a trial…", key="trial_select")

# Only show details AFTER a trial is selected
if selected_label is not None:
    trial_id = label_to_nct[selected_label]
    selected_trial = X[X[ID_COL] == trial_id].iloc[[0]]
    row = selected_trial.iloc[0]

    # --- TRIAL IDENTITY CARD ---
    st.markdown("#### Clinical Profile")
    st.markdown(f"""
    **{row.get('ui_title', row['brief_title'])}**
    - **NCT ID:** {row['nct_id']}
    - **Phase:** {row['phase']}
    - **Therapeutic Area:** {row['therapeutic_area']}
    """)

    # Grid for trial details
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(f"**Indication:**\n{row.get('therapeutic_subgroup_name', 'N/A')}")
    with c2:
        st.info(f"**Lead Sponsor:**\n{row.get('lead_sponsor', 'N/A')}")
    with c3:
        st.info(f"**Agent Category:**\n{row.get('agent_category', 'N/A')}")

    st.markdown("<br>", unsafe_allow_html=True)
    run_prediction = st.button("Generate Success Forecast")

    # ==========================
    # PREDICTION DASHBOARD
    # ==========================
    if run_prediction:
        # Prepare Data (Handle NaNs for JSON)
        row_dict = selected_trial.iloc[0].replace({np.nan: None}).to_dict()

        with st.spinner("Analyzing clinical signals..."):
            try:
                response = requests.post(API_URL, json=row_dict)

                if response.status_code == 200:
                    result = response.json()
                    
                    if "error" in result:
                        st.error(f"API Logic Error: {result['error']}")
                        st.stop()

                    score = result.get('score', 0)
                    pillar_impacts = result.get('pillar_impacts', [])
                    subcat_impacts = result.get('subcat_impacts', [])

                    # Risk Tier
                    tier, desc, t_color = get_risk_tier(score)
                    font_color = 'white' if (score < 50 or score > 75) else '#1f2a38'

                    # Draw Dashboard
                    st.markdown("#### Success Forecast Dashboard")
                    st.markdown("---")

                    config = {'displayModeBar': False}

                    col1, col2 = st.columns([1.0, 1.3])

                    with col1:
                        st.markdown(f"##### Clinical Success Score", help="A score of 50.0 represents the calibrated decision boundary for the specific Therapeutic Area.")
                        st.plotly_chart(plot_success_gauge(score), use_container_width=True, config=config)

                        st.markdown(f"""
                        <div style="background-color: {t_color}; padding: 20px; border-radius: 10px; color: {font_color};">
                            <h3 style="margin:0; color: inherit;">{tier} Zone</h3>
                            <p style="margin:0; font-size: 1.1rem; opacity: 0.9;">{desc}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        if pillar_impacts:
                            st.markdown("<br>##### Strategic Pillar Impact", unsafe_allow_html=True)
                            df_pillars = pd.DataFrame(pillar_impacts)
                            st.plotly_chart(plot_impact_bar(df_pillars), use_container_width=True, config=config)

                    with col2:
                        if subcat_impacts:
                            st.markdown("##### Clinical Driver Decomposition", help="Size represents the absolute magnitude of impact; Color represents direction (Success vs Risk).")
                            st.plotly_chart(plot_treemap(subcat_impacts, pillar_impacts), use_container_width=True, config=config)
                        else:
                            st.info("Visual explanations not available.")

                else:
                    st.error(f"API Error {response.status_code}: {response.text}")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                import traceback
                st.code(traceback.format_exc())