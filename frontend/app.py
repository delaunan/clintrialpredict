import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
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
    page_title="ClinTrialPredict | Discovery Engine",
    page_icon="🧪",
    layout="wide",
)

# ==========================
# 1. SETUP & PATHS
# ==========================
CURRENT_DIR = Path(__file__).resolve().parent
DATA_PATH = CURRENT_DIR / "data" / "search_registry.csv"
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
ID_COL = "nct_id"

# ==========================
# GLOBAL STYLES
# ==========================
st.markdown(
    """
    <style>
        .main > div { max-width: 1300px; margin: 0 auto; }
        .stButton > button {
            background: #1f2a38; color: white; padding: 0.5rem 1.2rem;
            border-radius: 6px; font-size: 0.9rem; border: none; font-weight: 600;
        }
        h1 { font-size: 2.1rem !important; font-weight: 800 !important; color: #1f2a38; margin-bottom: 0.5rem; }
        .risk-box { padding: 15px; border-radius: 8px; margin: 10px 0; border: 1px solid rgba(0,0,0,0.05); }
        .risk-title { font-size: 1.2rem; font-weight: 800; margin-bottom: 4px; }
        .risk-desc { font-size: 0.95rem; opacity: 0.8; }
        div[data-testid="stExpander"] { border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 10px; }
        .metric-card { background: #f8fafc; padding: 10px; border-radius: 6px; border-left: 4px solid #1f2a38; }
        .result-count { font-size: 14px; color: #64748b; margin-bottom: 15px; }
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
    df = pd.read_csv(DATA_PATH)
    # Ensure start_year is integer for better display
    if 'start_year' in df.columns:
        df['start_year'] = pd.to_numeric(df['start_year'], errors='coerce').fillna(0).astype(int)
    return df

X_ALL = load_predict_data()

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

def reset_filters():
    for key in ["f_sponsor", "f_ta", "f_indication", "f_year", "f_segment", "f_correct"]:
        if key in st.session_state:
            st.session_state[key] = None
    if "selected_nct_id" in st.session_state:
        st.session_state.selected_nct_id = None

# ==========================
# SIDEBAR: FACETED SEARCH
# ==========================
with st.sidebar:
    st.image("https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304af6292564449a.svg", width=40)
    st.header("Discovery Filters")
    
    if st.button("🔄 Reset Portfolio"):
        reset_filters()
        st.rerun()

    st.markdown("---")
    
    # 1. Start with full data for calculating options
    filtered_df = X_ALL.copy()

    # [SPONSOR FACET]
    sponsor_opts = sorted(X_ALL["lead_sponsor_canonical"].dropna().unique().tolist())
    sel_sponsor = st.selectbox("Company / Sponsor", sponsor_opts, index=None, key="f_sponsor", placeholder="All Companies")
    if sel_sponsor:
        filtered_df = filtered_df[filtered_df["lead_sponsor_canonical"] == sel_sponsor]

    # [TA FACET]
    ta_opts = sorted(filtered_df["therapeutic_area"].dropna().unique().tolist())
    sel_ta = st.selectbox("Therapeutic Area", ta_opts, index=None, key="f_ta", placeholder="All Areas")
    if sel_ta:
        filtered_df = filtered_df[filtered_df["therapeutic_area"] == sel_ta]

    # [INDICATION FACET]
    ind_opts = sorted(filtered_df["gbd_indication_name"].dropna().unique().tolist())
    sel_ind = st.selectbox("Indication (GBD)", ind_opts, index=None, key="f_indication", placeholder="All Indications")
    if sel_ind:
        filtered_df = filtered_df[filtered_df["gbd_indication_name"] == sel_ind]

    # [YEAR FACET]
    year_opts = sorted(filtered_df["start_year"].unique().tolist(), reverse=True)
    if 0 in year_opts: year_opts.remove(0)
    sel_year = st.selectbox("Start Year", year_opts, index=None, key="f_year", placeholder="All Years")
    if sel_year:
        filtered_df = filtered_df[filtered_df["start_year"] == sel_year]

    # [DEMO FACETS]
    st.markdown("---")
    st.subheader("🛠️ Demo Controls")
    
    sel_segment = st.selectbox("Trial Segment", ["HISTORICAL", "ONGOING"], index=None, key="f_segment")
    if sel_segment:
        filtered_df = filtered_df[filtered_df["trial_segment"] == sel_segment]
    
    sel_correct = st.selectbox("Prediction Accuracy", [True, False], index=None, key="f_correct", format_func=lambda x: "Correct Predictions" if x else "Incorrect Predictions")
    if sel_correct is not None:
        filtered_df = filtered_df[filtered_df["is_correct"] == sel_correct]

    st.markdown("---")
    st.info(f"💡 Found **{len(filtered_df):,}** trials matching your current criteria.")

# ==========================
# MAIN UI: DISCOVERY TABLE
# ==========================
st.markdown("# 🧪 ClinTrialPredict <span style='font-size: 16px; font-weight: normal; color: #64748b;'>v01 Production</span>", unsafe_allow_html=True)

# Selection Logic
if "selected_nct_id" not in st.session_state:
    st.session_state.selected_nct_id = None

# If a trial is selected, show details. Otherwise, show the discovery list.
if st.session_state.selected_nct_id is None:
    st.markdown("### Portfolio Discovery")
    st.markdown(f"<div class='result-count'>Showing {len(filtered_df):,} of {len(X_ALL):,} trials</div>", unsafe_allow_html=True)
    
    # Pre-process for display
    display_cols = ["nct_id", "ui_search_label", "therapeutic_area", "phase", "start_year", "Clinical_Score", "is_correct"]
    display_df = filtered_df[display_cols].copy()
    display_df.columns = ["NCT ID", "Identity", "Area", "Phase", "Year", "Success Prob", "Accurate"]

    # Use st.dataframe with selection mode
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )

    if event and event.selection and event.selection.rows:
        selected_row_idx = event.selection.rows[0]
        st.session_state.selected_nct_id = display_df.iloc[selected_row_idx]["NCT ID"]
        st.rerun()

else:
    # --- DETAIL VIEW ---
    row = X_ALL[X_ALL[ID_COL] == st.session_state.selected_nct_id].iloc[0]
    
    col_header_1, col_header_2 = st.columns([4, 1])
    with col_header_1:
        st.markdown(f"### {row.get('ui_title', row.get('brief_title', 'Trial Detail'))}")
        st.caption(f"**ID:** {row[ID_COL]} | **Identity:** {row.get('ui_search_label', 'N/A')}")
    with col_header_2:
        if st.button("⬅️ Back to List"):
            st.session_state.selected_nct_id = None
            st.rerun()

    # ORGANIZED IDENTITY CARD (4 PILLARS)
    id_c1, id_c2, id_c3, id_c4 = st.columns(4)
    
    with id_c1:
        with st.expander("🌍 Therapeutic Context", expanded=True):
            st.write(f"**Area:** {row['therapeutic_area']}")
            st.write(f"**Indication:** {row.get('gbd_indication_name', 'N/A')}")
            st.write(f"**Agent:** {row.get('alpha_drug_name', 'N/A')}")

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
            st.write(f"**Sponsor:** {row.get('lead_sponsor_canonical', row.get('lead_sponsor', 'N/A'))}")
            st.write(f"**Sponsor Tier:** {row.get('sponsor_tier', 'N/A')}")
            st.write(f"**DMC:** {'Yes' if row.get('has_dmc') else 'No'}")
            st.write(f"**Sites:** {row.get('number_of_facilities', 'N/A')}")

    with id_c4:
        with st.expander("👥 Patient Profile", expanded=True):
            st.write(f"**Enrollment:** {row.get('enrollment', 'N/A')}")
            st.write(f"**Start Year:** {row.get('start_year', 'N/A')}")
            with st.expander("View Eligibility Criteria"):
                st.write(row.get('ui_criteria', 'No criteria text available.'))

    st.markdown("<br>", unsafe_allow_html=True)
    
    # SCORING ENGINE
    if st.button("🎯 Generate Success Forecast"):
        # Prepare payload for API
        row_dict = row.replace({np.nan: None}).to_dict()

        with st.spinner("Analyzing clinical signals via forensic engine..."):
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
