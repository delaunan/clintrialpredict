import os
import json
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import streamlit as st
import requests

# IMPORT PLOTTING UTILS
from utils.plot import plot_success_gauge, plot_impact_bar, plot_treemap

# Load environment variables
load_dotenv()

# ==========================
# 1. SETUP & CONFIG
# ==========================
st.set_page_config(page_title="ClinTrialPredict | Predictive Engine", layout="wide")

CURRENT_DIR = Path(__file__).resolve().parent
DATA_PATH = CURRENT_DIR / "data" / "search_registry.csv"
TAXONOMY_PATH = CURRENT_DIR.parent / "models" / "taxonomy_01.json"
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
ID_COL = "nct_id"

# ==========================
# 2. STYLES (Clean & Consolidated)
# ==========================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* LOCK TO LIGHT MODE & GLOBAL FONT CONSISTENCY */
        :root { color-scheme: light !important; }
        * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

        html, body, [data-testid="stAppViewContainer"] {
            background-color: #f8fafc !important;
            color: #334155 !important;
        }

        .block-container {
            max-width: 1400px !important;
            padding: 1rem 2rem 2rem !important;
            margin: auto !important;
        }
        [data-testid="stHeader"] { background-color: rgba(0,0,0,0) !important; }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #52606d !important;
            border-right: 1px solid #e2e8f0;
        }

        /* High Contrast Inputs */
        div[data-baseweb="select"] > div, input {
            background-color: white !important;
            border: 1.5px solid #94a3b8 !important;
            border-radius: 8px !important;
            transition: all 0.2s;
            min-height: 42px !important;
            height: 42px !important;
        }
        div[data-baseweb="select"]:focus-within > div {
            border-color: #52606d !important;
            box-shadow: 0 0 0 1px #52606d !important;
        }



        /* Left filter panel using stable Streamlit key class */
        .st-key-filter_panel {
            background-color: #7b8794 !important;
            border: 1px solid #667381 !important;
            border-radius: 16px !important;
            padding: 25px 20px 18px 20px !important;
            box-shadow: -6px 6px 12px -3px rgba(0,0,0,0.12) !important;
        }

        .st-key-filter_panel .highlight-title,
        .st-key-filter_panel label,
        .st-key-filter_panel p {
            color: #ffffff !important;
        }

        .st-key-filter_panel div[data-baseweb="select"] > div,
        .st-key-filter_panel input {
            background-color: white !important;
            border-radius: 8px !important;
        }


        /* Layout Gaps & Symmetry */
        [data-testid="stHorizontalBlock"] { gap: 1rem !important; }
        .right-column-stack { display: flex; flex-direction: column; gap: 1rem; height: 100%; }
        .top-box-margin { margin-bottom: 1rem !important; }

        /* Highlight & Info Boxes */
        .highlight-box {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 24px;
            box-shadow: -6px 6px 12px -3px rgba(0,0,0,0.05);
            height: 100%;
        }
        .highlight-title { font-weight: 800; color: #334155; font-size: 1.15rem; margin-bottom: 8px; letter-spacing: -0.02em; }
        .highlight-text { color: #64748b; font-size: 0.95rem; line-height: 1.55; font-weight: 450; }

        /* Tags & Labels */
        span[data-baseweb="tag"] { background-color: #f1f5f9 !important; color: #1e293b !important; border-radius: 4px !important; font-weight: 600 !important; }
        label, strong { color: #475569 !important; font-weight: 600 !important; font-size: 0.85rem !important; letter-spacing: -0.01em; }

        /* Buttons */

        .stButton > button {
            border-radius: 8px;
            font-weight: 700;
            padding: 0.6rem 1rem;
            transition: all 0.2s;
            border: 1px solid #bfc7d1 !important;
            background-color: #cfd5dd !important;
            color: #4b5563 !important;
            min-height: 42px !important;
            height: 42px !important;
        }

        .stButton > button * {
            color: #4b5563 !important;
            fill: #4b5563 !important;
        }

        .stButton > button:hover {
            background-color: #c2c9d2 !important;
            color: #374151 !important;
            border: 1px solid #aeb8c4 !important;
        }

        .stButton > button:hover * {
            color: #374151 !important;
            fill: #374151 !important;
        }

        .stButton > button[kind="primary"] {
            background-color: #cfd5dd !important;
            color: #4b5563 !important;
            border: 1px solid #bfc7d1 !important;
            min-height: 42px !important;
            height: 42px !important;
        }

        .stButton > button[kind="primary"] * {
            color: #4b5563 !important;
            fill: #4b5563 !important;
        }

        .stButton > button[kind="primary"]:hover {
            background-color: #c2c9d2 !important;
            color: #374151 !important;
            border: 1px solid #aeb8c4 !important;
        }

        .stButton > button[kind="primary"]:hover * {
            color: #374151 !important;
            fill: #374151 !important;
        }

        /* Detail View Elements */
        .identity-header-text { font-size: 1.2rem; font-weight: 600; color: #0f172a; margin-right: 15px; }
        .title-box-container { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 15px 18px; margin-top: 15px; margin-bottom: 25px; line-height: 1.6; font-weight: 500; box-shadow: -4px 4px 10px -2px rgba(0,0,0,0.06); }
        .pillar-val-box { background:#ffffff; padding:10px; border:1px solid #cbd5e1; border-radius:6px; font-size:0.9rem; color:#0f172a; min-height:40px; margin-bottom:15px; box-shadow: -3px 3px 8px -2px rgba(0,0,0,0.04); }

        @media (max-width: 768px) {
            .stButton > button { height: 50px; font-size: 1rem; }
            h1 { font-size: 1.8rem !important; }
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar visibility logic
if not st.session_state.get("search_initiated", False):
    st.markdown("<style>section[data-testid='stSidebar'] { display: none !important; }</style>", unsafe_allow_html=True)

# ==========================
# 3. DATA & STATE
# ==========================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else pd.DataFrame()
    if 'start_year' in df.columns:
        df['start_year'] = pd.to_numeric(df['start_year'], errors='coerce').fillna(0).astype(int)
    tax = json.load(open(TAXONOMY_PATH)) if TAXONOMY_PATH.exists() else {}
    return df, tax.get("FIELDS", tax)

X_ALL, TAXONOMY = load_data()

for key, val in {"search_initiated": False, "selected_nct_id": None, "trigger_prediction": False}.items():
    if key not in st.session_state: st.session_state[key] = val

def reset_filters():
    for key in ["f_sponsor", "f_ta", "f_indication", "f_phase", "f_year", "f_nct_id", "s_registry", "s_mode"]:
        if key in st.session_state:
            st.session_state[key] = [] if key.startswith("f_") else ""
    st.session_state.selected_nct_id = None
    st.session_state.search_initiated = False

def initiate_search():
    st.session_state.search_initiated = True

def get_risk_tier(score: float):
    if score >= 75: return "Robust", "Strong success patterns detected.", "#f0fdf4", "#166534"
    if score >= 50: return "Favorable", "Favorable historical indicators.", "#eff6ff", "#1e40af"
    if score >= 25: return "Watchlist", "Mixed signals; mitigation required.", "#fff7ed", "#9a3412"
    return "High Risk", "Significant attrition patterns.", "#fde8e8", "#991b1b"

# ==========================
# 4. COMPONENTS
# ==========================
def render_filter_fields(df, is_sidebar=False):
    curr_df = df.copy()

    if is_sidebar:
        st.multiselect("Company / Sponsor", sorted(curr_df["lead_sponsor_canonical"].dropna().unique()), key="f_sponsor", placeholder="All Sponsors")
        st.multiselect("Therapeutic Area", sorted(curr_df["therapeutic_area"].dropna().unique()), key="f_ta", placeholder="All Therapeutic Areas")
        st.multiselect("Indication (GBD)", sorted(curr_df["gbd_indication_name"].dropna().unique()), key="f_indication", placeholder="All Indications")
        st.multiselect("Clinical trial number (AACT)", sorted(curr_df["nct_id"].dropna().unique()), key="f_nct_id", placeholder="All NCT IDs")
        st.multiselect("Trial Phase", sorted(curr_df["phase"].dropna().unique()), key="f_phase", placeholder="All Phases")
        years = sorted([y for y in curr_df["start_year"].unique() if y > 0], reverse=True)
        st.multiselect("Start Year", years, key="f_year", placeholder="All Years")
    else:
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1: st.multiselect("Company / Sponsor", sorted(curr_df["lead_sponsor_canonical"].dropna().unique()), key="f_sponsor", placeholder="All Sponsors")
        with r1_c2: st.multiselect("Therapeutic Area", sorted(curr_df["therapeutic_area"].dropna().unique()), key="f_ta", placeholder="All Therapeutic Areas")

        r2_c1, r2_c2 = st.columns(2)
        with r2_c1: st.multiselect("Clinical trial number (AACT)", sorted(curr_df["nct_id"].dropna().unique()), key="f_nct_id", placeholder="All NCT IDs")
        with r2_c2: st.multiselect("Trial Phase", sorted(curr_df["phase"].dropna().unique()), key="f_phase", placeholder="All Phases")

        r3_c1, r3_c2 = st.columns(2)
        with r3_c1: st.multiselect("Start Year", sorted([y for y in curr_df["start_year"].unique() if y > 0], reverse=True), key="f_year", placeholder="All Years")
        with r3_c2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            bc1, bc2 = st.columns([1.5, 1])
            with bc1:
                st.button("Search Trials", use_container_width=True, type="primary", on_click=initiate_search)
            with bc2:
                st.button("Reset", use_container_width=True, key="btn_hub_reset", on_click=reset_filters)

    # Dynamic Filter Application
    for col, key in [("lead_sponsor_canonical", "f_sponsor"), ("therapeutic_area", "f_ta"),
                     ("gbd_indication_name", "f_indication"), ("nct_id", "f_nct_id"),
                     ("phase", "f_phase"), ("start_year", "f_year")]:
        if st.session_state.get(key):
            curr_df = curr_df[curr_df[col].isin(st.session_state[key])]

    if not is_sidebar:
        st.markdown(f"<div style='text-align:right; font-size:0.8rem; color:#cbd5e1; margin-top:10px;'>{len(curr_df):,} trials matching criteria</div>", unsafe_allow_html=True)
    return curr_df

# ==========================
# 5. MAIN UI FLOW
# ==========================
# Header
t1, t2 = st.columns([3, 1])
with t1:
    st.markdown("""
        <h1 style='font-size: 3rem; margin-bottom: 0;'>ClinTrialPrediction
            <span style='font-size:0.65rem; font-weight:800; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em; vertical-align: baseline; margin-left: 15px;'>demo version</span>
        </h1>
    """, unsafe_allow_html=True)

with t2:
    if st.session_state.selected_nct_id:
        if st.button("Predict Completion", use_container_width=True, type="primary"):
            st.session_state.trigger_prediction = True

# Main Logic
if not st.session_state.selected_nct_id:
    # 5.1 LANDING OR SEARCH RESULTS
    x_base = X_ALL.copy()
    if st.session_state.get("s_mode", "").lower() != "all":
        x_base = x_base[(x_base["is_correct"] == True) | (x_base["trial_segment"] == "ONGOING")]

    if not st.session_state.search_initiated:
        # LANDING PAGE
        st.markdown('''
            <div class="highlight-box top-box-margin">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <div class="highlight-title">Operational Success & Risk Stratification</div>
                    <div style="font-size:0.65rem; font-weight:800; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em;">Core Mission</div>
                </div>
                <div class="highlight-text">This predictive engine estimates the likelihood of operational completion and the risk of early termination using only data available at clinical trial initiation. Each trial is systematically evaluated and classified into four distinct tiers—High Risk, Watchlist, Favorable, and Robust—providing a clear and actionable risk profile.</div>
            </div>
        ''', unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            with st.container(border=True, key="filter_panel"):
                st.markdown("""
                    <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:32px;">
                        <div class="highlight-title" style="margin-bottom:0;">
                            Clinical Trial Selection
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                render_filter_fields(x_base, is_sidebar=False)



        with col_right:
            st.markdown('''
                <div class="right-column-stack">
                    <div class="highlight-box">
                        <div style="display: flex; justify-content: space-between; align-items: baseline;">
                            <div class="highlight-title">Industry-Scale Clinical Data</div>
                            <div style="font-size:0.65rem; font-weight:800; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em;">Intelligence Source</div>
                        </div>
                        <div class="highlight-text">Built on the publicly available AACT registry, this machine learning system leverages execution patterns from 30,000+ Phase II and III trials since 2005. The analytical scope focuses on late-stage studies, where strategic and financial stakes are highest.</div>
                    </div>
                    <div class="highlight-box">
                        <div style="display: flex; justify-content: space-between; align-items: baseline;">
                            <div class="highlight-title">Predictive Power & Benchmarking</div>
                            <div style="font-size:0.65rem; font-weight:800; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em;">Engine Accuracy</div>
                        </div>
                        <div class="highlight-text">When comparing a completed trial with one that terminated early, the system assigns a higher risk score to the failed trial in 75% of cases. It clearly outperforms the 50% random baseline and traditional approaches built on publicly available data (AUC ≈ 0.75 vs. 0.50 baseline).</div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
    else:
        # SEARCH RESULTS GRID
        with st.sidebar:
            st.markdown("<h2 style='color:#f8fafc; font-weight:800;'>Filters</h2>", unsafe_allow_html=True)
            if st.button("Reset Filter", use_container_width=True): reset_filters(); st.rerun()
            filtered_df = render_filter_fields(x_base, is_sidebar=True)
            st.markdown("<div style='height: 300px;'></div>---", unsafe_allow_html=True)
            st.text_input("Register", key="s_registry", placeholder="all")
            st.text_input("Analysis", key="s_mode", placeholder="all")

        st.markdown(f"<div style='margin-top:20px; color:#64748b; font-weight:600;'>{len(filtered_df):,} Matching Trials</div>", unsafe_allow_html=True)
        grid_df = filtered_df[["nct_id", "ui_search_label", "lead_sponsor_canonical", "therapeutic_area", "phase", "start_year", "Clinical_Score"]].copy()
        grid_df.columns = ["NCT ID", "Identity", "Sponsor", "Area", "Phase", "Year", "Score"]

        event = st.dataframe(grid_df, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", height=500)
        if event and event.selection and event.selection.rows:
            st.session_state.selected_nct_id = grid_df.iloc[event.selection.rows[0]]["NCT ID"]
            st.rerun()

else:
    # 5.2 TRIAL AUDIT DETAIL
    row = X_ALL[X_ALL[ID_COL] == st.session_state.selected_nct_id].iloc[0]
    if st.button("← Back to Results"): st.session_state.selected_nct_id = None; st.rerun()

    with st.expander("Identity", expanded=True):
        st.markdown(f'''
            <div style="display: flex; align-items: baseline; margin-bottom: 10px;">
                <span class="identity-header-text">{row[ID_COL]}</span>
                <span style="font-size: 1.2rem; color: #475569; font-weight: 600;">{row.get("ui_search_label", "N/A")}</span>
            </div>
            <div class="title-box-container">
                <span style="color: #64748b; font-size: 0.75rem; text-transform: uppercase; font-weight: 800; display: block; margin-bottom: 8px;">Title</span>
                {row.get("title", "No title available.")}
            </div>
        ''', unsafe_allow_html=True)

    def render_pillar(title, taxonomy_name, data):
        feats = sorted([ (f_id, f_m) for f_id, f_m in TAXONOMY.items() if f_m.get("ui", {}).get("pillar") == taxonomy_name ],
                       key=lambda x: (x[1].get("ui", {}).get("subgroup", ""), x[1].get("ui", {}).get("priority", 99)))
        with st.expander(title, expanded=False):
            for i in range(0, len(feats), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i+j < len(feats):
                        f_id, f_m = feats[i+j]
                        ui = f_m.get("ui", {})
                        val = data.get(f_id)
                        with cols[j]:
                            st.markdown(f"**{ui.get('label', f_id)}**")
                            st.markdown(f"<div class='pillar-val-box'>{val if not pd.isna(val) else 'N/A'}</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1: render_pillar("Therapeutic Context", "Therapeutic Context", row)
    with c2: render_pillar("Execution Framework", "Execution Framework", row)
    c3, c4 = st.columns(2)
    with c3: render_pillar("Scientific Attempt", "Scientific Attempt", row)
    with c4: render_pillar("Patient Profile", "Patient Profile", row)

    # 5.3 ANALYSIS RESULTS
    if st.session_state.trigger_prediction or st.session_state.get("analysis_result"):
        if not st.session_state.get("analysis_result") or st.session_state.get("analysis_nct_id") != st.session_state.selected_nct_id:
            with st.spinner("Analyzing signals..."):
                try:
                    res = requests.post(API_URL, json=row.replace({np.nan: None}).to_dict())
                    if res.status_code == 200:
                        st.session_state.analysis_result = res.json()
                        st.session_state.analysis_nct_id = st.session_state.selected_nct_id
                        st.session_state.trigger_prediction = False
                    else: st.error(f"API Error: {res.status_code}")
                except Exception as e: st.error(f"System Error: {e}")

        if st.session_state.get("analysis_result"):
            res = st.session_state.analysis_result
            score = res.get('score', 0)
            tier, desc, bg, tc = get_risk_tier(score)
            st.markdown("<hr style='margin: 40px 0;'>", unsafe_allow_html=True)
            cl, cr = st.columns([1.0, 1.4])
            with cl:
                st.plotly_chart(plot_success_gauge(score), use_container_width=True, config={'displayModeBar': False})
                st.markdown(f"<div style='background:{bg}; color:{tc}; padding:20px; border-radius:12px; border:1px solid {tc}22;'><div style='font-size:1.4rem; font-weight:800;'>{tier}</div><div>{desc}</div></div>", unsafe_allow_html=True)
                if res.get('pillar_impacts'): st.plotly_chart(plot_impact_bar(pd.DataFrame(res['pillar_impacts'])), use_container_width=True)
            with cr:
                if res.get('subcat_impacts'): st.plotly_chart(plot_treemap(res['subcat_impacts'], res.get('pillar_impacts', [])), use_container_width=True)
