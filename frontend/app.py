import os
import json
import base64
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
st.set_page_config(
    page_title="ClinTrialPredict | Predictive Engine",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

        /* MATERIAL SYMBOLS FIX */
        .material-symbols-outlined {
            font-family: 'Material Symbols Outlined' !important;
            font-weight: normal;
            font-style: normal;
            font-size: 24px;
            line-height: 1;
            letter-spacing: normal;
            text-transform: none;
            display: inline-block;
            white-space: nowrap;
            word-wrap: normal;
            direction: ltr;
            -webkit-font-smoothing: antialiased;
        }

        /* LOCK TO LIGHT MODE & GLOBAL FONT CONSISTENCY */
        :root { color-scheme: light !important; }

        /* Target main text containers without breaking icon fonts */
        html, body, [data-testid="stAppViewContainer"], .stMarkdown, p, span, label, div {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background-color: #f8fafc !important;
            color: #334155 !important;
        }

        .block-container {
            max-width: 1400px !important;
            padding: 0rem 2rem 2rem !important; margin: auto !important;
        }

        /* Header Styling - Allow toggle button to show and stay visible */
        [data-testid="stHeader"] {
            background-color: rgba(0,0,0,0) !important;
            color: #334155 !important;
        }

        /* FORCE SIDEBAR TOGGLE BUTTONS VISIBILITY (No Hover Required) */
        button[kind="header"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="collapsedControl"] {
            opacity: 1 !important;
            visibility: visible !important;
        }

        /* Sidebar Styling - Balanced Mid-tone (#717d8b) */
        section[data-testid="stSidebar"] {
            background-color: #717d8b !important;
            border-right: 1px solid #606c7a;
        }

        /* High Contrast Inputs - Flexible Height for multi-select tags */
        div[data-baseweb="select"] > div, input {
            background-color: white !important;
            border: 1.5px solid #94a3b8 !important;
            border-radius: 8px !important;
            transition: all 0.2s;
            min-height: 36px !important;
            height: auto !important;
            font-size: 0.85rem !important;
        }
        div[data-baseweb="select"]:focus-within > div {
            border-color: #52606d !important;
            box-shadow: 0 0 0 1px #52606d !important;
        }

        /* GLOBAL MULTISELECT DROPDOWN ALIGNMENT */
        [data-baseweb="popover"] li,
        div[data-baseweb="select"] ul li,
        div[role="listbox"] li {
            font-size: 0.85rem !important;
        }

        /* --- HARMONIOUS SEPARATED SEARCH PANEL --- */

        /* Header Box (Top) - Matched to Mid-tone */
        .st-key-filter_header {
            background-color: #717d8b !important;
            border: 1px solid #606c7a !important;
            border-radius: 14px !important;
            padding: 22px 20px 36px 20px !important;
            box-shadow: -6px 6px 12px -3px rgba(0,0,0,0.12) !important;
            margin-bottom: 0px !important;
        }

        /* Body Box (Bottom) - Matched to Mid-tone */
        .st-key-filter_body {
            background-color: #717d8b !important;
            border: 1px solid #606c7a !important;
            border-radius: 14px !important;
            padding: 34px 25px 35px 25px !important; /* TUNED: Large internal margins for Top/Bottom room */
            box-shadow: -6px 6px 12px -3px rgba(0,0,0,0.12) !important;
            margin-bottom: 4px !important;
        }

        /* NUCLEAR COMPRESSION: Pull rows closer by force */
        .st-key-filter_body [data-testid="stVerticalBlock"] > div {
            margin-bottom: -6px !important;
            padding-bottom: 0px !important;
        }

        /* Standardize text colors for dark panels - Force White for Dropdown Labels */
        .st-key-filter_header .highlight-title,
        .st-key-filter_body label,
        .st-key-filter_body p,
        .st-key-filter_body div[data-testid="stMarkdownContainer"] p,
        .st-key-filter_body [data-testid="stWidgetLabel"] p {
            color: #ffffff !important;
            font-weight: 600 !important;
            margin-bottom: 01px !important; /* TUNED: Snap text to input box */
        }

        .st-key-filter_body div[data-baseweb="select"] > div,
        .st-key-filter_body input {
            background-color: white !important;
            color: #334155 !important;
            border: 1.5px solid #cbd5e1 !important;
            border-radius: 8px !important;
            font-size: 0.85rem !important;
        }

        .st-key-filter_body input::placeholder {
            color: #94a3b8 !important;
            font-size: 0.8rem !important;
        }

        /* COMPACT IDENTICAL DISTANCE BETWEEN ALL LINES */
        .st-key-filter_body [data-testid="stVerticalBlock"] {
            gap: 0rem !important;
        }

        /* BRING LABELS CLOSER TO INPUTS */
        .st-key-filter_body [data-testid="stWidgetLabel"] {
            min-height: 0px !important;
            margin-bottom: 0px !important;
        }

        /* Layout Gaps & Symmetry */
        [data-testid="stHorizontalBlock"] { gap: 1rem !important; }
        .right-column-stack { display: flex; flex-direction: column; gap: 1rem; height: 100%; }
        .top-box-margin { margin-top: 1.5rem !important; margin-bottom: 1rem !important; }

        /* Highlight & Info Boxes */
        .highlight-box {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 24px;
            box-shadow: -6px 6px 12px -3px rgba(0,0,0,0.12) !important;
            height: 100%;
        }

        /* Box 1 Tint (Clean, No thick left line) */
        .mission-box {
            background-color: #e2e8f0 !important;
            border: 1px solid #cbd5e1 !important;
        }
        .mission-box .highlight-text {
            color: #52606d !important;
        }

        .highlight-title { font-weight: 800; color: #52606d !important; font-size: 1.15rem; margin-bottom: 8px; letter-spacing: -0.02em; }
        .highlight-text { color: #64748b; font-size: 0.95rem; line-height: 1.55; font-weight: 450; }

        /* FORCE BOLD VISIBILITY */
        .highlight-text b, .highlight-text strong {
            font-weight: 700 !important;
            color: inherit !important;
        }

        /* Tags & Labels */
        span[data-baseweb="tag"] { background-color: #f1f5f9 !important; color: #334155 !important; border-radius: 4px !important; font-weight: 600 !important; font-size: 0.75rem !important; }
        label, strong { color: #475569 !important; font-weight: 600 !important; font-size: 0.85rem !important; letter-spacing: -0.01em; }

        /* --- RESTOREST BUTTON STYLES --- */
        .stButton > button {
            border-radius: 8px !important;
            font-weight: 700 !important;
            padding: 0px 1rem !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            border: 1.5px solid #99a7b9 !important;
            background-color: #b2bccb !important;
            color: #ffffff !important;
            min-height: 36px !important;
            height: 36px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        .stButton > button * {
            color: #ffffff !important;
            fill: #ffffff !important;
        }

        .stButton > button:hover {
            background-color: #334155 !important;
            border-color: #1e293b !important;
            box-shadow: 0 8px 20px rgba(0,0,0,0.2) !important;
            transform: scale(1.02) translateY(-2px) !important;
            color: #ffffff !important;
        }

        .stButton > button:active {
            transform: scale(0.98) translateY(0px) !important;
        }

        .stButton > button:hover * {
            color: #ffffff !important;
            fill: #ffffff !important;
        }

        /* Detail View Elements */
        h1 { color: #334155 !important; }
        .identity-header-text { font-size: 1.2rem; font-weight: 600; color: #334155 !important; margin-right: 15px; }
        .title-box-container { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 15px 18px; margin-top: 15px; margin-bottom: 25px; line-height: 1.6; font-weight: 500; box-shadow: -6px 6px 12px -3px rgba(0,0,0,0.12) !important; }
        .pillar-val-box { background:#ffffff; padding:10px; border:1px solid #cbd5e1; border-radius:6px; font-size:0.9rem; color:#334155 !important; min-height:40px; margin-bottom:15px; box-shadow: -6px 6px 12px -3px rgba(0,0,0,0.12) !important; }

        @media (max-width: 768px) {
            .stButton > button { height: 50px; font-size: 1rem; }
            h1 { font-size: 1.8rem !important; }
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar visibility logic
# Removed restrictive 'display: none' to allow manual expansion/collapse

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
    for key in ["f_sponsor", "f_ta", "f_phase", "f_year", "f_nct_id", "s_registry", "s_mode"]:
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
    # Dynamic Options Calculation (Interdependent Filtering)
    def get_opts(col, key):
        tdf = df.copy()
        # Filter tdf by ALL OTHER selections
        for c, k in [("lead_sponsor_canonical", "f_sponsor"), ("therapeutic_area", "f_ta"),
                     ("phase", "f_phase"), ("start_year", "f_year"), ("nct_id", "f_nct_id")]:
            if k == key: continue
            val = st.session_state.get(k)
            if val:
                tdf = tdf[tdf[c].isin(val)]
        if col == "start_year":
            return sorted([y for y in tdf[col].unique() if y > 0], reverse=True)
        return sorted(tdf[col].dropna().unique())

    if is_sidebar:
        st.multiselect("Company / Sponsor", get_opts("lead_sponsor_canonical", "f_sponsor"), key="f_sponsor", placeholder="All Sponsors")
        st.multiselect("Therapeutic Area", get_opts("therapeutic_area", "f_ta"), key="f_ta", placeholder="All Therapeutic Areas")
        st.multiselect("Trial Phase", get_opts("phase", "f_phase"), key="f_phase", placeholder="All Phases")
        st.multiselect("Start Year", get_opts("start_year", "f_year"), key="f_year", placeholder="All Years")
        st.multiselect("Clinical trial number (AACT)", get_opts("nct_id", "f_nct_id"), key="f_nct_id", placeholder="All NCT IDs")
    else:
        # Line 1: Company / Sponsor, Therapeutic Area
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1: st.multiselect("Company / Sponsor", get_opts("lead_sponsor_canonical", "f_sponsor"), key="f_sponsor", placeholder="All Sponsors")
        with r1_c2: st.multiselect("Therapeutic Area", get_opts("therapeutic_area", "f_ta"), key="f_ta", placeholder="All Therapeutic Areas")

        # Line 2: Trial Phase, Start Year
        r2_c1, r2_c2 = st.columns(2)
        with r2_c1: st.multiselect("Trial Phase", get_opts("phase", "f_phase"), key="f_phase", placeholder="All Phases")
        with r2_c2: st.multiselect("Start Year", get_opts("start_year", "f_year"), key="f_year", placeholder="All Years")

        # Line 3: Clinical trial number (AACT), Buttons
        r3_c1, r3_c2, r3_c3 = st.columns([2, 0.6, 1.4], vertical_alignment="bottom")
        with r3_c1: st.multiselect("Clinical trial number (AACT)", get_opts("nct_id", "f_nct_id"), key="f_nct_id", placeholder="All NCT IDs")
        with r3_c2: st.button("Reset", use_container_width=True, key="btn_hub_reset", on_click=reset_filters)
        with r3_c3: st.button("Search Trials", use_container_width=True, type="primary", on_click=initiate_search)

    # Dynamic Filter Application (for the final returned dataframe)
    curr_df = df.copy()
    for col, key in [("lead_sponsor_canonical", "f_sponsor"), ("therapeutic_area", "f_ta"),
                     ("phase", "f_phase"), ("start_year", "f_year"), ("nct_id", "f_nct_id")]:
        if st.session_state.get(key):
            curr_df = curr_df[curr_df[col].isin(st.session_state[key])]

    if not is_sidebar:
        st.markdown(f"<div style='text-align:right; font-size:0.8rem; color:#cbd5e1; margin-top: 4px; margin-bottom: -16px;'>{len(curr_df):,} trials matching criteria</div>", unsafe_allow_html=True)
    return curr_df

# ==========================
# 5. MAIN UI FLOW
# ==========================

# Helper to load image as base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Header
t1, t2 = st.columns([3, 1])
is_landing = not st.session_state.get("search_initiated", False) and not st.session_state.get("selected_nct_id")

with t1:
    logo_path = CURRENT_DIR / "logo_grey_title.png"
    img_base64 = ""
    if logo_path.exists():
        img_base64 = get_base64_image(logo_path)

    # --- FINAL SETTINGS ---
    HUE = 180
    INTENSITY = 0.8
    DARKNESS = 0.85
    THICKNESS = 0  # Removed to eliminate artificial borders
    # ----------------------

    # 1. We harden the outlines first to make them feel thicker/more solid
    harden = "contrast(1.5) brightness(0.9)"
    # 2. Apply your tuned slate-blue tint
    tint = f"grayscale(100%) sepia(100%) hue-rotate({HUE}deg) saturate({INTENSITY}) brightness({DARKNESS}) contrast(1.2)"
    # 3. Add the exact title color as a sharp shadow for the final "weight"
    shadows = f"drop-shadow({THICKNESS}px {THICKNESS}px 0px #52606d) drop-shadow(-{THICKNESS}px -{THICKNESS}px 0px #52606d)"

    brand_filter = f"{harden} {tint} {shadows}"

    # UNIFIED HEADER LAYOUT
    if is_landing:
        header_html = f"""
            <div style='display: flex; align-items: center; gap: 12px; margin-top: 15px; margin-left: 0px;'>
                <div style='background-color: white; border: 4px solid #52606d; margin-top: 12px; padding: 2px; border-radius: 18px; display: flex; align-items: center; justify-content: center; height: 72px; width: 72px; flex-shrink: 0; box-shadow: 0 4px 12px rgba(0,0,0,0.05); position: relative;'>
                    <img src='data:image/png;base64,{img_base64}' style='height: 70px; filter: {brand_filter}; border: none; outline: none;'>
                </div>
                <div>
                    <div style='font-size: 2.8rem; font-weight: 800; color: #52606d; line-height: 1; margin-top: 10px;'>CTPredict</div>
                    <div style='color: #52606d; white-space: nowrap; font-size: 1.5rem; font-weight: 800; display: flex; align-items: baseline; gap: 15px; margin-top: 5px;'>
                        <span style='line-height: 1;'>Late-Stage Clinical Trial Predictive Engine</span>
                        <span style='font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; line-height: 1; vertical-align: baseline;'>demo version</span>
                    </div>
                </div>
            </div>
        """
    else:
        header_html = f"""
            <div style='display: flex; align-items: center; gap: 10px; margin-top: 15px; margin-left: 0px;'>
                <div style='background-color: white; border: 2px solid #52606d; padding: 0px; border-radius: 7px; display: flex; align-items: center; justify-content: center; height: 44px; width: 44px; flex-shrink: 0; box-shadow: 0 4px 12px rgba(0,0,0,0.05); position: relative;'>
                    <img src='data:image/png;base64,{img_base64}' style='height: 40px; filter: {brand_filter}; border: none; outline: none;'>
                </div>
                <div style='display: flex; align-items: baseline; gap: 15px;'>
                    <div style='font-size: 3.2rem; font-weight: 800; color: #52606d; line-height: 1;'>CTPredict</div>
                    <span style='font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; line-height: 1; vertical-align: baseline;'>Demo Version</span>
                </div>
            </div>
        """
    st.markdown(header_html, unsafe_allow_html=True)

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
            <div class="highlight-box mission-box top-box-margin">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <div class="highlight-title">Operational Success & Risk Stratification</div>
                    <div style="font-size:0.65rem; font-weight:800; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em;">Core Mission</div>
                </div>
                <div class="highlight-text">This predictive engine estimates the <b>likelihood of operational completion</b> and the <b>risk of early termination</b> using only data available at clinical trial initiation. Each trial is systematically evaluated and classified into <b>four distinct tiers</b> - High Risk, Watchlist, Favorable, and Robust - providing a clear and actionable risk profile.</div>
            </div>
        ''', unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            # HEADER BOX
            with st.container(key="filter_header"):
                st.markdown("""
                    <div class="highlight-title" style="margin-bottom:0; color:white;">
                        Clinical Trial Selection
                    </div>
                """, unsafe_allow_html=True)

            # BODY BOX
            with st.container(key="filter_body"):
                render_filter_fields(x_base, is_sidebar=False)

        with col_right:
            st.markdown('''
                <div class="right-column-stack">
                    <div class="highlight-box">
                        <div style="display: flex; justify-content: space-between; align-items: baseline;">
                            <div class="highlight-title">Industry-Scale Clinical Data</div>
                            <div style="font-size:0.65rem; font-weight:800; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em;">Intelligence Source</div>
                        </div>
                        <div class="highlight-text">Built on the publicly available <b>AACT registry</b>, this machine learning system leverages execution patterns from <b>30,000+ Phase II and III trials</b> since 2005. The analytical scope focuses on <b>late-stage studies</b>, where strategic and financial stakes are highest.</div>
                    </div>
                    <div class="highlight-box">
                        <div style="display: flex; justify-content: space-between; align-items: baseline;">
                            <div class="highlight-title">Predictive Power & Benchmarking</div>
                            <div style="font-size:0.65rem; font-weight:800; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em;">Engine Accuracy</div>
                        </div>
                        <div class="highlight-text">When comparing a completed trial with one that terminated early, the system assigns a <b>higher risk score</b> to the failed trial in <b>75% of cases</b>. It outperforms the 50% random baseline and traditional approaches built on publicly available data (<b>ROC AUC ≈ 0.75</b> vs. 0.50 baseline).</div>
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
            st.text_input("Register", key="s_registry", placeholder="")
            st.text_input("Analysis", key="s_mode", placeholder="")

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
