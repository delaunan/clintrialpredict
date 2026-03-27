import os
import sys
import json
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
    page_title="ClinTrialPredict | Forensic Engine",
    layout="wide",
)

# ==========================
# 1. SETUP & PATHS
# ==========================
CURRENT_DIR = Path(__file__).resolve().parent
DATA_PATH = CURRENT_DIR / "data" / "search_registry.csv"
TAXONOMY_PATH = CURRENT_DIR.parent / "models" / "taxonomy_01.json"
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
ID_COL = "nct_id"

# ==========================
# GLOBAL STYLES
# ==========================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        /* Layout Space Optimization - Absolute Top Start */
        .block-container { padding-top: 0rem !important; padding-bottom: 0rem !important; }
        
        /* Sidebar Padding Overrides - Absolute Top Start */
        [data-testid="stSidebarUserContent"] {
            padding-top: 0rem !important;
        }
        
        /* Keep header for sidebar toggle but hide extra decoration */
        [data-testid="stDecoration"] { display: none !important; }
        [data-testid="stHeader"] { background-color: rgba(0,0,0,0) !important; }
        
        html, body, [class*="css"], .stMarkdown {
            font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }
        
        .main > div { max-width: 1400px; margin: 0 auto; }
        
        /* Sidebar Styling - Balanced Slate Grey (#52606d derivation) */
        section[data-testid="stSidebar"] {
            background-color: #52606d !important;
            border-right: 1px solid #d1d5db;
        }
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] label {
            color: #f8fafc !important;
            font-weight: 500;
        }
        
        /* Sidebar Inputs - Force White Background */
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] input {
            background-color: white !important;
            color: #1e293b !important;
            border: 1px solid #cbd5e1 !important;
        }

        /* Multiselect Tag Color */
        span[data-baseweb="tag"] {
            background-color: #f1f5f9 !important;
            color: #1e293b !important;
            border-radius: 4px !important;
            font-weight: 600;
        }

        /* Expander Headers Styling - Darker Derived Greys */
        div[data-testid="stExpander"] {
            border: none !important;
            border-radius: 8px !important;
            margin-bottom: 15px;
            overflow: hidden;
            box-shadow: none !important;
        }
        
        div[data-testid="stExpander"] summary {
            background-color: #c1c9d2 !important;
            padding: 10px 15px !important;
            border-radius: 8px 8px 0 0 !important;
            border: none !important;
            transition: background-color 0.2s;
        }
        div[data-testid="stExpander"] summary:hover {
            background-color: #cbd5e1 !important;
        }
        div[data-testid="stExpander"] summary span p {
            color: #1e293b !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
        }

        /* Expander Content Region - Pure White Body */
        div[data-testid="stExpander"] > div[role="region"] {
            border: none !important;
            padding: 20px !important;
            background-color: white !important;
        }

        /* Feature Title Consistency - Dark Grey (#334155) for all labels */
        label, div[data-testid="stMarkdownContainer"] p strong, .stSelectbox label {
            color: #334155 !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }

        /* Identity Content Style - Normalized Headers */
        .identity-header-text {
            font-size: 1.2rem; font-weight: 600; color: #0f172a; margin-right: 15px;
        }
        .identity-label-small {
            font-size: 1.2rem !important; color: #475569; font-weight: 600 !important;
        }
        .title-box-container {
            background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px;
            padding: 15px 18px; min-height: 120px; max-height: 160px; overflow-y: auto; 
            font-size: 1rem; color: #1e293b; margin-top: 15px; margin-bottom: 25px;
            line-height: 1.6; font-weight: 500;
        }

        /* Selectboxes - Always White, High Contrast even when disabled */
        div[data-baseweb="select"] > div {
            background-color: white !important;
            border: 1px solid #cbd5e1 !important;
            color: #0f172a !important;
        }
        
        /* Forced visible disabled state override */
        div[data-baseweb="select"] > div[aria-disabled="true"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
            opacity: 1 !important;
            border: 1px solid #cbd5e1 !important;
        }

        /* Button Styling - Executive Look */
        .stButton > button {
            background: #f1f5f9; color: #334155; padding: 0.5rem 1rem;
            border-radius: 6px; font-size: 0.85rem; border: 1px solid #cbd5e1; 
            font-weight: 600; width: 100%; transition: all 0.2s;
        }
        .stButton > button:hover {
            background: #e2e8f0; border-color: #94a3b8;
        }
        
        /* Reset Filter Button - Wider and correctly sized */
        div.reset-btn-sidebar > div > button {
            padding: 0.5rem 1.4rem !important;
            font-size: 0.85rem !important;
            width: auto !important;
            min-width: 140px !important;
            background: #f8fafc !important;
            border: 1px solid #cbd5e1 !important;
        }
        
        /* Execution Button */
        div.predict-btn > div > button {
            background: #1e293b !important;
            color: white !important;
            font-size: 1rem !important;
            padding: 0.8rem !important;
            border: none !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .result-count-label {
            font-size: 1.15rem; font-weight: 700; color: #1e293b; margin-bottom: 12px;
            line-height: 1.1;
        }
        
        h1 { margin-top: 0 !important; line-height: 1.2 !important; margin-bottom: 0.2rem !important; }
        
        /* DataFrame Styling - Compact Baseline */
        [data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden;
            margin-top: 0px !important;
        }
        
        /* Ghost Grid Styling */
        .ghost-mode {
            opacity: 0.3;
            pointer-events: none;
            filter: grayscale(1);
        }

        /* Sidebar Toggle Button Visibility - Ultra Visible */
        button[data-testid="stSidebarCollapseButton"] {
            background-color: #334155 !important;
            color: white !important;
            border: 2px solid #cbd5e1 !important;
            border-radius: 50% !important;
            width: 40px !important;
            height: 40px !important;
            top: 12px !important;
            left: 12px !important;
            z-index: 99999 !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
        }
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
    if 'start_year' in df.columns:
        df['start_year'] = pd.to_numeric(df['start_year'], errors='coerce').fillna(0).astype(int)
    return df

@st.cache_data
def load_taxonomy() -> dict:
    if not TAXONOMY_PATH.exists():
        return {}
    with open(TAXONOMY_PATH, 'r') as f:
        data = json.load(f)
        # Handle integrated format with root key "FIELDS" (v2.1+) or fallback to flat (v2.0)
        return data.get("FIELDS", data)

X_ALL = load_predict_data()
TAXONOMY = load_taxonomy()

# ==========================
# UI HELPER FUNCTIONS
# ==========================
def get_risk_tier(score: float):
    if score >= 75:
        return "Robust", "Trial exhibits strong success patterns across clinical pillars.", "#f0fdf4", "#166534"
    elif score >= 50:
        return "Favorable", "Trial exhibits favorable indicators aligned with historical benchmarks.", "#eff6ff", "#1e40af"
    elif score >= 25:
        return "Watchlist", "Mixed signals detected; several risk drivers require mitigation.", "#fff7ed", "#9a3412"
    else:
        return "High Risk", "Significant attrition patterns identified. High vulnerability.", "#fde8e8", "#991b1b"

def reset_filters():
    for key in ["f_sponsor", "f_ta", "f_indication", "f_phase", "f_year", "s_registry", "s_mode"]:
        if key in st.session_state:
            st.session_state[key] = [] if key.startswith("f_") else ""
    st.session_state.selected_nct_id = None
    st.session_state.search_initiated = True

def handle_filter_change():
    st.session_state.selected_nct_id = None
    st.session_state.search_initiated = True

# ==========================
# SIDEBAR: TRIALS FILTER
# ==========================
if "search_initiated" not in st.session_state:
    st.session_state.search_initiated = False

with st.sidebar:
    # Sidebar Header
    st.markdown("<h2 style='margin: 0 0 10px 0; font-size: 1.7rem; color: #f8fafc; font-weight: 800;'>Clinical Trials Filter</h2>", unsafe_allow_html=True)

    # Reset Filter Button at the absolute top
    col_reset_wrap, _ = st.columns([1, 1])
    with col_reset_wrap:
        st.markdown('<div class="reset-btn-sidebar">', unsafe_allow_html=True)
        if st.button("Reset Filter"):
            reset_filters()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Minimal space to keep dropdowns high
    st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
    
    filtered_df = X_ALL.copy()

    # Filters
    st.multiselect("Company / Sponsor", sorted(filtered_df["lead_sponsor_canonical"].dropna().unique().tolist()), key="f_sponsor", placeholder="All Companies", on_change=handle_filter_change)
    if st.session_state.f_sponsor:
        filtered_df = filtered_df[filtered_df["lead_sponsor_canonical"].isin(st.session_state.f_sponsor)]

    st.multiselect("Therapeutic Area", sorted(filtered_df["therapeutic_area"].dropna().unique().tolist()), key="f_ta", placeholder="All Areas", on_change=handle_filter_change)
    if st.session_state.f_ta:
        filtered_df = filtered_df[filtered_df["therapeutic_area"].isin(st.session_state.f_ta)]

    st.multiselect("Indication (GBD)", sorted(filtered_df["gbd_indication_name"].dropna().unique().tolist()), key="f_indication", placeholder="All Indications", on_change=handle_filter_change)
    if st.session_state.f_indication:
        filtered_df = filtered_df[filtered_df["gbd_indication_name"].isin(st.session_state.f_indication)]

    st.multiselect("Trial Phase", sorted(filtered_df["phase"].dropna().unique().tolist()), key="f_phase", placeholder="All Phases", on_change=handle_filter_change)
    if st.session_state.f_phase:
        filtered_df = filtered_df[filtered_df["phase"].isin(st.session_state.f_phase)]

    year_opts = sorted(filtered_df["start_year"].unique().tolist(), reverse=True)
    if 0 in year_opts: year_opts.remove(0)
    st.multiselect("Start Year", year_opts, key="f_year", placeholder="All Years", on_change=handle_filter_change)
    if st.session_state.f_year:
        filtered_df = filtered_df[filtered_df["start_year"].isin(st.session_state.f_year)]

    # Bottom Settings - Deeply hidden
    st.markdown("<div style='height: 550px;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    
    col_reg_l, col_reg_r = st.columns([1, 2])
    with col_reg_l: st.markdown("<p style='font-size: 0.85rem; margin-top: 8px;'>Register</p>", unsafe_allow_html=True)
    with col_reg_r: 
        st.text_input("Register", label_visibility="collapsed", key="s_registry", placeholder="")
    
    col_ana_l, col_ana_r = st.columns([1, 2])
    with col_ana_l: st.markdown("<p style='font-size: 0.85rem; margin-top: 8px;'>Analysis</p>", unsafe_allow_html=True)
    with col_ana_r: 
        st.text_input("Analysis", label_visibility="collapsed", key="s_mode", placeholder="")

    # Analysis logic: Default (empty) shows only correct historical + ongoing. "all" shows everything.
    s_mode = st.session_state.s_mode.lower() if st.session_state.s_mode else ""
    if s_mode != "all":
        filtered_df = filtered_df[(filtered_df["is_correct"] == True) | (filtered_df["trial_segment"] == "ONGOING")]

    # Register logic: Default (empty) shows only HISTORICAL (outcome known). "all" shows both segments.
    s_reg = st.session_state.s_registry.lower() if st.session_state.s_registry else ""
    if s_reg != "all":
        filtered_df = filtered_df[filtered_df["trial_segment"] == "HISTORICAL"]


# ==========================
# MAIN UI
# ==========================
# TOP HEADER
top_c1, top_c2 = st.columns([3, 1])
with top_c1:
    st.markdown("<h1>ClinTrialPredict <span style='font-size: 14px; font-weight: normal; color: #94a3b8; margin-left: 12px;'>v01 Production</span></h1>", unsafe_allow_html=True)

with top_c2:
    if st.session_state.get("selected_nct_id"):
        st.markdown('<div class="predict-btn" style="float: right; margin-top: 5px;">', unsafe_allow_html=True)
        if st.button("Predict Completion"):
            st.session_state.trigger_prediction = True
        st.markdown('</div>', unsafe_allow_html=True)

if "selected_nct_id" not in st.session_state:
    st.session_state.selected_nct_id = None

if st.session_state.selected_nct_id is None:
    # GRID VIEW
    # Tight offset to start 'Define Filter...' at same level as Identity box would be
    # Calibration to align with Sidebar 'Company / Sponsor' top border
    st.markdown("<div style='margin-top: 52px;'></div>", unsafe_allow_html=True)

    if not st.session_state.search_initiated:
        st.markdown("<div class='result-count-label'>← Define filter criteria to select trial</div>", unsafe_allow_html=True)
        st.markdown('<div class="ghost-mode">', unsafe_allow_html=True)
        display_df_ghost = pd.DataFrame(columns=["NCT ID", "Identity", "Sponsor", "Area", "Phase", "Year", "Score", "Accurate"])
        st.dataframe(display_df_ghost, use_container_width=True, hide_index=True, height=450)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-count-label'>Showing {len(filtered_df):,} trials matching criteria</div>", unsafe_allow_html=True)
        st.markdown('<div>', unsafe_allow_html=True)
        grid_cols = ["nct_id", "brief_title", "lead_sponsor_canonical", "therapeutic_area", "phase", "start_year", "Clinical_Score", "is_correct"]
        cols_present = [c for c in grid_cols if c in filtered_df.columns]
        display_df = filtered_df[cols_present].copy()
        
        if "Clinical_Score" in display_df.columns:
            display_df["Clinical_Score"] = display_df["Clinical_Score"].round(1)
        if "is_correct" in display_df.columns:
            display_df["is_correct"] = display_df["is_correct"].map({True: "Yes", False: "No"})

        col_map = {
            "nct_id": "NCT ID", "brief_title": "Identity", "lead_sponsor_canonical": "Sponsor",
            "therapeutic_area": "Area", "phase": "Phase", "start_year": "Year", 
            "Clinical_Score": "Score", "is_correct": "Accurate"
        }
        display_df.columns = [col_map[c] for c in display_df.columns]

        # Dynamic height fits perfectly up to 12 rows, then caps at 450px
        row_height = 35 
        header_height = 40
        dynamic_height = min(450, len(display_df) * row_height + header_height) if len(display_df) > 0 else 450

        event = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            height=dynamic_height
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if event and event.selection and event.selection.rows:
            selected_row_idx = event.selection.rows[0]
            st.session_state.selected_nct_id = display_df.iloc[selected_row_idx]["NCT ID"]
            st.rerun()

else:
    # --- DETAIL VIEW ---
    row = X_ALL[X_ALL[ID_COL] == st.session_state.selected_nct_id].iloc[0]
    
    col_back, _ = st.columns([1, 10])
    with col_back:
        if st.button("← Back"):
            st.session_state.selected_nct_id = None
            st.rerun()

    # 1. IDENTITY BOX (Expandable)
    with st.expander("Identity", expanded=True):
        st.markdown(f"""
        <div style="display: flex; align-items: baseline; margin-bottom: 10px;">
            <span class="identity-header-text">{row[ID_COL]}</span>
            <span class="identity-label-small">{row.get('official_title', 'N/A')}</span>
        </div>
        <div class="title-box-container">
            <span style="color: #64748b; font-size: 0.75rem; text-transform: uppercase; font-weight: 800; display: block; margin-bottom: 8px;">Title</span>
            {row.get('title', 'No title available.')}
        </div>
        """, unsafe_allow_html=True)

    # 2. PILLAR GRID (2x2)
    def render_pillar_features(pillar_ui_name, taxonomy_pillar_name, row_data):
        features = []
        for feat_id, feat_meta in TAXONOMY.items():
            ui = feat_meta.get("ui", {})
            if ui.get("pillar") == taxonomy_pillar_name:
                features.append((feat_id, feat_meta))
        
        # Sort by subgroup name then priority
        features.sort(key=lambda x: (x[1].get("ui", {}).get("subgroup", ""), x[1].get("ui", {}).get("priority", 99)))

        with st.expander(pillar_ui_name, expanded=False):
            # Two features per row
            for i in range(0, len(features), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(features):
                        feat_id, feat_meta = features[i + j]
                        ui = feat_meta.get("ui", {})
                        label = ui.get("label", feat_id)
                        val = row_data.get(feat_id)
                        options = ui.get("options")
                        
                        with cols[j]:
                            if options:
                                opt_labels = [opt[1] for opt in options]
                                curr_idx = 0
                                mapping = feat_meta.get("mapping", {})
                                str_val = str(val).upper()
                                if str_val in mapping:
                                    mapped_label = mapping[str_val][1]
                                    if mapped_label in opt_labels:
                                        curr_idx = opt_labels.index(mapped_label)
                                else:
                                    for idx, opt in enumerate(options):
                                        if str(opt[0]).upper() == str_val or str(opt[1]).upper() == str_val:
                                            curr_idx = idx
                                            break
                                st.selectbox(label, opt_labels, index=curr_idx, key=f"sim_{row_data[ID_COL]}_{feat_id}", disabled=True)
                            else:
                                if pd.isna(val): val = "N/A"
                                st.markdown(f"**{label}**")
                                st.markdown(f"<div style='background:#ffffff; padding:10px; border:1px solid #cbd5e1; border-radius:6px; font-size:0.9rem; color:#0f172a; min-height:40px; margin-bottom:15px;'>{val}</div>", unsafe_allow_html=True)

    col_r1_1, col_r1_2 = st.columns(2)
    with col_r1_1: render_pillar_features("Therapeutic Context", "Therapeutic Context", row)
    with col_r1_2: render_pillar_features("Execution Framework", "Execution Framework", row)

    col_r2_1, col_r2_2 = st.columns(2)
    with col_r2_1: render_pillar_features("Scientific Attempts", "Scientific Attempt", row)
    with col_r2_2: render_pillar_features("Patient Profile", "Patient Profile", row)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # SCORING ENGINE LOGIC
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "analysis_nct_id" not in st.session_state:
        st.session_state.analysis_nct_id = None

    if st.session_state.analysis_nct_id != st.session_state.selected_nct_id:
        st.session_state.analysis_result = None
        st.session_state.analysis_nct_id = st.session_state.selected_nct_id

    if st.session_state.get("trigger_prediction") or st.session_state.analysis_result is not None:
        if st.session_state.analysis_result is None:
            # Ensure nct_id is in the dict even if it is the index
            row_dict = row.replace({np.nan: None}).to_dict()
            row_dict[ID_COL] = st.session_state.selected_nct_id
            
            with st.spinner("Analyzing clinical signals via forensic engine..."):
                try:
                    response = requests.post(API_URL, json=row_dict)
                    if response.status_code == 200:
                        st.session_state.analysis_result = response.json()
                        st.session_state.trigger_prediction = False
                    else:
                        st.error(f"API Connection Error: {response.status_code}")
                except Exception as e:
                    st.error(f"System Error: {e}")

        if st.session_state.analysis_result:
            result = st.session_state.analysis_result
            
            if "error" in result:
                st.error(f"**Forensic Audit Unavailable:** {result.get('error')}")
                if "message" in result:
                    st.info(result.get('message'))
                st.session_state.analysis_result = None
            else:
                score = result.get('score', 0)
                pillar_impacts = result.get('pillar_impacts', [])
                subcat_impacts = result.get('subcat_impacts', [])
                tier, desc, bg_color, t_color = get_risk_tier(score)

            # Auto-Scroll to results
            st.components.v1.html("<script>window.parent.document.querySelector('section.main').scrollTo({top: 1300, behavior: 'smooth'});</script>", height=0)

            st.markdown("<hr style='border: 1px solid #e2e8f0; margin: 40px 0;'>", unsafe_allow_html=True)
            col_left, col_right = st.columns([1.0, 1.4])
            with col_left:
                st.markdown("##### Probability of Success Index")
                st.plotly_chart(plot_success_gauge(score), use_container_width=True, config={'displayModeBar': False})
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
                    st.markdown("##### Forensic Driver Decomposition")
                    st.plotly_chart(plot_treemap(subcat_impacts, pillar_impacts), use_container_width=True, config={'displayModeBar': False})
