import os
import json
import base64
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import streamlit as st
import requests
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# IMPORT PLOTTING UTILS
from utils.plot import plot_success_gauge, plot_impact_bar, plot_treemap

# Load environment variables
load_dotenv()

# ==========================
# 1. SETUP & CONFIGURATION
# ==========================
st.set_page_config(
    page_title="ClinTrialPredict | Predictive Engine",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- PATHS & URLS ---
CURRENT_DIR = Path(__file__).resolve().parent
DATA_PATH = CURRENT_DIR / "data" / "search_registry.csv"
TAXONOMY_PATH = CURRENT_DIR.parent / "models" / "taxonomy_01.json"
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
ID_COL = "nct_id"


# --- BRANDING CONSTANTS ---
HUE = 180
INTENSITY = 0.8
DARKNESS = 0.85
THICKNESS = 0
BRAND_FILTER = (
    f"contrast(1.5) brightness(0.9) grayscale(100%) sepia(100%) "
    f"hue-rotate({HUE}deg) saturate({INTENSITY}) brightness({DARKNESS}) "
    f"contrast(1.2) drop-shadow({THICKNESS}px {THICKNESS}px 0px #52606d) "
    f"drop-shadow(-{THICKNESS}px -{THICKNESS}px 0px #52606d)"
)

# ==========================
# 2. STYLES (Consolidated)
# ==========================
def inject_custom_styles():
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

            .material-symbols-outlined {{
                font-family: 'Material Symbols Outlined' !important;
                font-weight: normal; font-style: normal; font-size: 24px; line-height: 1;
                display: inline-block; white-space: nowrap; direction: ltr; -webkit-font-smoothing: antialiased;
            }}

            :root {{ color-scheme: light !important; }}

            html, body, [data-testid="stAppViewContainer"], .stMarkdown, p, span, label, div {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }}

            html, body, [data-testid="stAppViewContainer"] {{
                background-color: #f8fafc !important;
                color: #334155 !important;
            }}

            .block-container {{
                max-width: 1400px !important;
                padding: 0rem 2rem 2rem !important;
                margin-left: auto !important; margin-right: auto !important;
            }}

            [data-testid="stHeader"] {{ background-color: rgba(0,0,0,0) !important; color: #334155 !important; }}

            button[kind="header"], [data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"] {{
                opacity: 1 !important; visibility: visible !important;
            }}

            [data-testid="stSidebarCollapseButton"],
            [data-testid="collapsedControl"] {{
                transform: translateY(20px) !important;
            }}

            section[data-testid="stSidebar"] {{
                background-color: #717d8b !important;
                border-right: 1px solid #606c7a;
            }}

            div[data-testid="stSidebarContent"] {{
                padding-top: 0px !important;
                transform: translateY(-26px) !important;
            }}

            section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
                gap: 0rem !important;
            }}

            /* SELECTBOXES */
            div[data-baseweb="select"] > div {{
                background-color: white !important;
                border: 1.5px solid #94a3b8 !important;
                border-radius: 8px !important;
                transition: all 0.2s;
                min-height: 38px !important;
                height: 38px !important;
                font-size: 0.80rem !important;
                padding-top: 0 !important;
                padding-bottom: 0 !important;
                display: flex !important;
                align-items: center !important;
            }}

            div[data-baseweb="select"] > div > div {{
                align-items: center !important;
            }}

            div[data-baseweb="select"] span {{
                line-height: 1.1 !important;
            }}

            div[data-baseweb="select"] input {{
                padding-top: 0 !important;
                padding-bottom: 0 !important;
                line-height: 1.1 !important;
            }}

            /* TEXT INPUTS */
            [data-testid="stTextInputRootElement"] input {{
                background-color: white !important;
                border: 1.5px solid #94a3b8 !important;
                border-radius: 8px !important;
                transition: all 0.2s;
                min-height: 38px !important;
                height: 38px !important;
                font-size: 0.80rem !important;
            }}

            /* GLOBAL MULTISELECT DROPDOWN ALIGNMENT */
            [data-baseweb="popover"] li,
            div[data-baseweb="select"] ul li,
            div[role="listbox"] li {{
                font-size: 0.80rem !important;
            }}

            /* Filter Panel Styles */
            .st-key-filter_header {{
                background-color: #717d8b !important; border: 1px solid #606c7a !important;
                border-radius: 14px !important; padding: 22px 20px 36px 20px !important;
                box-shadow: -6px 6px 12px -3px rgba(0,0,0,0.12) !important;
                margin-bottom: 0px !important;
            }}

            .st-key-filter_header .highlight-title {{ color: #ffffff !important; }}

            .st-key-filter_body {{
                background-color: #717d8b !important; border: 1px solid #606c7a !important;
                border-radius: 14px !important; padding: 34px 25px 35px 25px !important;
                box-shadow: -6px 6px 12px -3px rgba(0,0,0,0.12) !important;
                margin-bottom: 4px !important;
            }}

            /* LANDING PAGE FILTERS */
            .st-key-filter_body [data-testid="stVerticalBlock"] {{
                gap: 0rem !important;
            }}

            .st-key-filter_body div[data-baseweb="select"] {{
                margin-top: 6px !important;
            }}

            .st-key-filter_body [data-testid="stVerticalBlock"] > div {{
                margin-bottom: -6px !important;
                padding-bottom: 0px !important;
            }}

            .st-key-filter_body [data-testid="stWidgetLabel"] {{
                min-height: 0px !important;
                margin-bottom: 0px !important;
            }}

            .st-key-filter_body label,
            .st-key-filter_body div[data-testid="stMarkdownContainer"] p,
            .st-key-filter_body [data-testid="stWidgetLabel"] p {{
                color: #ffffff !important;
                font-weight: 600 !important;
                font-size: 0.85rem !important;
                letter-spacing: -0.01em !important;
                margin-bottom: -3px !important;
            }}

            /* SIDEBAR FILTERS */
            .st-key-sidebar_filters div[data-baseweb="select"] {{
                margin-top: 6px !important;
            }}

            .st-key-sidebar_filters [data-testid="stElementContainer"] {{
                margin-bottom: 10px !important;
            }}

            .st-key-sidebar_filters [data-testid="stElementContainer"]:last-child {{
                margin-bottom: 0px !important;
            }}

            .st-key-sidebar_filters [data-testid="stWidgetLabel"] {{
                min-height: 0px !important;
                margin-bottom: 0px !important;
            }}

            .st-key-sidebar_filters label,
            .st-key-sidebar_filters div[data-testid="stMarkdownContainer"] p,
            .st-key-sidebar_filters [data-testid="stWidgetLabel"] p {{
                color: #ffffff !important;
                font-weight: 600 !important;
                font-size: 0.85rem !important;
                letter-spacing: -0.01em !important;
                margin-bottom: -3px !important;
            }}

            .right-column-stack {{ display: flex; flex-direction: column; gap: 1rem; height: 100%; }}

            /* Boxes & Highlights */
            .highlight-box {{
                background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
                padding: 24px; box-shadow: -6px 6px 12px -3px rgba(0,0,0,0.12) !important;
                height: 100%;
            }}

            .mission-box {{ background-color: #e2e8f0 !important; border: 1px solid #cbd5e1 !important; }}
            .highlight-title {{ font-weight: 800; color: #52606d !important; font-size: 1.15rem; margin-bottom: 8px; letter-spacing: -0.02em; }}
            .highlight-text {{ color: #64748b; font-size: 0.95rem; line-height: 1.55; font-weight: 450; }}
            .highlight-text b, .highlight-text strong {{ font-weight: 700 !important; color: inherit !important; }}

            /* Tags & Labels */
            label, strong {{ color: #475569 !important; font-weight: 600 !important; font-size: 0.85rem !important; letter-spacing: -0.01em; }}

            /* Detail View */
            .identity-header-text {{ font-size: 1.2rem; font-weight: 600; color: #334155 !important; margin-right: 15px; }}
            .title-box-container {{
                background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px;
                padding: 15px 18px; margin-top: 15px; margin-bottom: 25px; line-height: 1.6;
                font-weight: 500; box-shadow: -6px 6px 12px -3px rgba(0,0,0,0.12) !important;
            }}
            .pillar-val-box {{
                background:#ffffff; padding:10px; border:1px solid #cbd5e1; border-radius:6px;
                font-size:0.9rem; color:#334155 !important; min-height:40px; margin-bottom:15px;
                box-shadow: -6px 6px 12px -3px rgba(0,0,0,0.12) !important;
            }}

            /* Buttons */
            .stButton > button {{
                border-radius: 8px !important; font-weight: 700 !important; padding: 0px 1rem !important;
                transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
                border: 1.5px solid #99a7b9 !important; background-color: #b2bccb !important;
                color: #ffffff !important; min-height: 38px !important; height: 38px !important;
                display: flex !important; align-items: center !important; justify-content: center !important;
            }}
            .stButton > button:hover {{
                background-color: #334155 !important; border-color: #1e293b !important;
                box-shadow: 0 8px 20px rgba(0,0,0,0.2) !important; transform: scale(1.02) translateY(-2px) !important;
            }}
            .stButton > button * {{ color: #ffffff !important; fill: #ffffff !important; }}
        </style>
    """, unsafe_allow_html=True)

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

def init_session_state():
    defaults = {
        "search_initiated": False,
        "selected_nct_id": None,
        "trigger_prediction": False,
        "analysis_result": None,
        "analysis_nct_id": None,
        "f_sponsor": None,
        "f_ta": None,
        "f_phase": None,
        "f_year": None,
        "f_nct_id": None,
        "s_registry": "",
        "s_mode": ""
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def reset_filters():
    for key in ["f_sponsor", "f_ta", "f_phase", "f_year", "f_nct_id"]:
        st.session_state[key] = None
    for key in ["s_registry", "s_mode"]:
        st.session_state[key] = ""
    st.session_state.selected_nct_id = None
    st.session_state.search_initiated = False

def get_risk_tier(score: float):
    if score >= 75: return "Robust", "Strong success patterns detected.", "#f0fdf4", "#166534"
    if score >= 50: return "Favorable", "Favorable historical indicators.", "#eff6ff", "#1e40af"
    if score >= 25: return "Watchlist", "Mixed signals; mitigation required.", "#fff7ed", "#9a3412"
    return "High Risk", "Significant attrition patterns.", "#fde8e8", "#991b1b"

# ==========================
# 4. COMPONENTS
# ==========================

def render_header(is_landing=True):
    logo_path = CURRENT_DIR / "logo_grey_title.png"
    img_base64 = ""
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode()

    t1, t2 = st.columns([3, 1])
    with t1:
        size = 72 if is_landing else 44
        border = 4 if is_landing else 2
        radius = 18 if is_landing else 7
        title_size = "2.8rem" if is_landing else "3.2rem"

        html = f"""
            <div style='display: flex; align-items: center; gap: 12px; margin-top: {"15px" if is_landing else "25px"};'>
                <div style='background-color: white; border: {border}px solid #52606d; padding: 2px; border-radius: {radius}px; display: flex; align-items: center; justify-content: center; height: {size}px; width: {size}px; flex-shrink: 0; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-top: {"10px" if is_landing else "-0px"};'>
                    <img src='data:image/png;base64,{img_base64}' style='height: {size-2}px; filter: {BRAND_FILTER};'>
                </div>
                <div style='display: {"block" if is_landing else "flex"}; align-items: {"stretch" if is_landing else "flex-end"}; gap: {"0px" if is_landing else "14px"};'>
                    <div style='font-size: {title_size}; font-weight: 800; color: #52606d; line-height: 1; {"margin-top: 10px;" if is_landing else ""}'>CTPredict</div>
                    {"<div style='color: #52606d; font-size: 1.5rem; font-weight: 800; display: flex; align-items: baseline; gap: 15px; margin-top: 5px;'><span style='line-height: 1;'>Late-Stage Clinical Trial Predictive Engine</span><span style='font-size: 0.7rem; color: #94a3b8; text-transform: uppercase;'>demo version</span></div>" if is_landing else "<span style='font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; line-height: 1; margin-bottom: 5px;'>Demo Version</span>"}
                </div>
            </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    with t2:
        if st.session_state.selected_nct_id:
            if st.button("Predict Completion", use_container_width=True, type="primary"):
                st.session_state.trigger_prediction = True

def render_filters(df, is_sidebar=False):
    COL_MAP = {
        "f_sponsor": "lead_sponsor_canonical",
        "f_ta": "therapeutic_area_ui",
        "f_phase": "phase_ui",
        "f_year": "start_year",
        "f_nct_id": "nct_id"
    }

    def apply_filters(base_df, skip_key=None):
        tdf = base_df.copy()
        for k, c in COL_MAP.items():
            val = st.session_state.get(k)
            if k == skip_key or val in (None, ""):
                continue
            tdf = tdf[tdf[c] == val]
        return tdf

    def get_opts(col_key):
        tdf = apply_filters(df, skip_key=col_key)
        col = COL_MAP[col_key]

        if col == "start_year":
            return sorted([y for y in tdf[col].dropna().unique() if y > 0], reverse=True)

        return sorted(tdf[col].dropna().unique())

    def render_select(label, col_key, placeholder):
        opts = list(get_opts(col_key))

        if st.session_state.get(col_key) not in opts:
            st.session_state[col_key] = None

        st.selectbox(
            label,
            options=opts,
            key=col_key,
            index=None,
            placeholder=placeholder
        )

    if is_sidebar:
        render_select("Company / Sponsor", "f_sponsor", "All Sponsors")
        render_select("Therapeutic Area", "f_ta", "All Therapeutic Areas")
        render_select("Trial Phase", "f_phase", "All Phases")
        render_select("Start Year", "f_year", "All Years")
        render_select("Clinical trial number (AACT)", "f_nct_id", "All NCT IDs")
    else:
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            render_select("Company / Sponsor", "f_sponsor", "All Sponsors")
        with r1_c2:
            render_select("Therapeutic Area", "f_ta", "All Therapeutic Areas")

        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            render_select("Trial Phase", "f_phase", "All Phases")
        with r2_c2:
            render_select("Start Year", "f_year", "All Years")

        r3_c1, r3_c2, r3_c3 = st.columns([2, 0.6, 1.4], vertical_alignment="bottom")
        with r3_c1:
            render_select("Clinical trial number (AACT)", "f_nct_id", "All NCT IDs")
        with r3_c2:
            st.button("Reset", use_container_width=True, on_click=reset_filters)
        with r3_c3:
            st.button(
                "Search Trials",
                use_container_width=True,
                type="primary",
                on_click=lambda: setattr(st.session_state, "search_initiated", True)
            )

    curr_df = apply_filters(df)

    if not is_sidebar:
        st.markdown(
            f"<div style='text-align:right; font-size:0.8rem; color:#cbd5e1; margin-top: 4px; margin-bottom: -16px;'>{len(curr_df):,} trials matching criteria</div>",
            unsafe_allow_html=True
        )
    return curr_df

def render_trials_grid(df):
    grid_df = df[["nct_id", "ui_search_label", "lead_sponsor_canonical", "therapeutic_area_ui", "phase_ui", "start_year", "Clinical_Score"]].copy()
    grid_df.columns = ["NCT ID", "Identity", "Sponsor", "Area", "Phase", "Start Year", "Score"]
    grid_df = grid_df.sort_values("NCT ID", ascending=True, kind="stable").reset_index(drop=True)

    gb = GridOptionsBuilder.from_dataframe(grid_df)
    gb.configure_default_column(
        sortable=True,
        filter=False,
        resizable=True,
        suppressMenu=True,
        minWidth=95,
        flex=1
    )

    gb.configure_column(
        "NCT ID",
        maxWidth=110,
        flex=0.68,
        cellClass="ag-tight-center-cell",
        headerClass="ag-center-header"
    )
    gb.configure_column(
        "Identity",
        minWidth=300,
        flex=3.15,
        cellClass="ag-identity-cell",
        headerClass="ag-identity-header"
    )
    gb.configure_column(
        "Sponsor",
        maxWidth=175,
        flex=1.20,
        cellClass="ag-tight-center-cell",
        headerClass="ag-center-header"
    )
    gb.configure_column(
        "Area",
        maxWidth=160,
        flex=1.00,
        cellClass="ag-tight-center-cell",
        headerClass="ag-center-header"
    )
    gb.configure_column(
        "Phase",
        maxWidth=105,
        flex=0.65,
        cellClass="ag-tight-center-cell",
        headerClass="ag-center-header"
    )
    gb.configure_column(
        "Start Year",
        maxWidth=90,
        flex=0.58,
        cellClass="ag-tight-center-cell",
        headerClass="ag-center-header",
        filter=False
    )
    gb.configure_column(
        "Score",
        maxWidth=82,
        flex=0.52,
        cellClass="ag-tight-center-cell",
        headerClass="ag-center-header",
        filter=False,
        valueFormatter=JsCode(
            "function(params) { return params.value != null ? Number(params.value).toFixed(1).replace('.', ',') : ''; }"
        )
    )

    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_grid_options(
        rowHeight=35,
        headerHeight=36,
        suppressCellFocus=True,
        animateRows=True,
        onRowClicked=JsCode("function(e) { e.api.deselectAll(); e.node.setSelected(true, true); }")
    )

    dynamic_height = min(545, 36 + (len(grid_df) * 35) + 2)
    response = AgGrid(
        grid_df,
        gridOptions=gb.build(),
        height=dynamic_height,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        update_on=["selectionChanged"],
        theme="streamlit",
        custom_css={
            ".ag-root-wrapper": {
                "border": "1px solid #cbd5e1",
                "border-radius": "12px",
                "box-shadow": "-6px 6px 12px -3px rgba(0,0,0,0.12)"
            },
            ".ag-header": {
                "background-color": "#e2e8f0 !important"
            },
            ".ag-row": {
                "cursor": "pointer !important",
                "color": "#334155 !important",
                "font-size": "0.80rem !important"
            },
            ".ag-cell": {
                "display": "flex !important",
                "align-items": "center !important"
            },
            ".ag-tight-center-cell": {
                "justify-content": "center !important",
                "text-align": "center !important"
            },
            ".ag-identity-cell": {
                "justify-content": "flex-start !important",
                "padding-left": "6px !important"
            },
            ".ag-header-cell-label": {
                "width": "100% !important"
            },
            ".ag-center-header .ag-header-cell-label": {
                "justify-content": "center !important"
            },
            ".ag-center-header .ag-header-cell-text": {
                "width": "100% !important",
                "text-align": "center !important"
            },
            ".ag-identity-header .ag-header-cell-label": {
                "justify-content": "flex-start !important"
            },
            ".ag-identity-header .ag-header-cell-text": {
                "text-align": "left !important"
            }
        }
    )

    selected = response.get("selected_rows", [])
    if isinstance(selected, pd.DataFrame):
        if not selected.empty: return selected.iloc[0]["NCT ID"]
    elif selected:
        return selected[0]["NCT ID"]
    return None

def render_pillar_expander(title, pillar_name, data):
    feats = sorted([(f_id, f_m) for f_id, f_m in TAXONOMY.items() if f_m.get("ui", {}).get("pillar") == pillar_name],
                   key=lambda x: (x[1].get("ui", {}).get("subgroup", ""), x[1].get("ui", {}).get("priority", 99)))
    with st.expander(title, expanded=False):
        for i in range(0, len(feats), 2):
            cols = st.columns(2)
            for j in range(2):
                if i+j < len(feats):
                    f_id, f_m = feats[i+j]
                    display_col = f_id.replace("_ml", "_ui") if "_ml" in f_id else f"{f_id}_ui"
                    val = data.get(display_col, data.get(f_id))
                    with cols[j]:
                        st.markdown(f"**{f_m.get('ui', {}).get('label', f_id)}**")
                        st.markdown(f"<div class='pillar-val-box'>{val if not pd.isna(val) else 'N/A'}</div>", unsafe_allow_html=True)

# ==========================
# 5. MAIN UI FLOW
# ==========================
init_session_state()
inject_custom_styles()

# Landing or Detail Logic
if not st.session_state.selected_nct_id:
    x_base = X_ALL.copy()
    if st.session_state.get("s_mode", "").lower() != "all":
        x_base = x_base[(x_base["is_correct"] == True) | (x_base["trial_segment"] == "ONGOING")]

    render_header(is_landing=not st.session_state.search_initiated)

    if not st.session_state.search_initiated:
        st.markdown('''
            <div class="highlight-box mission-box" style="margin-top: 1.5rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div class="highlight-title">Operational Success & Risk Stratification</div>
                    <div style="font-size:0.65rem; font-weight:800; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em;">Core Mission</div>
                </div>
                <div class="highlight-text">This predictive engine estimates the <b>likelihood of operational completion</b> and the <b>risk of early termination</b> using only data available at clinical trial initiation. Each trial is systematically evaluated and classified into <b>four distinct tiers</b> - High Risk, Watchlist, Favorable, and Robust - providing a clear and actionable risk profile.</div>
            </div>
        ''', unsafe_allow_html=True)

        cl, cr = st.columns(2)
        with cl:
            with st.container(key="filter_header"): st.markdown('<div class="highlight-title" style="margin:0;">Clinical Trial Selection</div>', unsafe_allow_html=True)
            with st.container(key="filter_body"): render_filters(x_base)
        with cr:
            st.markdown('''
                <div class="right-column-stack">
                    <div class="highlight-box">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div class="highlight-title">Industry-Scale Clinical Data</div>
                            <div style="font-size:0.65rem; font-weight:800; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em;">Intelligence Source</div>
                        </div>
                        <div class="highlight-text">Built on the publicly available <b>AACT registry</b>, this machine learning system leverages execution patterns from <b>30,000+ Phase II and III trials</b> since 2005. The analytical scope focuses on <b>late-stage studies</b>, where strategic and financial stakes are highest.</div>
                    </div>
                    <div class="highlight-box">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div class="highlight-title">Predictive Power & Benchmarking</div>
                            <div style="font-size:0.65rem; font-weight:800; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em;">Engine Accuracy</div>
                        </div>
                        <div class="highlight-text">When comparing a completed trial with one that terminated early, the system assigns a <b>higher risk score</b> to the failed trial in <b>75% of cases</b>. It outperforms the 50% random baseline and traditional approaches built on publicly available data (<b>ROC AUC ≈ 0.75</b> vs. 0.50 baseline).</div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
    else:
        with st.sidebar:
            if st.button("Reset Filter", use_container_width=True):
                reset_filters()
                st.rerun()

            st.markdown("<div style='margin-top: 78px;'></div>", unsafe_allow_html=True)

            with st.container(key="sidebar_filters"):
                filtered_df = render_filters(x_base, is_sidebar=True)

            st.markdown("<div style='height: 300px;'></div>--- ", unsafe_allow_html=True)
            st.text_input("Register", key="s_registry")
            st.text_input("Analysis", key="s_mode")

        st.markdown(f"<div style='margin-top:1.5rem; position:relative; top:-4px; color:#94a3b8; font-weight:600; font-size:0.7rem;'>{len(filtered_df):,} Matching Trials</div>", unsafe_allow_html=True)
        selected_id = render_trials_grid(filtered_df)
        if selected_id:
            st.session_state.selected_nct_id = selected_id
            st.rerun()
else:
    render_header(is_landing=False)
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

    c1, c2 = st.columns(2)
    with c1: render_pillar_expander("Therapeutic Context", "Therapeutic Context", row)
    with c2: render_pillar_expander("Execution Framework", "Execution Framework", row)
    c3, c4 = st.columns(2)
    with c3: render_pillar_expander("Scientific Attempt", "Scientific Attempt", row)
    with c4: render_pillar_expander("Patient Profile", "Patient Profile", row)

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
                st.markdown(f"<div style='background:{bg}; color:{tc}; padding:20px; border-radius:12px; border:1px solid {tc + '22'};'><div style='font-size:1.4rem; font-weight:800;'>{tier}</div><div>{desc}</div></div>", unsafe_allow_html=True)
            with cr:
                if res.get('pillar_impacts'): st.plotly_chart(plot_impact_bar(pd.DataFrame(res['pillar_impacts'])), use_container_width=True)
