import os
import json
import html
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


DEBUG_OVERLAY = True
DEBUG_HOVER_INDEX = True

# ==========================
# 2. STYLES (Consolidated)
# ==========================
def inject_custom_styles():
    debug_overlay_css = """
            /* =========================
               DEBUG OVERLAP VISUALIZER
               ========================= */

            [data-testid="stHeader"] {
                background: rgba(255, 0, 0, 0.12) !important;
                outline: 2px solid red !important;
            }

            .block-container {
                outline: 2px solid orange !important;
            }

            [data-testid="stVerticalBlock"] {
                outline: 1px dashed magenta !important;
            }

            [data-testid="stElementContainer"] {
                outline: 1px dashed cyan !important;
                background: rgba(0, 255, 255, 0.05) !important;
            }

            .st-key-header_action_buttons {
                outline: 3px solid lime !important;
                background: rgba(0, 255, 0, 0.08) !important;
            }

            .st-key-header_action_buttons .stButton {
                outline: 2px solid blue !important;
                background: rgba(0, 0, 255, 0.05) !important;
            }

            .st-key-header_action_buttons .stButton > button {
                outline: 2px solid black !important;
            }
    """ if DEBUG_OVERLAY else ""

    debug_hover_index_css = """
            /* =========================
               DEBUG LABELS - HOVER + INDEX
               ========================= */

            body {
                counter-reset: dbg-ec dbg-vb dbg-btn;
            }

            [data-testid="stHeader"],
            .block-container,
            [data-testid="stVerticalBlock"],
            [data-testid="stElementContainer"],
            .st-key-header_action_buttons,
            .st-key-header_action_buttons .stButton,
            .st-key-header_action_buttons .stButton > button,
            .st-key-filter_header,
            .st-key-filter_body,
            .st-key-sidebar_filters,
            .st-key-trial_top_strip,
            .st-key-trial_title_identity,
            .st-key-trial_title_shell,
            .st-key-trial_meta_shell,
            .st-key-trial_top_meta {
                position: relative !important;
            }

            [data-testid="stElementContainer"] {
                counter-increment: dbg-ec;
            }

            [data-testid="stVerticalBlock"] {
                counter-increment: dbg-vb;
            }

            .st-key-header_action_buttons .stButton {
                counter-increment: dbg-btn;
            }

            [data-testid="stHeader"]::before,
            .block-container::before,
            [data-testid="stVerticalBlock"]::before,
            [data-testid="stElementContainer"]::before,
            .st-key-header_action_buttons::before,
            .st-key-header_action_buttons .stButton::before,
            .st-key-header_action_buttons .stButton > button::before,
            .st-key-filter_header::before,
            .st-key-filter_body::before,
            .st-key-sidebar_filters::before,
            .st-key-trial_top_strip::before,
            .st-key-trial_title_identity::before,
            .st-key-trial_title_shell::before,
            .st-key-trial_meta_shell::before,
            .st-key-trial_top_meta::before {
                position: absolute !important;
                top: 0 !important;
                left: 0 !important;
                transform: translate(0, -100%) !important;
                background: rgba(15, 23, 42, 0.92) !important;
                color: #ffffff !important;
                font-size: 10px !important;
                font-weight: 700 !important;
                line-height: 1 !important;
                padding: 3px 6px !important;
                border-radius: 4px !important;
                white-space: nowrap !important;
                z-index: 99999 !important;
                pointer-events: none !important;
                text-transform: none !important;
                letter-spacing: 0 !important;
                opacity: 0 !important;
                transition: opacity 0.12s ease !important;
            }

            [data-testid="stHeader"]:hover::before,
            .block-container:hover::before,
            [data-testid="stVerticalBlock"]:hover::before,
            [data-testid="stElementContainer"]:hover::before,
            .st-key-header_action_buttons:hover::before,
            .st-key-header_action_buttons .stButton:hover::before,
            .st-key-header_action_buttons .stButton > button:hover::before,
            .st-key-filter_header:hover::before,
            .st-key-filter_body:hover::before,
            .st-key-sidebar_filters:hover::before,
            .st-key-trial_top_strip:hover::before,
            .st-key-trial_title_identity:hover::before,
            .st-key-trial_title_shell:hover::before,
            .st-key-trial_meta_shell:hover::before,
            .st-key-trial_top_meta:hover::before {
                opacity: 1 !important;
            }

            [data-testid="stHeader"]::before {
                content: "stHeader" !important;
            }

            .block-container::before {
                content: "block-container" !important;
            }

            [data-testid="stVerticalBlock"]::before {
                content: "stVerticalBlock #" counter(dbg-vb) !important;
            }

            [data-testid="stElementContainer"]::before {
                content: "stElementContainer #" counter(dbg-ec) !important;
            }

            .st-key-header_action_buttons::before {
                content: "header_action_buttons" !important;
            }

            .st-key-header_action_buttons .stButton::before {
                content: "stButton wrapper #" counter(dbg-btn) !important;
            }

            .st-key-header_action_buttons .stButton > button::before {
                content: "actual button" !important;
            }

            .st-key-filter_header::before {
                content: "filter_header" !important;
            }

            .st-key-filter_body::before {
                content: "filter_body" !important;
            }

            .st-key-sidebar_filters::before {
                content: "sidebar_filters" !important;
            }

            .st-key-trial_top_strip::before {
                content: "trial_top_strip" !important;
            }

            .st-key-trial_title_identity::before {
                content: "trial_title_identity" !important;
            }

            .st-key-trial_title_shell::before {
                content: "trial_title_shell" !important;
            }

            .st-key-trial_meta_shell::before {
                content: "trial_meta_shell" !important;
            }

            .st-key-trial_top_meta::before {
                content: "trial_top_meta" !important;
            }
    """ if DEBUG_HOVER_INDEX else ""

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
                padding: 0.60rem 2rem 2rem !important;
                margin-left: auto !important; margin-right: auto !important;
            }}

            [data-testid="stHeader"] {{
                background-color: rgba(0,0,0,0) !important;
                color: #334155 !important;
            }}

            button[kind="header"],
            [data-testid="stSidebarCollapseButton"],
            [data-testid="collapsedControl"] {{
                opacity: 1 !important;
                visibility: visible !important;
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
            .st-key-sidebar_filters [data-testid="stWidgetLabel"] p {{
                color: #ffffff !important;
                font-weight: 600 !important;
                font-size: 0.85rem !important;
                letter-spacing: -0.01em !important;
                margin-bottom: -3px !important;
            }}

            /* TOP STRIP */
            .st-key-trial_top_strip {{
                position: relative !important;
                top: 0 !important;
                margin-top: 0 !important;
                padding-top: 0 !important;
            }}

            .st-key-trial_top_strip [data-testid="stVerticalBlock"] {{
                gap: 0 !important;
                margin-top: 0 !important;
                padding-top: 0 !important;
            }}

            .st-key-trial_title_identity {{
                margin-top: 0 !important;
                margin-bottom: 0 !important;
                padding-top: 0 !important;
                padding-bottom: 0 !important;
            }}

            .st-key-trial_title_identity [data-testid="stVerticalBlock"] {{
                gap: 0 !important;
                margin-top: 0 !important;
                padding-top: 0 !important;
            }}

            .st-key-trial_title_shell {{
                background-color: #e2e8f0 !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 14px !important;
                padding: 12px 16px !important;
                box-shadow: -6px 6px 12px -3px rgba(0,0,0,0.12) !important;
                min-height: 138px !important;
                margin-top: 0 !important;
            }}

            .st-key-trial_title_shell > div,
            .st-key-trial_title_shell [data-testid="stVerticalBlock"] {{
                width: 100% !important;
                height: 100% !important;
            }}

            .st-key-trial_title_shell [data-testid="stVerticalBlock"] {{
                gap: 0.20rem !important;
                justify-content: center !important;
            }}

            .st-key-trial_title_shell [data-testid="stTextArea"] {{
                margin: 0 !important;
            }}

            .st-key-trial_title_shell [data-testid="stTextArea"] [data-baseweb="textarea"] {{
                background: #ffffff !important;
                border: 1px solid #cbd5e1 !important;
                box-shadow: -4px 4px 10px -4px rgba(0,0,0,0.10) !important;
                border-radius: 10px !important;
                padding: 0 !important;
            }}

            .st-key-trial_title_shell .stTextArea textarea {{
                background: #ffffff !important;
                padding: 10px 12px !important;
                min-height: 86px !important;
                line-height: 1.36 !important;
                font-size: 0.90rem !important;
                font-weight: 600 !important;
                color: #334155 !important;
            }}

            .st-key-trial_title_shell .stTextArea textarea:disabled {{
                background: #ffffff !important;
                -webkit-text-fill-color: #334155 !important;
                opacity: 1 !important;
            }}

            .st-key-trial_title_shell .stTextArea textarea:not(:disabled) {{
                background: #ffffff !important;
                border-radius: 10px !important;
                padding: 10px 12px !important;
            }}

            .st-key-trial_meta_shell {{
                background-color: #e2e8f0 !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 14px !important;
                padding: 12px 14px !important;
                box-shadow: -6px 6px 12px -3px rgba(0,0,0,0.12) !important;
                min-height: 138px !important;
                margin-top: 0 !important;
            }}

            .st-key-trial_meta_shell > div,
            .st-key-trial_top_meta,
            .st-key-trial_top_meta [data-testid="stVerticalBlock"] {{
                width: 100% !important;
                height: 100% !important;
            }}

            .st-key-trial_meta_shell [data-testid="stVerticalBlock"] {{
                gap: 0rem !important;
            }}

            .st-key-trial_top_meta {{
                max-width: 100% !important;
            }}

            .st-key-trial_top_meta [data-testid="stVerticalBlock"] {{
                display: flex !important;
                flex-direction: column !important;
                justify-content: space-between !important;
                gap: 0.22rem !important;
            }}

            .st-key-trial_top_meta [data-testid="stElementContainer"] {{
                margin-bottom: 0 !important;
            }}

            .st-key-trial_top_meta div[data-baseweb="select"] > div,
            .st-key-trial_top_meta [data-testid="stTextInputRootElement"] input {{
                min-height: 34px !important;
                height: 34px !important;
                font-size: 0.79rem !important;
            }}

            /* UNIFIED TEXT PANELS: same look in read and edit mode */
            [data-testid="stTextArea"] [data-baseweb="textarea"] {{
                border: 1px solid #cbd5e1 !important;
                border-radius: 10px !important;
                background-color: #ffffff !important;
                box-shadow: -4px 4px 10px -4px rgba(0,0,0,0.10) !important;
                overflow: hidden !important;
            }}

            [data-testid="stTextArea"] [data-baseweb="textarea"] > div {{
                background-color: #ffffff !important;
                padding: 0 !important;
            }}

            .stTextArea textarea {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
                font-size: 0.84rem !important;
                line-height: 1.45 !important;
                font-weight: 500 !important;
                color: #334155 !important;
                background-color: #ffffff !important;
                border: none !important;
                border-radius: 10px !important;
                padding: 10px 12px !important;
                margin: 0 !important;
                resize: none !important;
                white-space: pre-wrap !important;
                overflow-wrap: break-word !important;
                tab-size: 4 !important;
                caret-color: #334155 !important;
            }}

            .stTextArea textarea:focus {{
                outline: none !important;
                box-shadow: none !important;
            }}

            .stTextArea textarea:disabled {{
                color: #334155 !important;
                -webkit-text-fill-color: #334155 !important;
                opacity: 1 !important;
                cursor: default !important;
                background-color: #ffffff !important;
            }}

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

            /* HEADER RIGHT COLUMN TIGHT WRAPPERS */
            .st-key-header_action_buttons,
            .st-key-header_edit_toggle_row,
            .st-key-trial_detail_header_spacer {{
                margin: 0 !important;
                padding: 0 !important;
            }}

            .st-key-header_action_buttons [data-testid="stVerticalBlock"],
            .st-key-header_edit_toggle_row [data-testid="stVerticalBlock"],
            .st-key-trial_detail_header_spacer [data-testid="stVerticalBlock"] {{
                gap: 0rem !important;
            }}

            .st-key-header_action_buttons [data-testid="stWidgetLabel"] {{
                margin-bottom: 0 !important;
            }}

            .st-key-header_action_buttons [data-testid="stWidgetLabel"] p,
            .st-key-header_action_buttons label p {{
                font-size: 0.72rem !important;
                line-height: 1 !important;
                white-space: nowrap !important;
            }}

            /* Buttons */
            .stButton > button {{
                border-radius: 8px !important;
                font-weight: 400 !important;
                font-size: 0.85rem !important;
                line-height: 1 !important;
                padding: 0px 1rem !important;
                transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
                min-height: 37px !important;
                height: 37px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                cursor: pointer !important;
            }}

            .stButton > button[kind="secondary"],
            .stButton > button:not([kind="primary"]) {{
                border: 1.5px solid #99a7b9 !important;
                background-color: #b2bccb !important;
                color: #ffffff !important;
            }}

            .stButton > button[kind="primary"] {{
                border: 1.5px solid #52606d !important;
                background-color: #52606d !important;
                color: #ffffff !important;
                box-shadow: -4px 4px 10px -3px rgba(0,0,0,0.18) !important;
            }}

            .stButton > button:hover {{
                background-color: #334155 !important;
                border-color: #1e293b !important;
                box-shadow: 0 8px 20px rgba(0,0,0,0.2) !important;
                transform: scale(1.02) translateY(-2px) !important;
            }}

            .stButton > button,
            .stButton > button span,
            .stButton > button p,
            .stButton > button div {{
                font-size: 0.85rem !important;
                font-weight: 400 !important;
                line-height: 1 !important;
                letter-spacing: 0 !important;
            }}

            .stButton > button:hover,
            .stButton > button:hover span,
            .stButton > button:hover p,
            .stButton > button:hover div {{
                font-weight: 700 !important;
            }}

            .stButton > button * {{
                color: #ffffff !important;
                fill: #ffffff !important;
                font-size: inherit !important;
                font-weight: inherit !important;
                line-height: inherit !important;
                letter-spacing: inherit !important;
            }}
            {debug_overlay_css}
            {debug_hover_index_css}

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
        "s_mode": "",
        "global_edit_mode": False,
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

def keep_search_widget_state():
    for key in [
        "f_sponsor",
        "f_ta",
        "f_phase",
        "f_year",
        "f_nct_id",
        "s_registry",
        "s_mode",
    ]:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]

def go_back_to_results():
    st.session_state.selected_nct_id = None
    st.session_state.trigger_prediction = False
    st.session_state.analysis_result = None
    st.session_state.analysis_nct_id = None
    st.session_state.global_edit_mode = False



def get_risk_tier(score: float):
    if score >= 75: return "Robust", "Strong success patterns detected.", "#f0fdf4", "#166534"
    if score >= 50: return "Favorable", "Favorable historical indicators.", "#eff6ff", "#1e40af"
    if score >= 25: return "Watchlist", "Mixed signals; mitigation required.", "#fff7ed", "#9a3412"
    return "High Risk", "Significant attrition patterns.", "#fde8e8", "#991b1b"

# ==========================
# 4. COMPONENTS
# ==========================

def render_header(is_landing=True, show_predict_button=False, show_back_button=False, show_global_edit_toggle=False):
    logo_path = CURRENT_DIR / "logo_grey_title.png"
    img_base64 = ""
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode()

    t1, t2 = st.columns([3, 2.5])
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

        if show_back_button or show_predict_button or show_global_edit_toggle:
            st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)

            with st.container(key="header_action_buttons"):
                c_toggle, c_back, c_predict = st.columns([1.18, 1.22,2.6], gap="small")

                with c_toggle:
                    if show_global_edit_toggle:
                        st.toggle(
                            "Edit trial fields",
                            key="global_edit_mode"
                        )

                with c_back:
                    if show_back_button:
                        st.button(
                            "Back to Results",
                            use_container_width=True,
                            key="header_back_btn",
                            on_click=go_back_to_results
                        )

                with c_predict:
                    if show_predict_button:
                        if st.button(
                            "Predict Trial Completion",
                            use_container_width=True,
                            type="primary",
                            key="header_predict_btn"
                        ):
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
            f"<div style='text-align:right; font-size:0.8rem; color:#cbd5e1; margin-top: 50px; margin-bottom: 10px;'>{len(curr_df):,} trials matching criteria</div>",
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

def get_edited_row(row):
    edited_row = row.copy()
    trial_key = st.session_state.get("selected_nct_id", "no_trial")

    # 1. Update from Smart Info Boxes (inputs and selectboxes)
    for key in st.session_state:
        if key.startswith(f"input_{trial_key}_"):
            field_id = key.replace(f"input_{trial_key}_", "")
            val = st.session_state[key]

            meta = TAXONOMY.get(field_id, {})
            options = meta.get("ui", {}).get("options")
            if options:
                # Map label back to code for ML compatibility
                for opt in options:
                    if opt[1] == val:
                        # If field_id ends with _ml, we store the code
                        # If it's a UI field, we store the label
                        if field_id.endswith("_ml"):
                            edited_row[field_id] = opt[0]
                        else:
                            edited_row[field_id] = opt[1]
                        break
            else:
                edited_row[field_id] = val

    # 2. Update from Scroll Panels (text areas)
    panel_map = {
        "top_title": "title",
        "study_summary": "summary_ui",
        "conditions": "conditions_ui",
        "interventions": "interventions_ui",
        "primary_outcomes": "primary_outcomes_ui",
        "eligibility_criteria": "criteria_ui"
    }
    for panel_key, col in panel_map.items():
        text_key = f"text_{trial_key}_{panel_key}"
        if text_key in st.session_state:
            edited_row[col] = st.session_state[text_key]

    return edited_row

def render_smart_info_box(label, field_id, row, min_h=48):
    """Dynamically renders a field as a selectbox, text_input, or static box based on taxonomy and edit mode."""
    trial_key = st.session_state.get("selected_nct_id", "no_trial")
    state_key = f"input_{trial_key}_{field_id}"
    is_edit = st.session_state.get("global_edit_mode", False)

    # Get initial value from row
    display_col = field_id.replace("_ml", "_ui") if "_ml" in field_id else f"{field_id}_ui"
    initial_val = trial_val(row, display_col, field_id)

    # Initialize state if not present
    if state_key not in st.session_state:
        st.session_state[state_key] = initial_val

    if is_edit:
        meta = TAXONOMY.get(field_id, {})
        options = meta.get("ui", {}).get("options")

        st.markdown(f"<div style='margin-bottom:4px; color:#475569; font-size:0.80rem; font-weight:700;'>{label}</div>", unsafe_allow_html=True)

        if options:
            # Categorical dropdown
            labels = [opt[1] for opt in options]
            try:
                curr = st.session_state[state_key]
                idx = labels.index(curr) if curr in labels else 0
            except:
                idx = 0
            st.selectbox(label, options=labels, index=idx, key=state_key, label_visibility="collapsed")
        else:
            # Simple text input
            st.text_input(label, key=state_key, label_visibility="collapsed")
    else:
        # Static display using original styling
        render_compact_info_box(label, st.session_state.get(state_key, initial_val), min_h=min_h)

def render_pillar_expander(title, pillar_name, data):
    feats = sorted([(f_id, f_m) for f_id, f_m in TAXONOMY.items() if f_m.get("ui", {}).get("pillar") == pillar_name],
                   key=lambda x: (x[1].get("ui", {}).get("subgroup", ""), x[1].get("ui", {}).get("priority", 99)))
    with st.expander(title, expanded=False):
        for i in range(0, len(feats), 2):
            cols = st.columns(2)
            for j in range(2):
                if i+j < len(feats):
                    f_id, f_m = feats[i+j]
                    label = f_m.get("ui", {}).get("label", f_id)
                    with cols[j]:
                        render_smart_info_box(label, f_id, data, min_h=40)

def open_trial_third_ui(selected_id):
    st.session_state.selected_nct_id = selected_id
    st.session_state.trigger_prediction = False
    st.session_state.analysis_result = None
    st.session_state.analysis_nct_id = None
    st.rerun()


def trial_val(row, *candidates, default="N/A"):
    for col in candidates:
        if col in row.index:
            val = row[col]
            if pd.notna(val) and str(val).strip() != "":
                if isinstance(val, float) and float(val).is_integer():
                    return str(int(val))
                return str(val)
    return default

def render_compact_info_box(label, value, min_h=48):
    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            border:1px solid #cbd5e1;
            border-radius:9px;
            padding:9px 11px;
            min-height:{min_h}px;
            margin-bottom:8px;
            box-shadow:-4px 4px 10px -4px rgba(0,0,0,0.10);
        ">
            <div style="
                color:#64748b;
                font-size:0.66rem;
                text-transform:uppercase;
                font-weight:800;
                letter-spacing:0.05em;
                margin-bottom:4px;
                line-height:1.1;
            ">{label}</div>
            <div style="
                color:#334155;
                font-size:0.84rem;
                line-height:1.25;
                font-weight:500;
                word-break:break-word;
            ">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_scroll_panel(label, value, height=180, key=None):
    trial_key = st.session_state.get("selected_nct_id", "no_trial")
    panel_key = key or label.lower().replace(" ", "_")
    text_key = f"text_{trial_key}_{panel_key}"

    safe_value = "" if value == "N/A" else str(value)

    if text_key not in st.session_state:
        st.session_state[text_key] = safe_value

    st.markdown(
        f"<div style='color:#475569; font-size:0.80rem; font-weight:700; margin-bottom:6px;'>{label}</div>",
        unsafe_allow_html=True
    )

    st.text_area(
        label,
        key=text_key,
        height=height,
        label_visibility="collapsed",
        disabled=not st.session_state.get("global_edit_mode", False)
    )

def render_top_meta_field(label, field_id, row):
    trial_key = st.session_state.get("selected_nct_id", "no_trial")
    state_key = f"input_{trial_key}_{field_id}"
    is_edit = st.session_state.get("global_edit_mode", False)

    display_col = field_id.replace("_ml", "_ui") if "_ml" in field_id else f"{field_id}_ui"
    initial_val = trial_val(row, display_col, field_id)

    if state_key not in st.session_state:
        st.session_state[state_key] = initial_val

    safe_label = html.escape(label)
    safe_value = html.escape(str(st.session_state.get(state_key, initial_val)))

    c_label, c_value = st.columns([0.68, 1.52], gap="small")

    with c_label:
        st.markdown(
            f"""
            <div style="
                height:34px;
                display:flex;
                align-items:center;
                justify-content:flex-end;
                color:#64748b;
                font-size:0.62rem;
                font-weight:800;
                text-transform:uppercase;
                letter-spacing:0.05em;
                white-space:nowrap;
                padding-right:4px;
                line-height:1;
            ">{safe_label}</div>
            """,
            unsafe_allow_html=True
        )

    with c_value:
        meta = TAXONOMY.get(field_id, {})
        options = meta.get("ui", {}).get("options")

        if is_edit:
            if options:
                labels = [opt[1] for opt in options]
                curr = st.session_state[state_key]
                idx = labels.index(curr) if curr in labels else 0
                st.selectbox(
                    label,
                    options=labels,
                    index=idx,
                    key=state_key,
                    label_visibility="collapsed"
                )
            else:
                st.text_input(
                    label,
                    key=state_key,
                    label_visibility="collapsed"
                )
        else:
            st.markdown(
                f"""
                <div style="
                    height:34px;
                    display:flex;
                    align-items:center;
                    background:#ffffff;
                    border:1px solid #cbd5e1;
                    border-radius:8px;
                    padding:0 10px;
                    color:#334155;
                    font-size:0.79rem;
                    font-weight:500;
                    box-shadow:-4px 4px 10px -4px rgba(0,0,0,0.10);
                    overflow:hidden;
                    white-space:nowrap;
                    text-overflow:ellipsis;
                ">{safe_value}</div>
                """,
                unsafe_allow_html=True
            )



def render_top_identity_line(row):
    safe_nct = html.escape(trial_val(row, "nct_id"))
    safe_identity = html.escape(trial_val(row, "ui_search_label"))

    st.markdown(
        (
            f"<div style='display:inline-flex; align-items:baseline; gap:16px; margin:0; flex-wrap:wrap;'>"
            f"<span style='color:#52606d; font-size:1.0rem; font-weight:800; letter-spacing:0.02em; line-height:1.18; white-space:nowrap;'>{safe_nct}</span>"
            f"<span style='color:#475569; font-size:1.0rem; font-weight:700; line-height:1.18; letter-spacing:-0.01em;'>{safe_identity}</span>"
            f"</div>"
        ),
        unsafe_allow_html=True
    )

def render_top_title_panel(row):
    trial_key = st.session_state.get("selected_nct_id", "no_trial")
    text_key = f"text_{trial_key}_top_title"

    current_value = trial_val(row, "title")
    current_value = "" if current_value == "N/A" else str(current_value)

    if text_key not in st.session_state:
        st.session_state[text_key] = current_value

    with st.container(key="trial_title_shell"):
        st.markdown(
            "<div style='color:#64748b; font-size:0.66rem; font-weight:800; text-transform:uppercase; letter-spacing:0.08em; margin:0 0 7px 0; line-height:1;'>Title</div>",
            unsafe_allow_html=True
        )

        st.text_area(
            "Title",
            key=text_key,
            height=86,
            label_visibility="collapsed",
            disabled=not st.session_state.get("global_edit_mode", False)
        )


def render_trial_top_strip_refined(row):
    with st.container(key="trial_top_strip"):

        with st.container(key="trial_title_identity"):
            render_top_identity_line(row)

        left, right = st.columns([3.62, 1.08], gap="small")

        with left:
            render_top_title_panel(row)

        with right:
            with st.container(key="trial_meta_shell"):
                with st.container(key="trial_top_meta"):
                    render_top_meta_field("Sponsor", "lead_sponsor_canonical", row)
                    render_top_meta_field("Phase", "phase_ml", row)
                    render_top_meta_field("Start Date", "start_date", row)


def render_trial_detail_tabs_refined(row):
    render_trial_top_strip_refined(row)

    tab1, tab2, tab3 = st.tabs([
        "Summary",
        "Study Information",
        "Population Details"
    ])

    with tab1:
        meta_c1, meta_c2, meta_c3 = st.columns([1, 1, 1], gap="small")
        with meta_c1:
            render_smart_info_box("Therapeutic Area", "therapeutic_area_ml", row, min_h=48)
        with meta_c2:
            st.empty()
        with meta_c3:
            st.empty()

        left, right = st.columns([1.15, 1.0], gap="medium")

        with left:
            render_scroll_panel(
                "Study Summary",
                trial_val(row, "summary_ui"),
                height=250
            )

        with right:
            render_scroll_panel(
                "Conditions",
                trial_val(row, "conditions_ui"),
                height=118
            )
            render_scroll_panel(
                "Interventions",
                trial_val(row, "interventions_ui"),
                height=118
            )

        render_scroll_panel(
            "Primary Outcomes",
            trial_val(row, "primary_outcomes_ui"),
            height=165
        )

    with tab2:
        c1, c2, c3, c4 = st.columns(4, gap="small")

        with c1:
            render_smart_info_box("Allocation", "allocation_ml", row)
            render_smart_info_box("Intervention Model", "intervention_model_ml", row)

        with c2:
            render_smart_info_box("Masking", "masking_ml", row)
            render_smart_info_box("Number of Arms", "number_of_arms_ml", row)

        with c3:
            render_smart_info_box("Has DMC", "has_dmc_ml", row)
            render_smart_info_box("Has Placebo", "has_placebo_ml", row)

        with c4:
            render_smart_info_box("Includes US", "includes_us_ml", row)
            render_smart_info_box("Healthy Volunteers", "healthy_volunteers_ml", row)

    with tab3:
        p1, p2, p3 = st.columns(3, gap="small")
        with p1:
            render_smart_info_box("Minimum Age", "minimum_age", row)
        with p2:
            render_smart_info_box("Maximum Age", "maximum_age", row)
        with p3:
            render_smart_info_box("Gender", "gender_ml", row)

        render_scroll_panel(
            "Eligibility Criteria",
            trial_val(row, "criteria_ui"),
            height=295
        )


def render_fourth_ui(row):
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
    with c1:
        render_pillar_expander("Therapeutic Context", "Therapeutic Context", row)
    with c2:
        render_pillar_expander("Execution Framework", "Execution Framework", row)

    c3, c4 = st.columns(2)
    with c3:
        render_pillar_expander("Scientific Challenge", "Scientific Challenge", row)
    with c4:
        render_pillar_expander("Patient Profile", "Patient Profile", row)

    if st.session_state.trigger_prediction or st.session_state.get("analysis_result"):
        if (
            not st.session_state.get("analysis_result")
            or st.session_state.get("analysis_nct_id") != st.session_state.selected_nct_id
        ):
            with st.spinner("Analyzing signals..."):
                try:
                    # Merge edited fields into the row before sending to API
                    row_to_predict = get_edited_row(row)
                    res = requests.post(API_URL, json=row_to_predict.replace({np.nan: None}).to_dict())
                    if res.status_code == 200:
                        st.session_state.analysis_result = res.json()
                        st.session_state.analysis_nct_id = st.session_state.selected_nct_id
                        st.session_state.trigger_prediction = False
                    else:
                        st.error(f"API Error: {res.status_code}")
                except Exception as e:
                    st.error(f"System Error: {e}")

        if st.session_state.get("analysis_result"):
            res = st.session_state.analysis_result
            score = res.get("score", 0)
            tier, desc, bg, tc = get_risk_tier(score)

            st.markdown("<hr style='margin: 40px 0;'>", unsafe_allow_html=True)
            cl, cr = st.columns([1.0, 1.4])

            with cl:
                st.plotly_chart(
                    plot_success_gauge(score),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )
                st.markdown(
                    f"<div style='background:{bg}; color:{tc}; padding:20px; border-radius:12px; border:1px solid {tc + '22'};'><div style='font-size:1.4rem; font-weight:800;'>{tier}</div><div>{desc}</div></div>",
                    unsafe_allow_html=True
                )

            with cr:
                if res.get("pillar_impacts"):
                    st.plotly_chart(
                        plot_impact_bar(pd.DataFrame(res["pillar_impacts"])),
                        use_container_width=True
                    )

                # Future fourth UI:
                # add your treemap here when ready
                # for example:
                # if res.get("treemap_data"):
                #     st.plotly_chart(
                #         plot_treemap(pd.DataFrame(res["treemap_data"])),
                #         use_container_width=True
                #     )





# ==========================
# 5. MAIN UI FLOW
# ==========================
init_session_state()
inject_custom_styles()

# Landing or Detail Logic
if not st.session_state.selected_nct_id:
    x_base = X_ALL.copy()

    registry_mode = st.session_state.get("s_registry", "").strip().lower()
    analysis_mode = st.session_state.get("s_mode", "").strip().lower()

    include_ongoing = (registry_mode == "all")
    include_incorrect_historical = (analysis_mode == "all")

    historical_mask = x_base["trial_segment"] != "ONGOING"
    ongoing_mask = x_base["trial_segment"] == "ONGOING"

    if include_incorrect_historical:
        historical_df = x_base[historical_mask]
    else:
        historical_df = x_base[historical_mask & (x_base["is_correct"] == True)]

    if include_ongoing:
        ongoing_df = x_base[ongoing_mask]
        x_base = pd.concat([historical_df, ongoing_df], ignore_index=True)
    else:
        x_base = historical_df.copy()

    render_header(is_landing=not st.session_state.search_initiated)


    if not st.session_state.search_initiated:
        st.markdown('''
            <div class="highlight-box mission-box" style="margin-top: 1.2rem; margin-bottom: 1rem;">
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

        st.markdown(
            f"<div style='margin:10px 0 -20px 0; color:#94a3b8; font-weight:600; font-size:0.7rem;'>{len(filtered_df):,} Matching Trials</div>",
            unsafe_allow_html=True
        )
        st.markdown("<div style='height: 0px;'></div>", unsafe_allow_html=True)
        selected_id = render_trials_grid(filtered_df)
        if selected_id:
            open_trial_third_ui(selected_id)
else:
    keep_search_widget_state()

    selected_df = X_ALL[X_ALL[ID_COL] == st.session_state.selected_nct_id]

    render_header(
        is_landing=False,
        show_predict_button=True,
        show_back_button=True,
        show_global_edit_toggle=True
    )


    if selected_df.empty:
        st.warning("Selected trial not found.")
    else:
        row = selected_df.iloc[0]
        render_trial_detail_tabs_refined(row)
