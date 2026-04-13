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


# ==========================
# 2. STYLES (Consolidated)
# ==========================
def inject_custom_styles():
    debug_overlay_css = """
            /* =========================
               DEBUG OVERLAP VISUALIZER
               ========================= */


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


    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

            .material-symbols-outlined {{
                font-family: 'Material Symbols Outlined' !important;
                font-weight: normal; font-style: normal; font-size: 24px; line-height: 1;
                display: inline-block; white-space: nowrap; direction: ltr; -webkit-font-smoothing: antialiased;
            }}

            :root {{
                --app-bg: #f1f5f9;
                --panel-bg: #717d8b;
                --panel-border: #606c7a;
                --ui-control-h: 38px;

                --ui-control-radius: 8px;
                --ui-control-border: 1.5px solid #94a3b8;
                --ui-control-shadow: -4px 4px 10px -4px rgba(0,0,0,0.10);
                --ui-control-font-size: 0.80rem;
                --ui-control-text: #334155;
                --ui-stack-label-gap: 6px;
                --ui-field-gap: 10px;
                --ui-title-label-gap: 2px;
                --ui-title-control-h: 96px;
                --ui-header-nonlanding-h: 56px;
                --ui-nonlanding-header-top-pad: 10px;
                --ui-nonlanding-body-gap: 0px;
                --ui-meta-shell-pad-top: 0px;
                --ui-meta-shell-pad-right: 10px;
                --ui-meta-shell-pad-bottom: 0px;
                --ui-meta-shell-pad-left: 10px;
                --ui-meta-top-gap: 26px;
                --ui-meta-row-gap: 24px;
                --ui-meta-bottom-gap: 26px;
                --ui-meta-inline-control-h: 28px;
                --ui-meta-label-pad-right: 15px;
                --ui-meta-label-y-offset: -6px;


            }}

            html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {{
                background-color: var(--app-bg) !important;
                color: #334155 !important;
            }}

            .block-container {{
                background: transparent !important;
                padding-top: 2rem !important;
            }}

            [data-testid="stHeader"] {{
                background-color: rgba(0,0,0,0) !important;
                color: #334155 !important;
            }}

            section[data-testid="stSidebar"],
            section[data-testid="stSidebar"] > div,
            div[data-testid="stSidebarContent"] {{
                background-color: var(--panel-bg) !important;
            }}

            section[data-testid="stSidebar"] {{
                border-right: 1px solid var(--panel-border) !important;
            }}

            div[data-testid="stSidebarContent"] {{
                padding-top: -0px !important;
                transform: translateY(0px) !important;
            }}

            section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
                gap: 0rem !important;
            }}



            /* SELECTBOXES + TEXT INPUTS */
            div[data-baseweb="select"],
            [data-testid="stSelectbox"],
            [data-testid="stTextInput"],
            [data-testid="stTextInputRootElement"] {{
                margin: 0 !important;
            }}

            div[data-baseweb="select"] {{
                margin-top: 0 !important;
            }}


            div[data-baseweb="select"] > div,
            [data-testid="stTextInputRootElement"] {{
                background-color: #ffffff !important;
                border: var(--ui-control-border) !important;
                border-radius: var(--ui-control-radius) !important;
                min-height: var(--ui-control-h) !important;
                height: var(--ui-control-h) !important;
                box-sizing: border-box !important;
                font-size: var(--ui-control-font-size) !important;
                color: var(--ui-control-text) !important;
                box-shadow: var(--ui-control-shadow) !important;
                transition: all 0.2s !important;
            }}


            div[data-baseweb="select"] > div {{
                padding-top: 0 !important;
                padding-bottom: 0 !important;
                display: flex !important;
                align-items: center !important;
            }}

            div[data-baseweb="select"] > div > div {{
                align-items: center !important;
            }}

            div[data-baseweb="select"] span,
            div[data-baseweb="select"] input,
            [data-testid="stTextInputRootElement"] input {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
                font-size: var(--ui-control-font-size) !important;
                font-weight: 500 !important;
                line-height: 1.1 !important;
                color: var(--ui-control-text) !important;
                letter-spacing: 0 !important;
            }}



            div[data-baseweb="select"] input {{
                padding-top: 0 !important;
                padding-bottom: 0 !important;
                line-height: 1.1 !important;
            }}

            div[data-baseweb="select"] > div > div:first-child {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
                font-size: var(--ui-control-font-size) !important;
                font-weight: 500 !important;
                line-height: 1.1 !important;
                color: var(--ui-control-text) !important;
                letter-spacing: 0 !important;
            }}



            [data-testid="stTextInputRootElement"] input {{
                padding: 0 12px !important;
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
                min-height: 100% !important;
                height: 100% !important;
            }}


            [data-testid="stTextInputRootElement"]:has(input:disabled) {{
                opacity: 1 !important;
                background-color: #ffffff !important;
                cursor: default !important;
            }}

            [data-testid="stTextInputRootElement"] input:disabled {{
                opacity: 1 !important;
                -webkit-text-fill-color: var(--ui-control-text) !important;
                color: var(--ui-control-text) !important;
                background: transparent !important;
                cursor: default !important;
            }}




            /* GLOBAL MULTISELECT DROPDOWN ALIGNMENT */
            [data-baseweb="popover"] li,
            div[data-baseweb="select"] ul li,
            div[role="listbox"] li {{
                font-size: 0.80rem !important;
            }}


            /* Filter Panel Styles */
            .st-key-filter_header {{
                background-color: var(--panel-bg) !important;
                border: 1px solid var(--panel-border) !important;
                border-radius: 14px !important;
                padding: 22px 24px 28px 24px !important;
                box-shadow: -6px 6px 12px -3px rgba(0,0,0,0.12) !important;
                margin-bottom: 0rem !important;
            }}

            .st-key-filter_header .highlight-title {{
                color: #ffffff !important;
            }}

            .st-key-filter_body {{
                background-color: var(--panel-bg) !important;
                border: 1px solid var(--panel-border) !important;
                border-radius: 14px !important;
                padding: 12px 25px 18px 25px !important;
                box-shadow: -6px 6px 12px -3px rgba(0,0,0,0.12) !important;
                margin-top: 0 !important;
                margin-bottom: 4px !important;
            }}

            .right-column-stack {{
                display: flex !important;
                flex-direction: column !important;
                gap: 1rem !important;
            }}

            .right-column-stack .highlight-box {{
                margin: 0 !important;
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

            .st-key-sidebar_reset_wrap {{
                margin-top: -12px !important;
                margin-bottom: 0px !important;
            }}

            .st-key-sidebar_filters {{
                margin-top: 46px !important;
            }}


            .st-key-sidebar_filters div[data-baseweb="select"] {{
                margin-top: 6px !important;
            }}

            .st-key-sidebar_filters [data-testid="stElementContainer"] {{
                margin-bottom: 10px !important;
            }}

            .st-key-sidebar_filters [data-testid="stElementContainer"]:last-child {{
                margin-bottom: 300px !important;
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



            .ui-field-label {{
                width: 100%;
            }}

            .ui-field-label--meta {{
                height: 100%;
                min-height: 100%;
                display: flex;
                align-items: center;
                justify-content: flex-end;
                color: #475569;
                font-size: 0.80rem;
                font-weight: 700;
                line-height: 1.15;
                letter-spacing: -0.01em;
                white-space: nowrap;
                padding-right: var(--ui-meta-label-pad-right);
                margin: 0;
                text-transform: none;
                transform: translateY(var(--ui-meta-label-y-offset));
            }}

            .ui-field-label--stack {{
                display: block;
                color: #475569;
                font-size: 0.80rem;
                font-weight: 700;
                line-height: 1.15;
                margin: 0 0 var(--ui-stack-label-gap) 0;
            }}

            [class*="st-key-ui_field_box_"] {{
                margin-bottom: var(--ui-field-gap) !important;
            }}

            [class*="st-key-ui_field_box_"]:last-child {{
                margin-bottom: 0 !important;
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

            .st-key-app_header_landing {{
                display: flex !important;
                align-items: flex-start !important;
                margin: 0 !important;
                padding: 10px 0 0 0 !important;
            }}

            .st-key-app_header_nonlanding {{
                display: flex !important;
                align-items: center !important;
                margin: 0 !important;
                padding: var(--ui-nonlanding-header-top-pad) 0 0 0 !important;
                min-height: var(--ui-header-nonlanding-h) !important;
            }}

            .st-key-app_header_landing [data-testid="stVerticalBlock"],
            .st-key-app_header_nonlanding [data-testid="stVerticalBlock"] {{
                gap: 0 !important;
            }}

            .st-key-header_action_buttons [data-testid="stVerticalBlock"] {{
                gap: 0rem !important;
            }}

            .st-key-header_action_buttons {{
                padding-top: var(--ui-nonlanding-header-top-pad) !important;
            }}

            .st-key-header_action_buttons [data-testid="column"] > div {{
                height: 100% !important;
                display: flex !important;
                flex-direction: column !important;
                justify-content: flex-start !important;
            }}

            .st-key-header_action_buttons [data-testid="stWidgetLabel"] {{
                min-height: 0 !important;
                margin: 0 !important;
                padding: 0 !important;
            }}

            .st-key-header_action_buttons [data-testid="stWidgetLabel"] p,
            .st-key-header_action_buttons label p {{
                font-size: 0.72rem !important;
                line-height: 1 !important;
                white-space: nowrap !important;
            }}


            .st-key-header_action_buttons [data-baseweb="checkbox"] {{
                display: inline-flex !important;
                align-items: center !important;
            }}

            .st-key-header_action_buttons [data-baseweb="checkbox"] > div {{
                display: inline-flex !important;
                align-items: center !important;
            }}

            .st-key-header_action_buttons [data-baseweb="checkbox"] label,
            .st-key-header_action_buttons [data-baseweb="checkbox"] span,
            .st-key-header_action_buttons [data-baseweb="checkbox"] p {{
                display: inline-flex !important;
                align-items: center !important;
                line-height: 1.1 !important;
                margin: 0 !important;
                padding: 0 !important;
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



            /* META SHELL = VISUAL SHELL ONLY */
            .st-key-trial_meta_shell {{
                background-color: #e2e8f0 !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 14px !important;
                box-shadow: -6px 6px 12px -3px rgba(0,0,0,0.12) !important;
                margin: 0 !important;
                padding: 0 !important;
            }}

            .st-key-trial_meta_shell > div,
            .st-key-trial_meta_shell > div > [data-testid="stVerticalBlock"] {{
                margin: 0 !important;
                padding: 0 !important;
                gap: 0 !important;
            }}

            /* META INNER = CONTENT INSET LAYER ONLY */
            .st-key-trial_meta_inner {{
                margin: 0 !important;
                padding: 0 !important;
            }}

            .st-key-trial_meta_inner > div {{
                margin: 0 !important;
                padding:
                    var(--ui-meta-shell-pad-top)
                    var(--ui-meta-shell-pad-right)
                    var(--ui-meta-shell-pad-bottom)
                    var(--ui-meta-shell-pad-left) !important;
            }}

            .st-key-trial_meta_inner > div > [data-testid="stVerticalBlock"] {{
                margin: 0 !important;
                padding: 0 !important;
                gap: 0 !important;
            }}

            .st-key-trial_meta_inner [data-testid="stWidgetLabel"] {{
                min-height: 0 !important;
                margin: 0 !important;
                padding: 0 !important;
                display: none !important;
            }}

            .st-key-trial_meta_inner div[data-baseweb="select"] > div,
            .st-key-trial_meta_inner [data-testid="stTextInputRootElement"] {{
                min-height: var(--ui-meta-inline-control-h) !important;
                height: var(--ui-meta-inline-control-h) !important;
            }}

            .st-key-trial_meta_inner [data-testid="stTextInputRootElement"] input {{
                min-height: 100% !important;
                height: 100% !important;
            }}


            /* EXPLICIT META SPACER BLOCKS */
            .trial-meta-top-gap {{
                height: var(--ui-meta-top-gap);
            }}

            .trial-meta-row-gap {{
                height: var(--ui-meta-row-gap);
            }}

            .trial-meta-bottom-gap {{
                height: var(--ui-meta-bottom-gap);
            }}



            .st-key-trial_top_strip {{
                margin-top: var(--ui-nonlanding-body-gap) !important;
            }}

            .st-key-trial_top_strip > div,
            .st-key-trial_top_strip [data-testid="stVerticalBlock"] {{
                margin-top: 0 !important;
                padding-top: 0 !important;
            }}

            /* TITLE SHELL = GREY ROUNDED CONTAINER */
            .st-key-trial_title_shell {{
                background-color: #e2e8f0 !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 14px !important;
                box-shadow: -6px 6px 12px -3px rgba(0,0,0,0.12) !important;
                margin: 0px 0px 0px 0px !important;
                padding: 10px 10px 10px 10px !important;
            }}

            .st-key-trial_title_shell > div,
            .st-key-trial_title_shell [data-testid="stVerticalBlock"] {{
                width: 100% !important;
                gap: 0 !important;
                margin: 0 !important;
                padding: 0px !important;
            }}


            .top-strip-title-label {{
                color: #475569 !important;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
                font-size: 1.0rem !important;
                font-weight: 700 !important;
                line-height: 1.15 !important;
                letter-spacing: -0.01em !important;
                text-transform: none !important;
                margin: 8px 0 6px 0 !important;
                padding: 0 !important;
                text-align: left !important;
                display: block !important;
                white-space: nowrap !important;
            }}

            .st-key-trial_title_shell [data-testid="stTextArea"] {{
                margin: 0 !important;
                width: 100% !important;
            }}

            .st-key-trial_title_shell .stTextArea textarea {{
                min-height: 0 !important;
                height: 75px !important;
                font-size: 0.90rem !important;
                font-weight: 600 !important;
                line-height: 1.34 !important;
                padding: 10px 12px !important;
            }}

            {debug_overlay_css}


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

    t1, t2 = st.columns([3, 2.5], vertical_alignment="top")
    with t1:
        shell_key = "app_header_landing" if is_landing else "app_header_nonlanding"
        with st.container(key=shell_key):
            size = 72 if is_landing else 44
            border = 4 if is_landing else 2
            radius = 18 if is_landing else 7
            title_size = "2.8rem" if is_landing else "2.5rem"
            logo_gap = "12px" if is_landing else "10px"
            title_demo_gap = "0px" if is_landing else "8px"

            html = f"""
                <div style='display: flex; align-items: center; gap: {logo_gap};'>
                    <div style='background-color: white; border: {border}px solid #52606d; padding: 2px; border-radius: {radius}px; display: flex; align-items: center; justify-content: center; height: {size}px; width: {size}px; flex-shrink: 0; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-top: {"0px" if is_landing else "0px"};'>
                        <img src='data:image/png;base64,{img_base64}' style='height: {size-2}px; filter: {BRAND_FILTER};'>
                    </div>
                    <div style='display: {"block" if is_landing else "flex"}; align-items: {"stretch" if is_landing else "flex-end"}; gap: {title_demo_gap};'>
                        <div style='font-size: {title_size}; font-weight: 800; color: #52606d; line-height: 1; {"margin-top: 0px;" if is_landing else ""}'>CTPredict</div>
                        {"<div style='color: #52606d; font-size: 1.5rem; font-weight: 800; display: flex; align-items: baseline; gap: 15px; margin-top: 0px;'><span style='line-height: 1;'>Late-Stage Clinical Trial Predictive Engine</span><span style='font-size: 0.7rem; color: #94a3b8; text-transform: uppercase;'>demo version</span></div>" if is_landing else "<span style='font-size: 0.7rem; font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; line-height: 1; margin-bottom: 0px;'>Demo Version</span>"}
                    </div>
                </div>
            """
            st.markdown(html, unsafe_allow_html=True)

    with t2:

        if show_back_button or show_predict_button or show_global_edit_toggle:

            with st.container(key="header_action_buttons"):
                c_toggle, c_back, c_predict = st.columns([1.18, 1.22, 2.6], gap="small", vertical_alignment="top")

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
            f"<div style='text-align:right; font-size:0.8rem; color:#cbd5e1; margin-top: 0.5px; margin-bottom: 0px; line-height:1.05;'>{len(curr_df):,} trials matching criteria</div>",
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


def _field_token(field_id):
    trial_key = st.session_state.get("selected_nct_id", "no_trial")
    raw = f"{trial_key}_{field_id}"
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in raw)


def _get_dynamic_field_options(field_id):
    if field_id == "lead_sponsor_canonical" and "lead_sponsor_canonical" in X_ALL.columns:
        sponsors = sorted(
            s.strip()
            for s in X_ALL["lead_sponsor_canonical"].dropna().astype(str).unique()
            if str(s).strip()
        )
        return [(s, s) for s in sponsors]

    return None


def _init_trial_field_state(field_id, row):
    trial_key = st.session_state.get("selected_nct_id", "no_trial")
    state_key = f"input_{trial_key}_{field_id}"

    display_col = field_id.replace("_ml", "_ui") if "_ml" in field_id else f"{field_id}_ui"
    initial_val = trial_val(row, display_col, field_id)

    if state_key not in st.session_state:
        st.session_state[state_key] = initial_val

    meta = TAXONOMY.get(field_id, {})
    options = meta.get("ui", {}).get("options")

    if not options:
        options = _get_dynamic_field_options(field_id)

    return state_key, initial_val, options


def _render_labeled_trial_field(label, field_id, row, layout="stack"):
    state_key, initial_val, options = _init_trial_field_state(field_id, row)
    token = _field_token(field_id)
    safe_label = html.escape(label)

    if layout == "inline":
        with st.container(key=f"ui_meta_row_{token}"):
            c_label, c_value = st.columns([0.74, 1.46], gap="small", vertical_alignment="bottom")

            with c_label:
                with st.container(key=f"ui_meta_label_{token}"):
                    st.markdown(
                        f"<div class='ui-field-label ui-field-label--meta'>{safe_label}</div>",
                        unsafe_allow_html=True
                    )

            with c_value:
                _render_two_state_field_control(
                    label=label,
                    state_key=state_key,
                    initial_val=initial_val,
                    options=options,
                    control_key=f"ui_meta_control_{token}"
                )
    else:
        with st.container(key=f"ui_field_box_{token}"):
            st.markdown(
                f"<div class='ui-field-label ui-field-label--stack'>{safe_label}</div>",
                unsafe_allow_html=True
            )

            _render_two_state_field_control(
                label=label,
                state_key=state_key,
                initial_val=initial_val,
                options=options,
                control_key=f"ui_field_control_{token}"
            )


def _render_two_state_field_control(label, state_key, initial_val, options, control_key):
    is_edit = st.session_state.get("global_edit_mode", False)

    with st.container(key=control_key):
        if options:
            labels = [opt[1] for opt in options]
            current_value = st.session_state.get(state_key, initial_val)

            if current_value not in labels and current_value not in (None, "", "N/A"):
                labels = [current_value] + labels

            selected_index = labels.index(current_value) if current_value in labels else 0

            if is_edit:
                st.selectbox(
                    label,
                    options=labels,
                    index=selected_index,
                    key=state_key,
                    label_visibility="collapsed"
                )
            else:
                readonly_key = f"{state_key}__readonly"
                readonly_value = labels[selected_index] if labels else ""

                st.session_state[readonly_key] = readonly_value

                st.text_input(
                    label,
                    key=readonly_key,
                    label_visibility="collapsed",
                    disabled=True
                )
        else:
            st.text_input(
                label,
                key=state_key,
                label_visibility="collapsed",
                disabled=not is_edit
            )


def render_smart_info_box(label, field_id, row, min_h=48):
    _ = min_h  # kept only for call compatibility
    _render_labeled_trial_field(
        label=label,
        field_id=field_id,
        row=row,
        layout="stack"
    )




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

    safe_nct = html.escape(trial_val(row, "nct_id"))
    safe_identity = html.escape(trial_val(row, "ui_search_label"))

    with st.container(key="trial_title_shell"):
        st.markdown(
            f"<div class='top-strip-title-label'>{safe_nct}&nbsp;&nbsp;&nbsp;{safe_identity}</div>",
            unsafe_allow_html=True
        )
        st.text_area(
            "Title",
            key=text_key,
            height=75,
            label_visibility="collapsed",
            disabled=not st.session_state.get("global_edit_mode", False)
        )


def _render_top_meta_select(label, field_id, row):
    state_key, initial_val, options = _init_trial_field_state(field_id, row)
    is_edit = st.session_state.get("global_edit_mode", False)

    labels = [opt[1] for opt in options] if options else []
    current_value = st.session_state.get(state_key, initial_val)

    if current_value not in labels and current_value not in (None, "", "N/A"):
        labels = [current_value] + labels

    if not labels:
        labels = [""]

    if state_key not in st.session_state:
        st.session_state[state_key] = current_value if current_value in labels else labels[0]

    selected_index = (
        labels.index(st.session_state[state_key])
        if st.session_state.get(state_key) in labels
        else 0
    )

    if is_edit:
        st.selectbox(
            label,
            options=labels,
            index=selected_index,
            key=state_key,
            label_visibility="collapsed"
        )
    else:
        readonly_key = f"{state_key}__readonly"
        readonly_value = labels[selected_index] if labels else ""

        st.session_state[readonly_key] = readonly_value

        st.text_input(
            label,
            key=readonly_key,
            label_visibility="collapsed",
            disabled=True
        )


def _render_top_meta_text(label, field_id, row):
    state_key, initial_val, _ = _init_trial_field_state(field_id, row)

    if state_key not in st.session_state:
        st.session_state[state_key] = initial_val

    st.text_input(
        label,
        key=state_key,
        label_visibility="collapsed",
        disabled=not st.session_state.get("global_edit_mode", False)
    )



def render_top_meta_row(label, field_renderer):
    c_label, c_field = st.columns([0.72, 1.48], gap=None, vertical_alignment="center")

    with c_label:
        st.markdown(
            f"<div class='ui-field-label ui-field-label--meta'>{html.escape(label)}</div>",
            unsafe_allow_html=True
        )

    with c_field:
        field_renderer()


def render_top_meta_panel(row):
    rows = [
        ("Sponsor", lambda: _render_top_meta_select("Sponsor", "lead_sponsor_canonical", row)),
        ("Phase", lambda: _render_top_meta_select("Phase", "phase_ml", row)),
        ("Start Date", lambda: _render_top_meta_text("Start Date", "start_date", row)),
    ]

    with st.container(key="trial_meta_shell"):
        with st.container(key="trial_meta_inner"):
            st.markdown("<div class='trial-meta-top-gap'></div>", unsafe_allow_html=True)

            for idx, (label, field_renderer) in enumerate(rows):
                render_top_meta_row(label, field_renderer)

                if idx < len(rows) - 1:
                    st.markdown(
                        "<div class='trial-meta-row-gap'></div>",
                        unsafe_allow_html=True
                    )

            st.markdown("<div class='trial-meta-bottom-gap'></div>", unsafe_allow_html=True)


def render_trial_top_strip_refined(row):
    with st.container(key="trial_top_strip"):
        left, right = st.columns([3.62, 1.08], gap="small")

        with left:
            render_top_title_panel(row)

        with right:
            render_top_meta_panel(row)



def render_trial_detail_tabs_refined(row):
    render_trial_top_strip_refined(row)

    tab1, tab2, tab3 = st.tabs([
        "Summary",
        "Study Information",
        "Population Details"
    ])

    with tab1:
        conditions_h = 80
        middle_top_h = 170
        middle_bottom_h = 170
        right_bottom_h = 170

        col_1, col_23, col_45 = st.columns([1, 2, 2], gap="small")

        with col_1:
            render_smart_info_box("Therapeutic Area", "therapeutic_area_ml", row)

            render_scroll_panel(
                "Conditions",
                trial_val(row, "conditions_ui"),
                height=conditions_h
            )

        with col_23:
            render_scroll_panel(
                "Study Summary",
                trial_val(row, "summary_ui"),
                height=middle_top_h
            )

            render_scroll_panel(
                "Interventions",
                trial_val(row, "interventions_ui"),
                height=middle_bottom_h
            )

        with col_45:
            col_4, col_5 = st.columns(2, gap="small")

            with col_4:
                render_smart_info_box("Allocation", "allocation_ml", row)
                render_smart_info_box("Intervention Model", "intervention_model_ml", row)
                render_smart_info_box("Number of Arms", "number_of_arms_ml", row)

            with col_5:
                render_smart_info_box("Masking", "masking_ml", row)
                render_smart_info_box("Has Placebo", "has_placebo_ml", row)
                render_smart_info_box("Data Monitoring Committee", "has_dmc_ml", row)

            render_scroll_panel(
                "Primary Outcomes",
                trial_val(row, "primary_outcomes_ui"),
                height=right_bottom_h
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
        render_pillar_expander("Scientific Attempt", "Scientific Attempt", row)
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
            with st.container(key="filter_header"):
                st.markdown('<div class="highlight-title" style="margin:0;">Clinical Trial Selection</div>', unsafe_allow_html=True)
            with st.container(key="filter_body"):
                render_filters(x_base)
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
            with st.container(key="sidebar_reset_wrap"):
                if st.button("Reset Filter", use_container_width=True):
                    reset_filters()
                    st.rerun()

            with st.container(key="sidebar_filters"):
                filtered_df = render_filters(x_base, is_sidebar=True)

            st.text_input("Register", key="s_registry")
            st.text_input("Analysis", key="s_mode")

        st.markdown(
            f"<div style='text-align:left; margin:var(--ui-nonlanding-body-gap) 0 6px 0; color:#94a3b8; font-weight:600; font-size:0.7rem; line-height:1;'>{len(filtered_df):,} trials matching criteria</div>",
            unsafe_allow_html=True
        )

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
