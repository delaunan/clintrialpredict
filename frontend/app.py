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
DETAIL_TAB_INFO = "Trial Information"
DETAIL_TAB_POPULATION = "Population Details"
DETAIL_TAB_SCORE = "Completion Score"


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


DEBUG_OVERLAY = False

TEXTAREA_HEIGHTS = {
    "top_title": 70,
    "conditions": 286,
    "study_summary": 160,
    "interventions": 160,
    "primary_outcomes": 160,
    "eligibility_criteria": 337,
    "completion_prediction_left": 220,
    "completion_prediction_right": 430,
}


# ==========================
# 2. STYLES (Consolidated)
# ==========================

def inject_custom_styles():
    # SIDEBAR HIDER: Removes sidebar and toggle in Landing and Detail Views
    is_landing = not st.session_state.get("search_initiated", False)
    is_detail = st.session_state.get("selected_nct_id") is not None

    hide_sidebar_style = ""
    if is_landing or is_detail:
        hide_sidebar_style = """
            [data-testid="stSidebar"] {
                display: none !important;
            }
            [data-testid="collapsedControl"] {
                display: none !important;
            }
            .stApp [data-testid="stHeader"] {
                left: 0 !important;
                width: 100% !important;
            }
        """

    if is_landing or is_detail:
        hide_sidebar_style = """
            [data-testid="stSidebar"] {
                display: none !important;
            }
            [data-testid="collapsedControl"] {
                display: none !important;
            }
            .stApp [data-testid="stHeader"] {
                left: 0 !important;
                width: 100% !important;
            }
        """

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

    is_edit = st.session_state.get("global_edit_mode", False)
    field_bg = "#ffffff" if is_edit else "#f8fafc"
    field_text = "#334155" if is_edit else "#64748b"


    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

            {hide_sidebar_style}

            :root {{
                --app-bg: #f1f5f9;

                --ui-field-bg: {field_bg};
                --ui-field-text: {field_text};

                --ui-scrollbar-thumb: #cbd5e1;
                --ui-scrollbar-track: #e2e8f0;

                --panel-bg: #717d8b;
                --panel-border: #606c7a;
                --ui-control-h: 38px;

                --ui-control-radius: 10px;
                --ui-control-border: 1px solid #cbd5e1;
                --ui-control-shadow: -4px 4px 10px -4px rgba(0,0,0,0.10);
                --ui-control-font-size: 0.80rem;

                --ui-stack-label-gap: 0px;
                --ui-field-gap: 10px;

                --ui-header-nonlanding-h: 56px;
                --ui-nonlanding-header-top-pad: 10px;
                --ui-nonlanding-body-gap: 0px;
                --ui-meta-shell-pad-top: 0px;
                --ui-meta-shell-pad-right: 10px;
                --ui-meta-shell-pad-bottom: 0px;
                --ui-meta-shell-pad-left: 10px;
                --ui-meta-top-gap: 25px;
                --ui-meta-row-gap: 25px;
                --ui-meta-bottom-gap: 25px;
                --ui-meta-inline-control-h: var(--ui-top-strip-control-h);
                --ui-meta-label-pad-right: 15px;
                --ui-meta-label-y-offset: -6px;
                --ui-top-strip-control-h: 24px;


                --ui-detail-tabs-offset-y: 0px;


                --ui-summary-tab-top-pad: 8px;
                --ui-summary-row-overlap: -8px;
                --ui-population-bottom-extension: 148px;

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
                padding-top: 0px !important;
                transform: translateY(0px) !important;
            }}

            section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
                gap: 0rem !important;
            }}


            /* SELECTBOXES + TEXT INPUTS */
            div[data-baseweb="select"],
            div[data-baseweb="input"],
            div[data-baseweb="base-input"],
            [data-testid="stSelectbox"],
            [data-testid="stTextInput"],
            [data-testid="stTextInputRootElement"] {{
                margin: 0 !important;
            }}

            div[data-baseweb="select"] {{
                margin-top: 0 !important;
            }}

            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] > div,
            div[data-baseweb="base-input"] > div {{
                background-color: var(--ui-field-bg) !important;
                border: var(--ui-control-border) !important;
                border-radius: var(--ui-control-radius) !important;
                min-height: var(--ui-control-h) !important;
                height: var(--ui-control-h) !important;
                box-sizing: border-box !important;
                font-size: var(--ui-control-font-size) !important;
                color: var(--ui-field-text) !important;
                box-shadow: var(--ui-control-shadow) !important;
                transition: none !important;
                opacity: 1 !important;
            }}

            [data-testid="stTextInputRootElement"] {{
                background: transparent !important;
                border: none !important;
                border-radius: 0 !important;
                box-shadow: none !important;
                min-height: 0 !important;
                height: auto !important;
                padding: 0 !important;
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
            div[data-baseweb="select"] > div > div:first-child,
            div[data-baseweb="input"] input,
            div[data-baseweb="base-input"] input,
            div[data-baseweb="input"] input:disabled,
            div[data-baseweb="base-input"] input:disabled,
            [data-testid="stTextInputRootElement"] input,
            [data-testid="stTextInputRootElement"] input:disabled {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
                font-size: var(--ui-control-font-size) !important;
                font-weight: 500 !important;
                line-height: 1.1 !important;
                color: var(--ui-field-text) !important;
                -webkit-text-fill-color: var(--ui-field-text) !important;
                letter-spacing: 0 !important;
                opacity: 1 !important;
                background: transparent !important;
            }}

            div[data-baseweb="input"] input,
            div[data-baseweb="base-input"] input,
            [data-testid="stTextInputRootElement"] input {{
                padding: 0 12px !important;
                margin: 0 !important;
                border: none !important;
                box-shadow: none !important;
                min-height: 0 !important;
                height: auto !important;
                line-height: 1 !important;
                align-self: center !important;
                flex: 1 1 auto !important;
                width: 100% !important;
            }}

            [data-testid="stTextInputRootElement"]:has(input:disabled),
            div[data-baseweb="input"]:has(input:disabled) > div,
            div[data-baseweb="base-input"]:has(input:disabled) > div {{
                background-color: var(--ui-field-bg) !important;
                opacity: 1 !important;
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
                min-height: var(--ui-meta-inline-control-h);
                display: flex;
                align-items: center;
                justify-content: flex-end;
                color: #475569;
                font-size: 0.80rem;
                font-weight: 700;
                line-height: 1.08;
                letter-spacing: -0.01em;
                white-space: normal;
                overflow-wrap: anywhere;
                word-break: break-word;
                text-align: right;
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

            [class*="st-key-meta_native_field_"] {{
                margin-bottom: 10px !important;
            }}

            [class*="st-key-meta_native_field_"]:last-child {{
                margin-bottom: 0 !important;
            }}

            [class*="st-key-meta_native_field_"] [data-testid="stWidgetLabel"] {{
                min-height: 0 !important;
                margin: 0 0 3px 0 !important;
                padding: 0 !important;
                display: block !important;
            }}

            [class*="st-key-meta_native_field_"] [data-testid="stWidgetLabel"] p,
            [class*="st-key-meta_native_field_"] label p {{
                color: #475569 !important;
                font-size: 0.80rem !important;
                font-weight: 700 !important;
                line-height: 1.15 !important;
                white-space: nowrap !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
                margin: 0 !important;
            }}


            /* TEXTAREAS — CLEAN RESET */
            [data-testid="stTextArea"] [data-baseweb="textarea"] {{
                border: 1px solid #cbd5e1 !important;
                border-radius: 10px !important;
                background-color: var(--ui-field-bg) !important;
                box-shadow: -4px 4px 10px -4px rgba(0,0,0,0.10) !important;
                overflow: hidden !important;
            }}

            [data-testid="stTextArea"] [data-baseweb="textarea"] > div {{
                background-color: var(--ui-field-bg) !important;
                padding: 0 !important;
            }}

            .stTextArea textarea,
            .stTextArea textarea:disabled {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
                font-size: 0.84rem !important;
                line-height: 1.45 !important;
                font-weight: 500 !important;
                color: var(--ui-field-text) !important;
                -webkit-text-fill-color: var(--ui-field-text) !important;
                background-color: var(--ui-field-bg) !important;
                border: none !important;
                border-radius: 10px !important;
                padding: 4px 12px 10px 12px !important;
                margin: 0 !important;
                resize: vertical !important;
                white-space: pre-wrap !important;
                overflow-wrap: break-word !important;
                overflow-y: auto !important;
                overflow-x: hidden !important;
                scrollbar-width: thin !important;
                scrollbar-color: var(--ui-scrollbar-thumb) var(--ui-scrollbar-track) !important;
                tab-size: 4 !important;
                caret-color: var(--ui-field-text) !important;
                opacity: 1 !important;
                cursor: default !important;
            }}

            [data-testid="stTextArea"] [data-baseweb="textarea"]:has(textarea:disabled),
            [data-testid="stTextArea"] [data-baseweb="textarea"]:has(textarea:disabled) > div,
            [data-testid="stTextArea"] [data-baseweb="textarea"] textarea:disabled {{
                background-color: var(--ui-field-bg) !important;
                opacity: 1 !important;
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


            .st-key-trial_meta_inner div[data-baseweb="select"] > div,
            .st-key-trial_meta_inner div[data-baseweb="input"] > div,
            .st-key-trial_meta_inner div[data-baseweb="base-input"] > div {{
                min-height: var(--ui-meta-inline-control-h) !important;
                height: var(--ui-meta-inline-control-h) !important;
                display: flex !important;
                align-items: center !important;
                padding: 0 !important;
                overflow: hidden !important;
            }}

            .st-key-trial_meta_inner [data-testid="stTextInputRootElement"] input {{
                min-height: 100% !important;
                height: 100% !important;
            }}

            /* SUMMARY SIDE SHELLS = WHITE TAB BOXES, SAME SPACING SYSTEM AS TOP META BOX */
            [class*="st-key-summary_side_shell_"] {{
                background-color: #ffffff !important;
                border: 1px solid #e2e8f0 !important;
                border-radius: 14px !important;
                box-shadow: -6px 6px 12px -3px rgba(0,0,0,0.12) !important;
                margin: 0 !important;
                padding: 0 !important;
            }}


            [class*="st-key-summary_side_shell_"] > div,
            [class*="st-key-summary_side_shell_"] > div > [data-testid="stVerticalBlock"] {{
                margin: 0 !important;
                padding: 0 !important;
                gap: 0 !important;
            }}

            [class*="st-key-summary_side_inner_"] {{
                margin: 0 !important;
                padding: 0 !important;
            }}

            [class*="st-key-summary_side_inner_"] > div {{
                margin: 0 !important;
                padding:
                    var(--ui-meta-shell-pad-top)
                    var(--ui-meta-shell-pad-right)
                    var(--ui-meta-shell-pad-bottom)
                    var(--ui-meta-shell-pad-left) !important;
            }}

            [class*="st-key-summary_side_inner_"] > div > [data-testid="stVerticalBlock"] {{
                margin: 0 !important;
                padding: 0 !important;
                gap: 0 !important;
            }}

            [class*="st-key-summary_side_inner_"] div[data-baseweb="select"] > div,
            [class*="st-key-summary_side_inner_"] div[data-baseweb="input"] > div,
            [class*="st-key-summary_side_inner_"] div[data-baseweb="base-input"] > div {{
                min-height: var(--ui-meta-inline-control-h) !important;
                height: var(--ui-meta-inline-control-h) !important;
                display: flex !important;
                align-items: center !important;
                padding: 0 !important;
                overflow: hidden !important;
            }}

            [class*="st-key-summary_side_inner_"] [data-testid="stTextInputRootElement"] input {{
                min-height: 100% !important;
                height: 100% !important;
            }}

            .st-key-summary_side_inner_design_block [data-baseweb="checkbox"] {{
                display: flex !important;
                align-items: center !important;
                gap: 8px !important;
                min-height: 24px !important;
                padding-left: 6px !important;
            }}

            .st-key-summary_side_inner_design_block [data-baseweb="checkbox"] > div {{
                display: flex !important;
                align-items: center !important;
            }}

            .st-key-summary_side_inner_design_block [data-baseweb="checkbox"] label {{
                display: flex !important;
                align-items: center !important;
                gap: 8px !important;
                margin: 0 !important;
            }}

            .st-key-summary_side_inner_design_block [data-baseweb="checkbox"] p {{
                margin: 0 !important;
                line-height: 1 !important;
            }}


            [class*="st-key-summary_side_inner_"] [data-testid="stTextArea"] {{
                margin: 0 !important;
                width: 100% !important;
            }}

            [class*="st-key-summary_side_inner_"] .stTextArea textarea {{
                padding: 4px 12px 10px 12px !important;
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
                margin: 0 !important;
                padding:
                    var(--ui-meta-shell-pad-top)
                    var(--ui-meta-shell-pad-right)
                    10px
                    var(--ui-meta-shell-pad-left) !important;
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
                font-size: 0.9rem !important;
                font-weight: 800 !important;
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
                font-size: 0.85rem !important;
                font-weight: 600 !important;
                line-height: 1.34 !important;
                padding: 4px 12px 10px 12px !important;
            }}


            /* PREDICT TRIAL COMPLETION — REMOVE ALL BOX SHADOWS */
            .st-key-trial_title_shell,
            .st-key-trial_meta_shell,
            [class*="st-key-summary_side_shell_"],
            .st-key-trial_title_shell [data-testid="stTextArea"] [data-baseweb="textarea"],
            .st-key-trial_top_strip div[data-baseweb="select"] > div,
            .st-key-trial_top_strip div[data-baseweb="input"] > div,
            .st-key-trial_top_strip div[data-baseweb="base-input"] > div,
            [class*="st-key-summary_side_inner_"] div[data-baseweb="select"] > div,
            [class*="st-key-summary_side_inner_"] div[data-baseweb="input"] > div,
            [class*="st-key-summary_side_inner_"] div[data-baseweb="base-input"] > div,
            [class*="st-key-summary_side_inner_"] [data-testid="stTextArea"] [data-baseweb="textarea"] {{
                box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
            }}

            /* DETAIL TABS */
            .st-key-trial_detail_tabs {{
                margin-top: var(--ui-detail-tabs-offset-y) !important;
            }}

            .st-key-trial_detail_tabs .stTabs [data-baseweb="tab-list"] {{
                gap: 8px !important;
                background: transparent !important;
                border: none !important;
                border-radius: 0 !important;
                padding: 3px 0 0 0 !important;
                margin: -3px 0 0px 0 !important;
                box-shadow: none !important;
            }}

            .st-key-trial_detail_tabs .stTabs [data-baseweb="tab"] {{
                height: 37px !important;
                background-color: #e2e8f0 !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 10px !important;
                padding: 0 15px !important;
                color: #64748b !important;
                font-size: 0.85rem !important;
                font-weight: 400 !important;
                line-height: 1 !important;
                letter-spacing: 0 !important;
                transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
                box-shadow: none !important;
            }}

            .st-key-trial_detail_tabs .stTabs [data-baseweb="tab"]:hover {{
                background-color: #334155 !important;
                border-color: #1e293b !important;
                color: #ffffff !important;
                transform: scale(1.02) translateY(-2px) !important;
            }}

            .st-key-trial_detail_tabs .stTabs [data-baseweb="tab"]:hover p,
            .st-key-trial_detail_tabs .stTabs [data-baseweb="tab"]:hover span,
            .st-key-trial_detail_tabs .stTabs [data-baseweb="tab"]:hover div {{
                color: #ffffff !important;
            }}

            .st-key-trial_detail_tabs .stTabs [aria-selected="true"] {{
                background-color: #52606d !important;
                border-color: #52606d !important;
                color: #ffffff !important;
                font-weight: 400 !important;
                line-height: 1 !important;
                box-shadow: -4px 4px 10px -4px rgba(0,0,0,0.10) !important;
            }}

            .st-key-trial_detail_tabs .stTabs [aria-selected="true"] p,
            .st-key-trial_detail_tabs .stTabs [aria-selected="true"] span,
            .st-key-trial_detail_tabs .stTabs [aria-selected="true"] div {{
                color: #ffffff !important;
                font-weight: 400 !important;
                line-height: 1 !important;
            }}

            .st-key-trial_detail_tabs .stTabs [data-baseweb="tab-highlight"] {{
                display: none !important;
            }}

            .st-key-trial_detail_tabs .stTabs [data-baseweb="tab-panel"] {{
                padding-top: var(--ui-summary-tab-top-pad) !important;
                margin-top: 0 !important;
            }}

            .st-key-trial_detail_tabs .st-key-summary_top_row {{
                margin-bottom: var(--ui-summary-row-overlap) !important;
            }}

            .st-key-trial_detail_tabs .st-key-completion_prediction_top_row {{
                margin-bottom: var(--ui-summary-row-overlap) !important;
            }}

            {debug_overlay_css}


        </style>
    """, unsafe_allow_html=True)





# ==========================
# 3. DATA & STATE
# ==========================
@st.cache_data
def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.DataFrame()

    if "start_year" in df.columns:
        df["start_year"] = (
            pd.to_numeric(df["start_year"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    if TAXONOMY_PATH.exists():
        with TAXONOMY_PATH.open("r", encoding="utf-8") as f:
            tax = json.load(f)
    else:
        tax = {}

    return df, tax.get("FIELDS", tax)

X_ALL, TAXONOMY = load_data()


@st.cache_data
def load_logo_base64():
    logo_path = CURRENT_DIR / "logo_grey_title.png"
    if not logo_path.exists():
        return ""

    with logo_path.open("rb") as f:
        return base64.b64encode(f.read()).decode()




def init_session_state():
    defaults = {
        "search_initiated": False,
        "last_search_state": None,
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
        "s_detail": "",
        "global_edit_mode": False,
        "detail_completion_tab_visible": False,
        "detail_prediction_notice": False,
        "detail_active_tab": DETAIL_TAB_INFO,
        "detail_last_nonscore_tab": DETAIL_TAB_INFO,
        "detail_tab_default_request": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

FILTER_COL_MAP = {
    "f_sponsor": "lead_sponsor_canonical",
    "f_ta": "therapeutic_area_ui",
    "f_phase": "phase_ui",
    "f_year": "start_year",
    "f_nct_id": "nct_id",
}


def reset_filters():
    for key in FILTER_COL_MAP:
        st.session_state[key] = None
    for key in ["s_registry", "s_mode", "s_detail"]:
        st.session_state[key] = ""
    st.session_state.selected_nct_id = None
    st.session_state.search_initiated = False
    st.session_state.last_search_state = None



def start_search():
    st.session_state.search_initiated = True
    st.session_state.selected_nct_id = None


def clear_prediction_state():
    st.session_state.trigger_prediction = False
    st.session_state.analysis_result = None
    st.session_state.analysis_nct_id = None

def hide_completion_score_tab():
    current_tab = st.session_state.get("detail_active_tab", DETAIL_TAB_INFO)
    fallback_tab = st.session_state.get("detail_last_nonscore_tab", DETAIL_TAB_INFO)

    if fallback_tab not in {DETAIL_TAB_INFO, DETAIL_TAB_POPULATION}:
        fallback_tab = DETAIL_TAB_INFO

    st.session_state.detail_completion_tab_visible = False
    st.session_state.detail_prediction_notice = False
    st.session_state.detail_tab_default_request = None

    if current_tab == DETAIL_TAB_SCORE:
        st.session_state.detail_active_tab = fallback_tab
    elif current_tab in {DETAIL_TAB_INFO, DETAIL_TAB_POPULATION}:
        st.session_state.detail_active_tab = current_tab
    else:
        st.session_state.detail_active_tab = fallback_tab


def show_completion_score_tab():
    st.session_state.detail_completion_tab_visible = True
    st.session_state.detail_prediction_notice = False
    st.session_state.detail_active_tab = DETAIL_TAB_SCORE
    st.session_state.detail_tab_default_request = DETAIL_TAB_SCORE

def handle_predict_trial_completion():
    if st.session_state.get("global_edit_mode", False):
        hide_completion_score_tab()
        st.session_state.detail_prediction_notice = True
        st.session_state.trigger_prediction = False
        return

    show_completion_score_tab()
    st.session_state.trigger_prediction = True


def reset_trial_editor_state():
    selected_id = st.session_state.get("selected_nct_id")
    if not selected_id:
        return

    selected_df = X_ALL[X_ALL[ID_COL] == selected_id]
    if selected_df.empty:
        return

    row = selected_df.iloc[0]
    trial_key = selected_id

    field_ids = [
        "lead_sponsor_canonical",
        "start_date",
        "therapeutic_area_ml",
        "phase_ml",
        "allocation_ml",
        "intervention_model_ml",
        "number_of_arms_ml",
        "masking_ml",
        "has_placebo_ml",
        "has_dmc_ml",
        "healthy_volunteers_ml",
        "minimum_age",
        "maximum_age",
        "gender_ml",
    ]

    for field_id in field_ids:
        state_key = f"input_{trial_key}_{field_id}"

        if field_id in {"has_placebo_ml", "has_dmc_ml"}:
            initial_val = _coerce_checkbox_value(
                trial_val(row, field_id.replace("_ml", "_ui"), field_id, default=False)
            )
        else:
            display_col = field_id.replace("_ml", "_ui") if "_ml" in field_id else f"{field_id}_ui"
            initial_val = trial_val(row, display_col, field_id)

        st.session_state[state_key] = initial_val

    text_map = {
        "top_title": trial_val(row, "title"),
        "study_summary": trial_val(row, "summary_ui"),
        "conditions": trial_val(row, "conditions_ui"),
        "interventions": trial_val(row, "interventions_ui"),
        "primary_outcomes": trial_val(row, "primary_outcomes_ui"),
        "eligibility_criteria": trial_val(row, "criteria_ui"),
    }

    for suffix, value in text_map.items():
        state_key = f"text_{trial_key}_{suffix}"
        st.session_state[state_key] = "" if value == "N/A" else str(value)


def handle_global_edit_toggle():
    hide_completion_score_tab()

    if not st.session_state.get("global_edit_mode", False):
        reset_trial_editor_state()

def snapshot_search_state():
    st.session_state.last_search_state = {
        "f_sponsor": st.session_state.get("f_sponsor"),
        "f_ta": st.session_state.get("f_ta"),
        "f_phase": st.session_state.get("f_phase"),
        "f_year": st.session_state.get("f_year"),
        "f_nct_id": st.session_state.get("f_nct_id"),
        "s_registry": st.session_state.get("s_registry", ""),
        "s_mode": st.session_state.get("s_mode", ""),
        "s_detail": st.session_state.get("s_detail", ""),
        "search_initiated": st.session_state.get("search_initiated", False),
    }

def restore_search_state():
    saved = st.session_state.get("last_search_state")
    if not saved:
        return

    for key, value in saved.items():
        st.session_state[key] = value

def go_back_to_results():
    restore_search_state()
    st.session_state.selected_nct_id = None
    clear_prediction_state()
    hide_completion_score_tab()
    st.session_state.global_edit_mode = False


def apply_trial_filters(base_df, skip_key=None):
    tdf = base_df.copy()

    # 1. APPLY SECRET "MODE" FILTERS FIRST (Global Constraints)
    registry_mode = str(st.session_state.get("s_registry", "")).strip().lower()
    analysis_mode = str(st.session_state.get("s_mode", "")).strip().lower()

    # Register Filter: Default is Historical only (trial_segment != ONGOING)
    if registry_mode != "all":
        tdf = tdf[tdf["trial_segment"] != "ONGOING"]

    # Analysis Filter: Default is Accurate only (is_correct == True)
    if analysis_mode != "all":
        if "is_correct" in tdf.columns:
            # Logic: Keep if (Correct == True) OR (Accuracy is Pending/Ongoing)
            # Ongoing trials have is_correct as NaN/None
            tdf = tdf[tdf["is_correct"].apply(lambda x:
                str(x).lower() == "true" or
                x == 1 or
                x is True or
                pd.isna(x) or
                str(x).lower() == "none" or
                str(x).lower() == "nan"
            )]

    # 2. APPLY STANDARD DROPDOWN FILTERS
    for state_key, col_name in FILTER_COL_MAP.items():
        val = st.session_state.get(state_key)
        if state_key == skip_key or val in (None, ""):
            continue
        tdf = tdf[tdf[col_name] == val]
    return tdf




def get_risk_tier(score: float):
    if score >= 75: return "Low Risk"
    if score >= 50: return "Favorable"
    if score >= 25: return "Watchlist"
    return "High Risk"


# ==========================
# 4. COMPONENTS
# ==========================




def render_header(is_landing=True, show_predict_button=False, show_back_button=False, show_global_edit_toggle=False):
    img_base64 = load_logo_base64()

    t1, t2 = st.columns([3.8, 3.2], vertical_alignment="top")
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
                    <div style='background-color: white; border: {border}px solid #52606d; padding: 2px; border-radius: {radius}px; display: flex; align-items: center; justify-content: center; height: {size}px; width: {size}px; flex-shrink: 0; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-top: 0px;'>
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
                c_toggle, c_back, c_predict = st.columns([0.75, 1.1, 1.25], gap="small", vertical_alignment="top")

                with c_toggle:
                    if show_global_edit_toggle:
                        st.toggle(
                            "Edit trial fields",
                            key="global_edit_mode",
                            on_change=handle_global_edit_toggle
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
                        st.button(
                            "Predict Trial Completion",
                            use_container_width=True,
                            type="primary",
                            key="header_predict_btn",
                            on_click=handle_predict_trial_completion
                        )



def render_filters(df, is_sidebar=False):
    def apply_filters(base_df, skip_key=None):
        return apply_trial_filters(base_df, skip_key=skip_key)

    def get_opts(col_key):
        tdf = apply_filters(df, skip_key=col_key)
        col = FILTER_COL_MAP[col_key]

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
                on_click=start_search
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
        rowHeight=30,
        headerHeight=28,
        suppressCellFocus=True,
        animateRows=True,
        onRowClicked=JsCode("function(e) { e.api.deselectAll(); e.node.setSelected(true, true); }")
    )

    dynamic_height = min(505, 28 + (len(grid_df) * 30) + 2)
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

            if field_id in {"has_placebo_ml", "has_dmc_ml"} and isinstance(val, bool):
                edited_row[field_id] = int(val)
                continue

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




def render_pillar_expander(title, pillar_name, data, key_suffix=""):
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
                        render_smart_info_box(label, f_id, data, key_suffix=key_suffix)

def open_trial_third_ui(selected_id):
    snapshot_search_state()
    st.session_state.selected_nct_id = selected_id
    clear_prediction_state()
    hide_completion_score_tab()
    st.session_state.global_edit_mode = False
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


def _field_token(field_id, key_suffix=""):
    trial_key = st.session_state.get("selected_nct_id", "no_trial")
    raw = f"{trial_key}_{field_id}_{key_suffix}"
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

def _coerce_checkbox_value(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False

    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "oui"}

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


def _render_labeled_trial_field(label, field_id, row, layout="stack", key_suffix=""):
    state_key, initial_val, options = _init_trial_field_state(field_id, row)
    token = _field_token(field_id, key_suffix=key_suffix)
    safe_label = html.escape(label)

    if layout == "inline":
        with st.container(key=f"ui_meta_row_{token}"):
            c_label, c_value = st.columns([0.72, 1.48], gap=None, vertical_alignment="center")

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
                    control_key=f"ui_meta_control_{token}",
                    key_suffix=key_suffix
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
                control_key=f"ui_field_control_{token}",
                key_suffix=key_suffix
            )


def _render_two_state_field_control(label, state_key, initial_val, options, control_key, key_suffix=""):
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
                    key=f"{state_key}_{key_suffix}" if key_suffix else state_key,
                    label_visibility="collapsed"
                )
            else:
                readonly_key = f"{state_key}__readonly_{key_suffix}" if key_suffix else f"{state_key}__readonly"
                readonly_value = labels[selected_index] if labels else ""

                # SAFE UPDATE: Avoid StreamlitAPIException if rendered multiple times
                try:
                    if st.session_state.get(readonly_key) != readonly_value:
                        st.session_state[readonly_key] = readonly_value
                except st.errors.StreamlitAPIException:
                    pass

                st.text_input(
                    label,
                    key=readonly_key,
                    label_visibility="collapsed",
                    disabled=True
                )
        else:
            st.text_input(
                label,
                key=f"{state_key}_{key_suffix}" if key_suffix else state_key,
                label_visibility="collapsed",
                disabled=not is_edit
            )


def _render_native_meta_field(label, field_id, row, key_suffix=""):
    trial_key = st.session_state.get("selected_nct_id", "no_trial")
    state_key = f"input_{trial_key}_{field_id}"
    token = _field_token(field_id, key_suffix=key_suffix)

    with st.container(key=f"meta_native_field_{token}"):
        if field_id in {"has_placebo_ml", "has_dmc_ml"}:
            initial_val = _coerce_checkbox_value(
                trial_val(row, field_id.replace("_ml", "_ui"), field_id, default=False)
            )

            if state_key not in st.session_state:
                st.session_state[state_key] = initial_val

            st.checkbox(
                label,
                key=f"{state_key}_{key_suffix}" if key_suffix else state_key,
                disabled=not st.session_state.get("global_edit_mode", False)
            )
            return

        state_key, initial_val, options = _init_trial_field_state(field_id, row)
        is_edit = st.session_state.get("global_edit_mode", False)

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
                    key=f"{state_key}_{key_suffix}" if key_suffix else state_key
                )
            else:
                readonly_key = f"{state_key}__readonly_{key_suffix}" if key_suffix else f"{state_key}__readonly"
                readonly_value = labels[selected_index] if labels else ""

                # SAFE UPDATE: Avoid StreamlitAPIException if rendered multiple times
                try:
                    if st.session_state.get(readonly_key) != readonly_value:
                        st.session_state[readonly_key] = readonly_value
                except st.errors.StreamlitAPIException:
                    pass

                st.text_input(
                    label,
                    key=readonly_key,
                    disabled=True
                )
        else:
            st.text_input(
                label,
                key=f"{state_key}_{key_suffix}" if key_suffix else state_key,
                disabled=not is_edit
            )



def _render_native_meta_textarea_field(label, value, state_suffix, height):
    trial_key = st.session_state.get("selected_nct_id", "no_trial")
    state_key = f"text_{trial_key}_{state_suffix}"
    safe_value = "" if value == "N/A" else str(value)

    if state_key not in st.session_state:
        st.session_state[state_key] = safe_value

    with st.container(key=f"meta_native_field_{state_suffix}"):
        st.text_area(
            label,
            key=state_key,
            height=height,
            disabled=not st.session_state.get("global_edit_mode", False)
        )

def render_smart_info_box(label, field_id, row, key_suffix=""):
    _render_labeled_trial_field(
        label=label,
        field_id=field_id,
        row=row,
        layout="stack",
        key_suffix=key_suffix
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
            height=TEXTAREA_HEIGHTS["top_title"],
            label_visibility="collapsed",
            disabled=not st.session_state.get("global_edit_mode", False)
        )


def render_top_meta_panel(row):
    rows = [
        ("Sponsor", "lead_sponsor_canonical"),
        ("Start Date", "start_date"),
    ]

    with st.container(key="trial_meta_shell"):
        with st.container(key="trial_meta_inner"):
            st.markdown("<div class='trial-meta-top-gap'></div>", unsafe_allow_html=True)

            for idx, (label, field_id) in enumerate(rows):
                _render_native_meta_field(
                    label=label,
                    field_id=field_id,
                    row=row
                )

                if idx < len(rows) - 1:
                    st.markdown(
                        "<div class='trial-meta-row-gap'></div>",
                        unsafe_allow_html=True
                    )

            st.markdown("<div class='trial-meta-bottom-gap'></div>", unsafe_allow_html=True)


def render_summary_side_panel(row, rows, panel_suffix, bottom_extension_var=None):
    with st.container(key=f"summary_side_shell_{panel_suffix}"):
        with st.container(key=f"summary_side_inner_{panel_suffix}"):
            st.markdown("<div class='trial-meta-top-gap'></div>", unsafe_allow_html=True)

            for idx, (label, field_id) in enumerate(rows):
                _render_native_meta_field(
                    label=label,
                    field_id=field_id,
                    row=row
                )

                if idx < len(rows) - 1:
                    st.markdown("<div class='trial-meta-row-gap'></div>", unsafe_allow_html=True)

            if bottom_extension_var:
                st.markdown(
                    f"<div style='height: var({bottom_extension_var});'></div>",
                    unsafe_allow_html=True
                )

            st.markdown("<div class='trial-meta-bottom-gap'></div>", unsafe_allow_html=True)


def render_ta_conditions_panel(row):
    with st.container(key="summary_side_shell_ta_conditions_block"):
        with st.container(key="summary_side_inner_ta_conditions_block"):
            st.markdown("<div class='trial-meta-top-gap'></div>", unsafe_allow_html=True)

            _render_native_meta_field(
                label="Therapeutic Area",
                field_id="therapeutic_area_ml",
                row=row
            )

            st.markdown("<div class='trial-meta-row-gap'></div>", unsafe_allow_html=True)

            _render_native_meta_textarea_field(
                label="Conditions",
                value=trial_val(row, "conditions_ui"),
                state_suffix="conditions",
                height=TEXTAREA_HEIGHTS["conditions"]
            )
            st.markdown("<div class='trial-meta-bottom-gap'></div>", unsafe_allow_html=True)


def render_population_side_panel(row):
    render_summary_side_panel(
        row=row,
        rows=[
            ("Population Type", "healthy_volunteers_ml"),
            ("Minimum Age", "minimum_age"),
            ("Maximum Age", "maximum_age"),
            ("Patient Gender Eligibility Status", "gender_ml"),
        ],
        panel_suffix="population_block",
        bottom_extension_var="--ui-population-bottom-extension"
    )


def render_summary_text_shell_panel(label, value, state_suffix, panel_suffix, height):
    with st.container(key=f"summary_side_shell_{panel_suffix}"):
        with st.container(key=f"summary_side_inner_{panel_suffix}"):
            st.markdown("<div class='trial-meta-top-gap'></div>", unsafe_allow_html=True)

            _render_native_meta_textarea_field(
                label=label,
                value=value,
                state_suffix=state_suffix,
                height=height
            )

            st.markdown("<div class='trial-meta-bottom-gap'></div>", unsafe_allow_html=True)

def render_summary_placeholder_panel(panel_suffix, height):
    with st.container(key=f"summary_side_shell_{panel_suffix}"):
        with st.container(key=f"summary_side_inner_{panel_suffix}"):
            st.markdown("<div class='trial-meta-top-gap'></div>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='height: {height}px;'></div>",
                unsafe_allow_html=True
            )
            st.markdown("<div class='trial-meta-bottom-gap'></div>", unsafe_allow_html=True)


def render_box_spacer(height):
    st.markdown(
        f"<div style='height: {height}px;'></div>",
        unsafe_allow_html=True
    )


def render_summary_plot_shell_panel(panel_suffix, body_renderer):
    with st.container(key=f"summary_side_shell_{panel_suffix}"):
        with st.container(key=f"summary_side_inner_{panel_suffix}"):
            st.markdown("<div class='trial-meta-top-gap'></div>", unsafe_allow_html=True)
            body_renderer()
            st.markdown("<div class='trial-meta-bottom-gap'></div>", unsafe_allow_html=True)


def render_trial_top_strip_refined(row):
    with st.container(key="trial_top_strip"):
        left, right = st.columns([3.70, 0.82], gap="xsmall")

        with left:
            render_top_title_panel(row)

        with right:
            render_top_meta_panel(row)


def render_trial_detail_tabs_refined(row):
    render_trial_top_strip_refined(row)

    tab_labels = [DETAIL_TAB_INFO, DETAIL_TAB_POPULATION]
    if st.session_state.get("detail_completion_tab_visible", False):
        tab_labels.append(DETAIL_TAB_SCORE)

    if st.session_state.get("detail_active_tab") not in tab_labels:
        st.session_state.detail_active_tab = st.session_state.get(
            "detail_last_nonscore_tab",
            DETAIL_TAB_INFO
        )

    if st.session_state.get("detail_prediction_notice", False):
        st.warning("Contact owner to know more and try out simulation mode")

    tabs_kwargs = {
        "key": "detail_active_tab",
        "on_change": "rerun",
    }

    forced_default = st.session_state.get("detail_tab_default_request")
    if forced_default in tab_labels:
        tabs_kwargs["default"] = forced_default

    with st.container(key="trial_detail_tabs"):
        tabs = st.tabs(tab_labels, **tabs_kwargs)
        tab_map = dict(zip(tab_labels, tabs))

        current_tab = st.session_state.get("detail_active_tab", DETAIL_TAB_INFO)
        if current_tab in {DETAIL_TAB_INFO, DETAIL_TAB_POPULATION}:
            st.session_state.detail_last_nonscore_tab = current_tab

        if forced_default in tab_labels:
            st.session_state.detail_tab_default_request = None
            st.rerun()

        with tab_map[DETAIL_TAB_INFO]:

            left_col, middle_col, right_col = st.columns([0.82, 2.88, 0.82], gap="xsmall")

            with left_col:
                render_ta_conditions_panel(row)

            with middle_col:
                with st.container(key="summary_top_row"):
                    render_summary_text_shell_panel(
                        label="Study Summary",
                        value=trial_val(row, "summary_ui"),
                        state_suffix="study_summary",
                        panel_suffix="study_summary_block",
                        height=TEXTAREA_HEIGHTS["study_summary"]
                    )

                bottom_left, bottom_right = st.columns(2, gap="xsmall")

                with bottom_left:
                    render_summary_text_shell_panel(
                        label="Interventions",
                        value=trial_val(row, "interventions_ui"),
                        state_suffix="interventions",
                        panel_suffix="interventions_block",
                        height=TEXTAREA_HEIGHTS["interventions"]
                    )

                with bottom_right:
                    render_summary_text_shell_panel(
                        label="Primary Outcomes",
                        value=trial_val(row, "primary_outcomes_ui"),
                        state_suffix="primary_outcomes",
                        panel_suffix="primary_outcomes_block",
                        height=TEXTAREA_HEIGHTS["primary_outcomes"]
                    )

            with right_col:
                render_summary_side_panel(
                    row=row,
                    rows=[
                        ("Phase", "phase_ml"),
                        ("Allocation", "allocation_ml"),
                        ("Intervention Model", "intervention_model_ml"),
                        ("Number of Arms", "number_of_arms_ml"),
                        ("Masking", "masking_ml"),
                        ("Placebo Control", "has_placebo_ml"),
                        ("DMC Involvment Status", "has_dmc_ml"),
                    ],
                    panel_suffix="design_block"
                )

        with tab_map[DETAIL_TAB_POPULATION]:
            left_col, right_col = st.columns([3.70, 0.82], gap="xsmall")

            with left_col:
                render_summary_text_shell_panel(
                    label="Eligibility Criteria",
                    value=trial_val(row, "criteria_ui"),
                    state_suffix="eligibility_criteria",
                    panel_suffix="eligibility_block",
                    height=TEXTAREA_HEIGHTS["eligibility_criteria"]
                )

            with right_col:
                render_population_side_panel(row)

        if DETAIL_TAB_SCORE in tab_map:
            with tab_map[DETAIL_TAB_SCORE]:
                render_completion_prediction_tab(row)


def get_analysis_result_for_selected_trial(row):
    if not (st.session_state.trigger_prediction or st.session_state.get("analysis_result")):
        return None

    if (
        not st.session_state.get("analysis_result")
        or st.session_state.get("analysis_nct_id") != st.session_state.selected_nct_id
    ):
        with st.spinner("Analyzing signals..."):
            try:
                row_to_predict = get_edited_row(row)
                res = requests.post(
                    API_URL,
                    json=row_to_predict.replace({np.nan: None}).to_dict(),
                    timeout=60
                )

                if res.status_code == 200:
                    st.session_state.analysis_result = res.json()
                    st.session_state.analysis_nct_id = st.session_state.selected_nct_id
                    st.session_state.trigger_prediction = False
                else:
                    st.error(f"API Error: {res.status_code}")
                    return None

            except Exception as e:
                st.error(f"System Error: {e}")
                return None

    return st.session_state.get("analysis_result")


def render_completion_prediction_tab(row):
    res = get_analysis_result_for_selected_trial(row)

    left_box_h = TEXTAREA_HEIGHTS["completion_prediction_left"]
    right_box_h = TEXTAREA_HEIGHTS["completion_prediction_right"]

    gauge_plot_h = max(110, left_box_h - 60)
    bar_plot_h = max(120, left_box_h - 20)
    treemap_plot_h = max(260, right_box_h - 40)

    left_col, right_col = st.columns([3, 4], gap="xsmall")

    with left_col:
        with st.container(key="completion_prediction_top_row"):

            def _render_gauge_panel():
                if not res:
                    render_box_spacer(left_box_h)
                    return

                score = res.get("score", 0)
                tier = get_risk_tier(score)

                st.plotly_chart(
                    plot_success_gauge(score, height=gauge_plot_h),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )

                st.markdown(
                    f"<div style='text-align:center; font-family:\"Inter\", sans-serif; font-size:1.0rem; font-weight:800; color:#334155; margin-top:-40px;'>{tier}</div>",
                    unsafe_allow_html=True
                )

            render_summary_plot_shell_panel(
                panel_suffix="completion_prediction_left_top_block",
                body_renderer=_render_gauge_panel
            )

        def _render_bar_panel():
            if not res or not res.get("pillar_impacts"):
                render_box_spacer(left_box_h)
                return

            st.plotly_chart(
                plot_impact_bar(
                    pd.DataFrame(res["pillar_impacts"]),
                    height=bar_plot_h
                ),
                use_container_width=True,
                config={"displayModeBar": False}
            )

        render_summary_plot_shell_panel(
            panel_suffix="completion_prediction_left_bottom_block",
            body_renderer=_render_bar_panel
        )

    with right_col:

        def _render_treemap_panel():
            if not res or not res.get("subcat_impacts") or not res.get("pillar_impacts"):
                render_box_spacer(right_box_h)
                return

            show_detailed = str(st.session_state.get("s_detail", "")).strip().lower() == "true"

            st.plotly_chart(
                plot_treemap(
                    res["subcat_impacts"],
                    res["pillar_impacts"],
                    show_values=show_detailed,
                    height=treemap_plot_h
                ),
                use_container_width=True,
                config={"displayModeBar": False}
            )

        render_summary_plot_shell_panel(
            panel_suffix="completion_prediction_right_block",
            body_renderer=_render_treemap_panel
        )

# ==========================
# 5. MAIN UI FLOW
# ==========================
init_session_state()
inject_custom_styles()


# Main Content Logic
if not st.session_state.selected_nct_id:
    # Use raw data as base; apply_trial_filters handles modes internally
    x_base = X_ALL.copy()

    if not st.session_state.search_initiated:
        render_header(is_landing=True)
        st.markdown('''
            <div class="highlight-box mission-box" style="margin-top: 1.2rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div class="highlight-title">Operational Success & Risk Stratification</div>
                    <div style="font-size:0.65rem; font-weight:800; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em;">Core Mission</div>
                </div>
                <div class="highlight-text">This predictive engine estimates the <b>likelihood of operational completion</b> and the <b>risk of early termination</b> using only data available at clinical trial initiation. Each trial is systematically evaluated and classified into <b>four distinct tiers</b> - High Risk, Watchlist, Favorable, and Low Risk - providing a clear and actionable risk profile.</div>
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
                            <div class="highlight-title">Industry-Scale Public Clinical Data</div>
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
        # RESULTS VIEW: Sidebar filters + Grid
        with st.sidebar:
            with st.container(key="sidebar_reset_wrap"):
                if st.button("Reset Filter", use_container_width=True):
                    reset_filters()
                    st.rerun()

            with st.container(key="sidebar_filters"):
                filtered_df = render_filters(x_base, is_sidebar=True)

            st.text_input("Register", key="s_registry")
            st.text_input("Analysis", key="s_mode")
            st.text_input("", key="s_detail")

        render_header(is_landing=False)

        st.markdown(
            f"<div style='text-align:left; margin:var(--ui-nonlanding-body-gap) 0 6px 0; color:#94a3b8; font-weight:600; font-size:0.7rem; line-height:1;'>{len(filtered_df):,} trials matching criteria</div>",
            unsafe_allow_html=True
        )

        selected_id = render_trials_grid(filtered_df)

        if selected_id:
            open_trial_third_ui(selected_id)
else:


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

# ==========================
# 6. HIDDEN STATE KEEPER
# ==========================
# This ensures that secret mode variables persist when switching to Detail View
# where the primary sidebar widgets are not rendered.
if st.session_state.selected_nct_id:
    with st.sidebar:
        st.text_input("Register", key="s_registry_keeper", value=st.session_state.get("s_registry", ""), label_visibility="collapsed")
        st.text_input("Analysis", key="s_mode_keeper", value=st.session_state.get("s_mode", ""), label_visibility="collapsed")
        st.text_input("", key="s_detail_keeper", value=st.session_state.get("s_detail", ""), label_visibility="collapsed")

        # Sync back to primary keys if keeper changes (unlikely in hidden state)
        st.session_state["s_registry"] = st.session_state["s_registry_keeper"]
        st.session_state["s_mode"] = st.session_state["s_mode_keeper"]
        st.session_state["s_detail"] = st.session_state["s_detail_keeper"]
