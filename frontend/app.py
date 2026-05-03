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
    initial_sidebar_state="expanded"
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

UI_ACCENT_BLUE = "#89A7C9"


# Plot / completion palette mirrors utils.plot STYLE_CONFIG["colors"]
PLOT_BLUE_SOFT_RGB = "rgb(162,198,228)"
PLOT_BLUE_DEEP_RGB = "rgb(47,98,166)"
PLOT_GREY_WARM_RGB = "rgb(242,244,248)"
PLOT_RED_SOFT_RGB = "rgb(236,162,162)"
PLOT_RED_DEEP_RGB = "rgb(176,63,63)"


BRAND_FILTER = (
    f"contrast(1.5) brightness(0.9) grayscale(100%) sepia(100%) "
    f"hue-rotate({HUE}deg) saturate({INTENSITY}) brightness({DARKNESS}) "
    f"contrast(1.2) drop-shadow({THICKNESS}px {THICKNESS}px 0px #52606d) "
    f"drop-shadow(-{THICKNESS}px -{THICKNESS}px 0px #52606d)"
)




TEXTAREA_HEIGHTS = {
    # Compact Python defaults.
    # The final visual heights are then harmonized by the late CSS
    # responsive height-contract block.
    "top_title": 70,
    "conditions": 245,
    "study_summary": 150,
    "interventions": 140,
    "primary_outcomes": 140,
    "eligibility_criteria": 330,
    "completion_prediction_left": 265,
    "completion_prediction_right": 560,
}



COMPLETION_GAUGE_HELP_TOOLTIP = f"""
<p class="tooltip-section">
  <span><b>COMPLETION LIKELIHOOD SCORE</b> generated at trial design stage.</span><br>
  <span><b>Above 50:</b> higher likelihood of full completion.</span><br>
  <span><b>Below 50:</b> higher likelihood of early termination.</span><br>
  <span>Preliminary operational read, based on <b>30,000 precedent trials</b>.</span><br>
</p>

<p class="tooltip-section">
  <span><b>Full trial completion</b> does not always imply scientific success.</span><br>
  <span><b>Early termination</b>, however, may reflect operational strain or emerging signs of scientific underperformance, which can lead to investment reallocation.</span><br>
</p>

<p class="tooltip-section">
  <span><b><span style="color:{PLOT_RED_DEEP_RGB};">RED ZONE / RED DRIVERS.</span></b> Not inherently negative operationally.</span><br>
  <span>Can reflect higher scientific rigor, more ambitious innovation, greater complexity.</span><br>
  <span>Often: higher early-stop risk, but potentially higher value.</span>
</p>

<p class="tooltip-section">
  <span><b><span style="color:{PLOT_BLUE_DEEP_RGB};">BLUE ZONE / BLUE DRIVERS.</span></b> Can reflect strong, well-structured execution.</span><br>
  <span>Can also reflect simpler, more conventional design.</span><br>
  <span>Example: fixed design, no adaptive stopping, lower-risk signal.</span>
</p>
"""

COMPLETION_TIER_SCALE_TOOLTIP = f"""
<p class="tooltip-section">
  <span style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
    <span style="width:10px; height:10px; background:linear-gradient(90deg, {PLOT_BLUE_SOFT_RGB} 0%, {PLOT_BLUE_DEEP_RGB} 100%); border-radius:2px; display:inline-block; flex:0 0 10px;"></span>
    <span style="display:inline-block; width:72px;"><b>Low Risk</b></span>
    <span style="display:inline-block; min-width:52px; text-align:left;">75–100</span>
  </span>

  <span style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
    <span style="width:10px; height:10px; background:linear-gradient(90deg, {PLOT_GREY_WARM_RGB} 0%, {PLOT_BLUE_SOFT_RGB} 100%); border-radius:2px; display:inline-block; flex:0 0 10px;"></span>
    <span style="display:inline-block; width:72px;"><b>Favorable</b></span>
    <span style="display:inline-block; min-width:52px; text-align:left;">50–75</span>
  </span>

  <span style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
    <span style="width:10px; height:10px; background:linear-gradient(90deg, {PLOT_RED_SOFT_RGB} 0%, {PLOT_GREY_WARM_RGB} 100%); border-radius:2px; display:inline-block; flex:0 0 10px;"></span>
    <span style="display:inline-block; width:72px;"><b>Watchlist</b></span>
    <span style="display:inline-block; min-width:52px; text-align:left;">25–50</span>
  </span>

  <span style="display:flex; align-items:center; gap:8px;">
    <span style="width:10px; height:10px; background:linear-gradient(90deg, {PLOT_RED_DEEP_RGB} 0%, {PLOT_RED_SOFT_RGB} 100%); border-radius:2px; display:inline-block; flex:0 0 10px;"></span>
    <span style="display:inline-block; width:72px;"><b>High Risk</b></span>
    <span style="display:inline-block; min-width:52px; text-align:left;">0–25</span>
  </span>
</p>
"""
COMPLETION_WORKFLOW_INFO_HTML = """
<div class="completion-workflow-note">
  <div class="completion-workflow-note-label">Explore Additional Capabilities</div>

  <div class="completion-workflow-note-text">
    To explore <strong>Simulation mode</strong>, access additional trials, or learn more
    about the prediction tool, please contact Nicolas at
    <strong>delaunay80@gmail.com</strong> or via WhatsApp at
    <strong>+33 7 86 72 21 43</strong>.
  </div>

  <ul class="completion-workflow-note-list">
    <li>This pilot tool can also support broader analytical perspectives, including:</li>
    <li>company portfolio views,</li>
    <li>therapeutic area views across the industry,</li>
    <li>market potential views balancing trial risk profile and opportunity, currently under development,</li>
    <li>custom views based on specific questions you would like to explore, <strong>open to and welcoming new ideas!</strong></li>
  </ul>
</div>
"""



TRIAL_EDITOR_FIELD_IDS = [
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

TRIAL_EDITOR_TEXT_FIELDS = {
    "top_title": ("title",),
    "study_summary": ("summary_ui",),
    "conditions": ("conditions_ui",),
    "interventions": ("interventions_ui",),
    "primary_outcomes": ("primary_outcomes_ui",),
    "eligibility_criteria": ("criteria_ui",),
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



    is_edit = is_detail and st.session_state.get("global_edit_mode", False)
    field_bg = "#ffffff" if is_edit else "#f8fafc"
    field_text = "#334155" if is_edit else "#64748b"
    shell_shadow = "0 1px 4px rgba(0,0,0,0.05)" if is_detail else "-6px 6px 12px -3px rgba(0,0,0,0.12)"
    textarea_shadow = "0 1px 4px rgba(0,0,0,0.05)" if is_detail else "-4px 4px 10px -4px rgba(0,0,0,0.10)"
    button_primary_shadow = "-4px 4px 10px -3px rgba(0,0,0,0.18)"
    button_hover_shadow = "0 8px 20px rgba(0,0,0,0.2)"
    selected_tab_shadow = "-4px 4px 10px -4px rgba(0,0,0,0.10)"



    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

            {hide_sidebar_style}

            :root {{
                --app-bg: #f1f5f9;
                --ui-accent-blue: {UI_ACCENT_BLUE};

                /* RESPONSIVE DESIGN SYSTEM
                   Baseline = current laptop design.
                   Large screens scale moderately.
                   Very large screens are capped.
                   Do not use global zoom: scale the composition tokens instead. */
                --ui-page-max-w: 1760px;
                --ui-page-pad-x: clamp(2.5rem, 5vw, 6.5rem);
                --ui-page-pad-top: 2rem;
                --ui-page-pad-bottom: 2rem;

                --ui-card-radius: 14px;
                --ui-card-pad: 24px;
                --ui-filter-header-pad: 22px 24px 28px 24px;
                --ui-filter-body-pad: 12px 25px 18px 25px;
                --ui-card-gap: 1rem;

                /* Landing composition heights.
                   These prevent text wrapping from deciding visual card height. */
                --ui-landing-shell-min-h: calc(100vh - var(--ui-page-pad-top) - var(--ui-page-pad-bottom));
                --ui-landing-lower-section-min-h: clamp(355px, 39vh, 470px);
                --ui-landing-filter-header-min-h: 74px;
                --ui-landing-right-card-min-h: 170px;

                --ui-highlight-title-size: 1.15rem;
                --ui-highlight-text-size: 0.95rem;
                --ui-label-font-size: 0.85rem;
                --ui-kicker-font-size: 0.65rem;

                --ui-button-h: 37px;
                --ui-button-radius: 8px;
                --ui-button-pad-x: 1rem;
                --ui-button-font-size: 0.85rem;

                --ui-logo-size-landing: 72px;
                --ui-logo-size-nonlanding: 44px;
                --ui-logo-border-landing: 4px;
                --ui-logo-border-nonlanding: 2px;
                --ui-logo-radius-landing: 18px;
                --ui-logo-radius-nonlanding: 7px;
                --ui-logo-gap-landing: 12px;
                --ui-logo-gap-nonlanding: 10px;
                --ui-logo-pad: 2px;

                --ui-title-size-landing: 2.8rem;
                --ui-title-size-nonlanding: 2.5rem;
                --ui-subtitle-size-landing: 1.5rem;
                --ui-demo-size: 0.7rem;

                --ui-landing-header-pad-top: 10px;
                --ui-mission-top: 1.2rem;
                --ui-mission-bottom: 1rem;

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
                --ui-nonlanding-header-y-shift: 0px;
                --ui-sidebar-reset-y-shift: 0px;
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

                --ui-treemap-toggle-top: -42px;
                --ui-treemap-toggle-right: 30px;
                --ui-treemap-hint-left: 25px;

                /* Treemap hint only.
                   Increase = move hint DOWN.
                   Decrease = move hint UP.
                   The toggle itself is not modified. */
                --ui-treemap-hint-y-shift: 17px;

                --ui-shell-shadow: {shell_shadow};
                --ui-textarea-shadow: {textarea_shadow};
                --ui-button-primary-shadow: {button_primary_shadow};
                --ui-button-hover-shadow: {button_hover_shadow};
                --ui-selected-tab-shadow: {selected_tab_shadow};

            }}

            /* Large laptop / desktop */
            @media (min-width: 1800px) and (min-height: 950px) {{
                :root {{
                    --ui-page-max-w: 2060px;
                    --ui-page-pad-top: 2.3rem;
                    --ui-page-pad-bottom: 2.3rem;
                    --ui-page-pad-x: clamp(3rem, 5vw, 7.5rem);

                    --ui-control-h: 41px;
                    --ui-control-radius: 11px;
                    --ui-control-font-size: 0.83rem;
                    --ui-field-gap: 11px;

                    --ui-card-radius: 15px;
                    --ui-card-pad: 27px;
                    --ui-filter-header-pad: 25px 27px 31px 27px;
                    --ui-filter-body-pad: 14px 28px 21px 28px;
                    --ui-card-gap: 1.1rem;

                    --ui-highlight-title-size: 1.22rem;
                    --ui-highlight-text-size: 1.00rem;
                    --ui-label-font-size: 0.88rem;
                    --ui-kicker-font-size: 0.68rem;

                    --ui-button-h: 40px;
                    --ui-button-radius: 9px;
                    --ui-button-pad-x: 1.1rem;
                    --ui-button-font-size: 0.88rem;

                    --ui-logo-size-landing: 78px;
                    --ui-logo-size-nonlanding: 48px;
                    --ui-logo-radius-landing: 19px;
                    --ui-logo-radius-nonlanding: 8px;
                    --ui-logo-gap-landing: 13px;

                    --ui-title-size-landing: 3.0rem;
                    --ui-title-size-nonlanding: 2.65rem;
                    --ui-subtitle-size-landing: 1.6rem;
                    --ui-demo-size: 0.73rem;

                    --ui-landing-header-pad-top: 14px;
                    --ui-mission-top: 1.4rem;
                    --ui-mission-bottom: 1.15rem;

                    --ui-landing-lower-section-min-h: clamp(420px, 42vh, 560px);
                    --ui-landing-filter-header-min-h: 84px;
                    --ui-landing-right-card-min-h: 205px;
                }}
            }}

            /* Very large screen, capped */
            @media (min-width: 2250px) and (min-height: 1050px) {{
                :root {{
                    --ui-page-max-w: 2360px;
                    --ui-page-pad-top: 2.7rem;
                    --ui-page-pad-bottom: 2.7rem;
                    --ui-page-pad-x: clamp(4rem, 5vw, 9rem);

                    --ui-control-h: 45px;
                    --ui-control-radius: 12px;
                    --ui-control-font-size: 0.88rem;
                    --ui-field-gap: 13px;

                    --ui-card-radius: 17px;
                    --ui-card-pad: 32px;
                    --ui-filter-header-pad: 18px 33px 25px 33px;
                    --ui-filter-body-pad: 18px 33px 25px 33px;
                    --ui-card-gap: 1.25rem;

                    --ui-highlight-title-size: 1.32rem;
                    --ui-highlight-text-size: 1.06rem;
                    --ui-label-font-size: 0.92rem;
                    --ui-kicker-font-size: 0.72rem;

                    --ui-button-h: 44px;
                    --ui-button-radius: 10px;
                    --ui-button-pad-x: 1.2rem;
                    --ui-button-font-size: 0.92rem;

                    --ui-logo-size-landing: 88px;
                    --ui-logo-size-nonlanding: 52px;
                    --ui-logo-radius-landing: 22px;
                    --ui-logo-radius-nonlanding: 9px;
                    --ui-logo-gap-landing: 15px;

                    --ui-title-size-landing: 3.35rem;
                    --ui-title-size-nonlanding: 2.85rem;
                    --ui-subtitle-size-landing: 1.78rem;
                    --ui-demo-size: 0.78rem;

                    --ui-landing-header-pad-top: 18px;
                    --ui-mission-top: 1.7rem;
                    --ui-mission-bottom: 1.35rem;

                    --ui-landing-lower-section-min-h: clamp(500px, 44vh, 660px);
                    --ui-landing-filter-header-min-h: 96px;
                    --ui-landing-right-card-min-h: 240px;
                }}
            }}


                        /* Ultra-wide / very high resolution screens.
               This keeps the app visibly larger without infinite stretching. */
            @media (min-width: 2700px) and (min-height: 1250px) {{
                :root {{
                    --ui-page-max-w: 2520px;
                    --ui-page-pad-top: 3rem;
                    --ui-page-pad-bottom: 3rem;
                    --ui-page-pad-x: clamp(5rem, 5vw, 10rem);

                    --ui-control-h: 48px;
                    --ui-control-radius: 13px;
                    --ui-control-font-size: 0.92rem;
                    --ui-field-gap: 14px;

                    --ui-card-radius: 18px;
                    --ui-card-pad: 36px;
                    --ui-filter-header-pad: 22px 36px 29px 36px;
                    --ui-filter-body-pad: 22px 36px 29px 36px;
                    --ui-card-gap: 1.35rem;

                    --ui-highlight-title-size: 1.40rem;
                    --ui-highlight-text-size: 1.12rem;
                    --ui-label-font-size: 0.96rem;
                    --ui-kicker-font-size: 0.76rem;

                    --ui-button-h: 47px;
                    --ui-button-radius: 11px;
                    --ui-button-pad-x: 1.3rem;
                    --ui-button-font-size: 0.96rem;

                    --ui-logo-size-landing: 96px;
                    --ui-logo-size-nonlanding: 56px;
                    --ui-logo-radius-landing: 24px;
                    --ui-logo-radius-nonlanding: 10px;
                    --ui-logo-gap-landing: 16px;

                    --ui-title-size-landing: 3.65rem;
                    --ui-title-size-nonlanding: 3.0rem;
                    --ui-subtitle-size-landing: 1.92rem;
                    --ui-demo-size: 0.82rem;

                    --ui-landing-header-pad-top: 20px;
                    --ui-mission-top: 1.9rem;
                    --ui-mission-bottom: 1.5rem;

                    --ui-landing-lower-section-min-h: clamp(570px, 45vh, 740px);
                    --ui-landing-filter-header-min-h: 108px;
                    --ui-landing-right-card-min-h: 270px;
                }}
            }}

            /* Treemap hint vertical calibration by resolution.
               These values affect only the "Click a block..." label.
               They do not touch the Detailed View Mode toggle geometry. */

            /* 1440 x 900 and smaller desktop baseline */
            :root {{
                --ui-treemap-hint-y-shift: 17px;
            }}

            /* Around 1920 x 1080 */
            @media (min-width: 1800px) and (min-height: 950px) {{
                :root {{
                    --ui-treemap-hint-y-shift: 0px;
                }}
            }}

            /* Around 2400 / 2560 wide screens */
            @media (min-width: 2400px) and (min-height: 1200px) {{
                :root {{
                    --ui-treemap-hint-y-shift: 40px;
                }}
            }}

            /* Around 2880 wide screens */
            @media (min-width: 2700px) and (min-height: 1250px) {{
                :root {{
                    --ui-treemap-hint-y-shift: 40px;
                }}
            }}

            html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {{
                background-color: var(--app-bg) !important;
                color: #334155 !important;
            }}

            .block-container {{
                background: transparent !important;
                width: 100% !important;
                padding-top: var(--ui-page-pad-top) !important;
                padding-bottom: var(--ui-page-pad-bottom) !important;
                padding-left: var(--ui-page-pad-x) !important;
                padding-right: var(--ui-page-pad-x) !important;
                max-width: var(--ui-page-max-w) !important;
                margin-left: auto !important;
                margin-right: auto !important;
            }}

            /* LANDING PAGE SHELL
               Centers the landing composition vertically on tall screens.
               Other views remain normal application screens. */
            .st-key-landing_shell {{
                min-height: var(--ui-landing-shell-min-h) !important;
                display: flex !important;
                align-items: center !important;
                width: 100% !important;
            }}

            .st-key-landing_shell > div {{
                width: 100% !important;
            }}

            @media (max-height: 780px) {{
                .st-key-landing_shell {{
                    min-height: auto !important;
                    align-items: flex-start !important;
                }}
            }}


            /* STREAMLIT TOP HEADER
               Keep Streamlit's native top bar visually transparent.
               Do not try to disable/re-enable its internal click layers. */
            [data-testid="stHeader"] {{
                background-color: rgba(0,0,0,0) !important;
                color: #334155 !important;
            }}

            /* OUR HEADER ACTION BUTTONS
               When the page scrolls and these buttons pass under the transparent
               Streamlit top band, keep them above that band and clickable. */
            .st-key-header_action_buttons {{
                position: relative !important;
                z-index: 1000001 !important;
                pointer-events: auto !important;
            }}

            .st-key-header_action_buttons *,
            .st-key-header_action_buttons button,
            .st-key-header_action_buttons [role="button"],
            .st-key-header_action_buttons [data-baseweb="checkbox"] {{
                pointer-events: auto !important;
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
            div[data-baseweb="input"] > div {{
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
            div[data-baseweb="input"]:has(input:disabled) > div {{
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
                border-radius: var(--ui-card-radius) !important;
                padding: var(--ui-filter-header-pad) !important;
                box-shadow: var(--ui-shell-shadow) !important;
                margin-bottom: 0rem !important;
            }}

            .st-key-filter_header .highlight-title {{
                color: #ffffff !important;
            }}

            .st-key-filter_body {{
                background-color: var(--panel-bg) !important;
                border: 1px solid var(--panel-border) !important;
                border-radius: var(--ui-card-radius) !important;
                padding: var(--ui-filter-body-pad) !important;
                box-shadow: var(--ui-shell-shadow) !important;
                margin-top: 0 !important;
                margin-bottom: 4px !important;
                box-sizing: border-box !important;
            }}

            /* LANDING EQUAL-HEIGHT COMPOSITION
               The grey filter block and the two right cards now follow
               the same visual height system instead of depending on text wrapping. */
            .st-key-landing_shell .st-key-filter_header {{
                min-height: var(--ui-landing-filter-header-min-h) !important;
                display: flex !important;
                align-items: center !important;
                box-sizing: border-box !important;
            }}

            .st-key-landing_shell .st-key-filter_body {{
                min-height: calc(
                    var(--ui-landing-lower-section-min-h)
                    - var(--ui-landing-filter-header-min-h)
                ) !important;
            }}

            .right-column-stack {{
                display: flex !important;
                flex-direction: column !important;
                gap: var(--ui-card-gap) !important;
                min-height: var(--ui-landing-lower-section-min-h) !important;
                height: 100% !important;
            }}

            .right-column-stack .highlight-box {{
                margin: 0 !important;
                height: auto !important;
                min-height: var(--ui-landing-right-card-min-h) !important;
                flex: 1 1 0 !important;
                box-sizing: border-box !important;
                display: flex !important;
                flex-direction: column !important;
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
                font-size: var(--ui-label-font-size) !important;
                letter-spacing: -0.01em !important;
                margin-bottom: -3px !important;
            }}

            /* SIDEBAR FILTERS */

            .st-key-sidebar_reset_wrap {{
                margin-top: -12px !important;
                margin-bottom: 0px !important;
                transform: translateY(calc(15px + var(--ui-sidebar-reset-y-shift))) !important;
            }}

            .st-key-sidebar_filters {{
                margin-top: 65px !important;
            }}


            .st-key-sidebar_filters div[data-baseweb="select"] {{
                margin-top: 6px !important;
            }}

            .st-key-sidebar_filters [data-testid="stElementContainer"] {{
                margin-bottom: 10px !important;
            }}

            .st-key-sidebar_filters [data-testid="stElementContainer"]:last-child {{
                margin-bottom: 10px !important;
            }}

            .st-key-sidebar_secret_fields {{
                margin-top: var(--ui-sidebar-secret-fields-top-gap) !important;
            }}

            .st-key-sidebar_secret_fields [data-testid="stElementContainer"] {{
                margin-bottom: 10px !important;
            }}

            .st-key-sidebar_filters [data-testid="stWidgetLabel"] {{
                min-height: 0px !important;
                margin-bottom: 0px !important;
            }}

            .st-key-sidebar_filters label,
            .st-key-sidebar_filters [data-testid="stWidgetLabel"] p {{
                color: #ffffff !important;
                font-weight: 600 !important;
                font-size: var(--ui-label-font-size) !important;
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
                box-shadow: var(--ui-textarea-shadow) !important;
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
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: var(--ui-card-radius);
                padding: var(--ui-card-pad);
                box-shadow: var(--ui-shell-shadow) !important;
                height: 100%;
            }}

            .mission-box {{
                background-color: #e2e8f0 !important;
                border: 1px solid #cbd5e1 !important;
            }}

            .landing-mission-box {{
                margin-top: var(--ui-mission-top) !important;
                margin-bottom: var(--ui-mission-bottom) !important;
            }}

            .highlight-title {{
                font-weight: 800;
                color: #52606d !important;
                font-size: var(--ui-highlight-title-size);
                margin-bottom: 8px;
                letter-spacing: -0.02em;
            }}

            .highlight-text {{
                color: #64748b;
                font-size: var(--ui-highlight-text-size);
                line-height: 1.55;
                font-weight: 450;
            }}

            .highlight-kicker {{
                font-size: var(--ui-kicker-font-size);
                font-weight: 800;
                color: #94a3b8;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                line-height: 1.1;
                white-space: nowrap;
            }}
            .highlight-text b, .highlight-text strong {{ font-weight: 700 !important; color: inherit !important; }}

            /* Tags & Labels */
            label, strong {{
                color: #475569 !important;
                font-weight: 600 !important;
                font-size: var(--ui-label-font-size) !important;
                letter-spacing: -0.01em;
            }}

            /* Detail View */
            .identity-header-text {{ font-size: 1.2rem; font-weight: 600; color: #334155 !important; margin-right: 15px; }}
            .title-box-container {{
                background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px;
                padding: 15px 18px; margin-top: 15px; margin-bottom: 25px; line-height: 1.6;
                font-weight: 500; box-shadow: var(--ui-shell-shadow) !important;
            }}
            .pillar-val-box {{
                background:#ffffff; padding:10px; border:1px solid #cbd5e1; border-radius:6px;
                font-size:0.9rem; color:#334155 !important; min-height:40px; margin-bottom:15px;
                box-shadow: var(--ui-shell-shadow) !important;
            }}

            /* HEADER RIGHT COLUMN TIGHT WRAPPERS */

            .st-key-app_header_landing {{
                display: flex !important;
                align-items: flex-start !important;
                margin: 0 !important;
                padding: var(--ui-landing-header-pad-top) 0 0 0 !important;
            }}

            .st-key-app_header_nonlanding {{
                display: flex !important;
                align-items: center !important;
                margin: 0 !important;
                padding: var(--ui-nonlanding-header-top-pad) 0 0 0 !important;
                min-height: var(--ui-header-nonlanding-h) !important;
                transform: translateY(var(--ui-nonlanding-header-y-shift)) !important;
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
                transform: translateY(var(--ui-nonlanding-header-y-shift)) !important;
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
            .st-key-header_action_buttons label,
            .st-key-header_action_buttons label span,
            .st-key-header_action_buttons label p,
            .st-key-header_action_buttons [data-baseweb="checkbox"] label,
            .st-key-header_action_buttons [data-baseweb="checkbox"] span,
            .st-key-header_action_buttons [data-baseweb="checkbox"] p {{
                font-size: var(--ui-simulation-toggle-label-size) !important;
                font-weight: 700 !important;
                line-height: 1.05 !important;
                white-space: nowrap !important;
                color: #64748b !important;
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

            .st-key-header_action_buttons [data-baseweb="checkbox"] p {{
                transform: translateY(var(--ui-simulation-toggle-label-y-shift)) !important;
            }}


            /* NATIVE RESULTS TABLE — SAFE OUTER WRAPPER ONLY */
            [data-testid="stDataFrame"] {{
                border: 1px solid #e2e8f0 !important;
                border-radius: 14px !important;
                box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
                overflow: hidden !important;
                background: #ffffff !important;
            }}

            [data-testid="stDataFrame"] div {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            }}

            /* Buttons */
            .stButton > button {{
                border-radius: var(--ui-button-radius) !important;
                font-weight: 400 !important;
                font-size: var(--ui-button-font-size) !important;
                line-height: 1 !important;
                padding: 0px var(--ui-button-pad-x) !important;
                transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
                min-height: var(--ui-button-h) !important;
                height: var(--ui-button-h) !important;
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
                border: 1.5px solid var(--ui-accent-blue) !important;
                background-color: var(--ui-accent-blue) !important;
                color: #ffffff !important;
                box-shadow: var(--ui-button-primary-shadow) !important;
            }}

            .stButton > button:hover {{
                background-color: #334155 !important;
                border-color: #1e293b !important;
                box-shadow: var(--ui-button-hover-shadow) !important;
                transform: scale(1.02) translateY(-2px) !important;
            }}

            .stButton > button,
            .stButton > button span,
            .stButton > button p,
            .stButton > button div {{
                font-size: var(--ui-button-font-size) !important;
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
                box-shadow: var(--ui-shell-shadow) !important;
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
            .st-key-trial_meta_inner div[data-baseweb="input"] > div {{
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
                box-shadow: var(--ui-shell-shadow) !important;
                margin: 0 !important;
                padding: 0 !important;
            }}


            [class*="st-key-summary_side_shell_"] > div,
            [class*="st-key-summary_side_shell_"] > div > [data-testid="stVerticalBlock"] {{
                margin: 0 !important;
                padding: 0 !important;
                gap: 0 !important;
            }}


            .st-key-summary_side_shell_completion_prediction_left_top_block {{
                position: relative !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-gauge-help-wrap {{
                position: absolute !important;
                top: 8px !important;
                right: 8px !important;
                z-index: 30 !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-gauge-help-anchor {{
                width: 22px !important;
                height: 22px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                border: 1px solid #1e293b !important;
                border-radius: 10px !important;
                background: #334155 !important;
                color: #ffffff !important;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
                font-size: 0.86rem !important;
                font-weight: 700 !important;
                line-height: 1 !important;
                text-decoration: none !important;
                cursor: help !important;
                user-select: none !important;
                box-shadow: none !important;
                transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-gauge-help-wrap:hover .completion-gauge-help-anchor,
            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-gauge-help-wrap:focus-within .completion-gauge-help-anchor {{
                background: #e2e8f0 !important;
                border-color: #cbd5e1 !important;
                color: #64748b !important;
                transform: scale(1.02) translateY(-2px) !important;
                outline: none !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-gauge-help-tooltip {{
                position: absolute !important;
                top: 26px !important;
                right: 0 !important;
                width: 520px !important;
                max-width: min(520px, calc(100vw - 48px)) !important;
                padding: 13px 15px !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 10px !important;
                background: #ffffff !important;
                color: #334155 !important;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
                font-size: 0.8rem !important;
                font-weight: 500 !important;
                line-height: 1.38 !important;
                letter-spacing: 0 !important;
                text-align: left !important;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.10) !important;
                opacity: 0 !important;
                visibility: hidden !important;
                transform: translateY(2px) !important;
                transition:
                    opacity 0.08s ease-out,
                    transform 0.08s ease-out,
                    visibility 0s linear 0.08s !important;
                pointer-events: none !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-gauge-help-tooltip .tooltip-section {{
                margin: 0 0 10px 0 !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-gauge-help-tooltip .tooltip-section:last-child {{
                margin-bottom: 0 !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-gauge-help-tooltip p {{
                margin: 0 !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-gauge-help-wrap:hover .completion-gauge-help-tooltip,
            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-gauge-help-wrap:focus-within .completion-gauge-help-tooltip {{
                opacity: 1 !important;
                visibility: visible !important;
                transform: translateY(0) !important;
                transition:
                    opacity 0.08s ease-out,
                    transform 0.08s ease-out,
                    visibility 0s linear 0s !important;
            }}


            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-tier-row {{
                text-align: center !important;
                margin-top: -20px !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-tier-inline-wrap {{
                position: relative !important;
                display: inline-block !important;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
                font-size: 1.0rem !important;
                font-weight: 750 !important;
                color: #334155 !important;
                line-height: 1 !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-tier-text {{
                display: inline-block !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-tier-info-wrap {{
                position: absolute !important;
                left: calc(100% + 5px) !important;
                top: 50% !important;
                transform: translateY(-50%) !important;
                z-index: 25 !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-tier-info-anchor {{
                width: 13px !important;
                height: 13px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                border: 1px solid #1e293b !important;
                border-radius: 999px !important;
                background: #334155 !important;
                color: #ffffff !important;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
                font-size: 0.52rem !important;
                font-weight: 800 !important;
                line-height: 1 !important;
                text-decoration: none !important;
                cursor: help !important;
                user-select: none !important;
                box-shadow: none !important;
                transition:
                    background 0.18s ease,
                    border-color 0.18s ease,
                    color 0.18s ease,
                    opacity 0.18s ease !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-tier-info-wrap:hover .completion-tier-info-anchor,
            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-tier-info-wrap:focus-within .completion-tier-info-anchor {{
                background: #eef2f7 !important;
                border-color: #cbd5e1 !important;
                color: #607083 !important;
                outline: none !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-tier-info-tooltip {{
                position: absolute !important;
                top: calc(100% + 8px) !important;
                left: 50% !important;
                width: 180px !important;
                max-width: min(180px, calc(100vw - 48px)) !important;
                padding: 10px 12px !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 10px !important;
                background: #ffffff !important;
                color: #334155 !important;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
                font-size: 0.75rem !important;
                font-weight: 500 !important;
                line-height: 1.34 !important;
                letter-spacing: 0 !important;
                text-align: left !important;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.10) !important;
                opacity: 0 !important;
                visibility: hidden !important;
                transform: translateX(-50%) translateY(2px) !important;
                transition:
                    opacity 0.08s ease-out,
                    transform 0.08s ease-out,
                    visibility 0s linear 0.08s !important;
                pointer-events: none !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-tier-info-tooltip .tooltip-section {{
                margin: 0 !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-tier-info-tooltip p {{
                margin: 0 !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-tier-info-wrap:hover .completion-tier-info-tooltip,
            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-tier-info-wrap:focus-within .completion-tier-info-tooltip {{
                opacity: 1 !important;
                visibility: visible !important;
                transform: translateX(-50%) translateY(0) !important;
                transition:
                    opacity 0.08s ease-out,
                    transform 0.08s ease-out,
                    visibility 0s linear 0s !important;
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

            .st-key-summary_side_inner_completion_prediction_right_block > div {{
                padding: 0 10px 0 10px !important;
            }}
            /* Completion Score — gauge vertical alignment
               Single manual control for the gauge chart + score + tier label.

               Increase --ui-completion-gauge-up-shift = move the whole gauge group UP.
               Decrease --ui-completion-gauge-up-shift = move the whole gauge group DOWN.

               This is intentionally applied to the Plotly chart and tier label directly,
               because spacer-only controls can be overridden by Streamlit layout wrappers.
            */
            :root {{
                --ui-completion-gauge-up-shift: 22px;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .trial-meta-top-gap {{
                height: 0px !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .trial-meta-bottom-gap {{
                height: 0px !important;
            }}

            .st-key-summary_side_inner_completion_prediction_left_top_block [data-testid="stPlotlyChart"],
            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-tier-row {{
                transform: translateY(calc(-1 * var(--ui-completion-gauge-up-shift))) !important;
            }}

            @media (min-width: 1800px) and (min-height: 950px) {{
                :root {{
                    --ui-completion-gauge-up-shift: 15px;
                }}
            }}

            @media (min-width: 2400px) and (min-height: 1200px) {{
                :root {{
                    --ui-completion-gauge-up-shift: 10px;
                }}
            }}

            @media (min-width: 2700px) and (min-height: 1250px) {{
                :root {{
                    --ui-completion-gauge-up-shift: 5px;
                }}
            }}

            /* Completion Score — treemap vertical alignment
               1440: preserve compact alignment.
               1920 / 2560 / 2880: progressively move treemap down. */
            .st-key-summary_side_shell_completion_prediction_right_block .trial-meta-top-gap {{
                height: 6px !important;
            }}

            .st-key-summary_side_shell_completion_prediction_right_block .trial-meta-bottom-gap {{
                height: 0px !important;
            }}

            @media (min-width: 1800px) and (min-height: 950px) {{
                .st-key-summary_side_shell_completion_prediction_right_block .trial-meta-top-gap {{
                    height: 30px !important;
                }}
            }}

            @media (min-width: 2400px) and (min-height: 1200px) {{
                .st-key-summary_side_shell_completion_prediction_right_block .trial-meta-top-gap {{
                    height: 40px !important;
                }}
            }}

            @media (min-width: 2700px) and (min-height: 1250px) {{
                .st-key-summary_side_shell_completion_prediction_right_block .trial-meta-top-gap {{
                    height: 50px !important;
                }}
            }}

            /* TREEMAP TOGGLE — FLOATING ABOVE TREEMAP BOX, INSIDE TAB 3 ONLY */
            .st-key-trial_detail_tabs .stTabs [data-baseweb="tab-panel"] {{
                overflow: visible !important;
            }}

            .st-key-summary_side_shell_completion_prediction_right_block {{
                position: relative !important;
                overflow: visible !important;
            }}

            .st-key-summary_side_inner_completion_prediction_right_block > div {{
                padding: 0px 10px 0 10px !important;
            }}

            .st-key-treemap_zoom_hint {{
                position: absolute !important;
                top: var(--ui-treemap-toggle-top) !important;
                left: var(--ui-treemap-hint-left) !important;
                right: auto !important;
                z-index: 60 !important;
                width: max-content !important;
                max-width: max-content !important;
                min-width: 0 !important;
                margin: 0 !important;
                padding: 0 !important;
                transform: translateY(var(--ui-treemap-hint-y-shift)) !important;
                pointer-events: none !important;
            }}

            .st-key-treemap_zoom_hint p {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
                font-size: 0.72rem !important;
                font-weight: 600 !important;
                line-height: 1 !important;
                color: #64748b !important;
                white-space: nowrap !important;
                margin: 0 !important;
                padding: 0 !important;
            }}

            .st-key-treemap_detailed_drivers_toggle {{
                position: absolute !important;
                top: var(--ui-treemap-toggle-top) !important;
                right: var(--ui-treemap-toggle-right) !important;
                left: auto !important;
                z-index: 60 !important;
                width: max-content !important;
                max-width: max-content !important;
                min-width: 0 !important;
                margin: 0 !important;
                padding: 0 !important;
                pointer-events: none !important;
            }}

            .st-key-treemap_detailed_drivers_toggle [data-testid="stToggle"],
            .st-key-treemap_detailed_drivers_toggle [data-baseweb="checkbox"],
            .st-key-treemap_detailed_drivers_toggle label,
            .st-key-treemap_detailed_drivers_toggle input {{
                pointer-events: auto !important;
            }}

            .st-key-treemap_detailed_drivers_toggle [data-testid="stToggle"] {{
                margin: 0 !important;
                padding: 0 !important;
                width: max-content !important;
                max-width: max-content !important;
            }}

            .st-key-treemap_detailed_drivers_toggle [data-testid="stWidgetLabel"] {{
                min-height: 0 !important;
                margin: 0 !important;
                padding: 0 !important;
            }}

            .st-key-treemap_detailed_drivers_toggle [data-testid="stWidgetLabel"] p,
            .st-key-treemap_detailed_drivers_toggle label p {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
                font-size: 0.72rem !important;
                font-weight: 600 !important;
                line-height: 1 !important;
                color: #64748b !important;
                white-space: nowrap !important;
                margin: 0 !important;
                padding: 0 !important;
            }}

            .st-key-treemap_detailed_drivers_toggle [data-baseweb="checkbox"] {{
                display: inline-flex !important;
                align-items: center !important;
            }}

            .st-key-treemap_detailed_drivers_toggle [data-baseweb="checkbox"] > div {{
                display: inline-flex !important;
                align-items: center !important;
            }}

            .st-key-treemap_detailed_drivers_toggle [data-baseweb="checkbox"] label,
            .st-key-treemap_detailed_drivers_toggle [data-baseweb="checkbox"] span,
            .st-key-treemap_detailed_drivers_toggle [data-baseweb="checkbox"] p {{
                display: inline-flex !important;
                align-items: center !important;
                line-height: 1.1 !important;
                margin: 0 !important;
                padding: 0 !important;
            }}

            .st-key-summary_side_inner_completion_prediction_right_block [data-testid="stPlotlyChart"] {{
                margin: 0 !important;
            }}

            .st-key-summary_side_inner_completion_prediction_right_block [data-testid="stPlotlyChart"] > div {{
                padding: 0 !important;
            }}

            [class*="st-key-summary_side_inner_"] div[data-baseweb="select"] > div,
            [class*="st-key-summary_side_inner_"] div[data-baseweb="input"] > div {{
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
                box-shadow: var(--ui-shell-shadow) !important;
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

            .completion-workflow-note {{
                margin: 10px 0 12px 0;
                padding: 15px 18px;
                background: #fff6cc;
                border: 1px solid #ead98a;
                border-radius: 14px;
                box-shadow: var(--ui-shell-shadow) !important;
            }}

            .completion-workflow-note-label {{
                margin-bottom: 8px;
                color: #8a6b14;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                line-height: 1.1;
            }}

            .completion-workflow-note-text {{
                color: #334155;
                font-size: 0.96rem;
                line-height: 1.55;
                font-weight: 500;
                margin: 0;
            }}

            .completion-workflow-note-list {{
                margin: 10px 0 0 0;
                padding-left: 1.15rem;
                color: #334155;
                font-size: 0.96rem;
                line-height: 1.55;
                font-weight: 500;
            }}

            .completion-workflow-note-list li {{
                margin: 0 0 4px 0;
            }}

            .completion-workflow-note-list li:first-child {{
                list-style: none;
                margin-left: -1.15rem;
                margin-bottom: 6px;
            }}

            .completion-workflow-note strong,
            .completion-workflow-note a,
            .completion-workflow-note a:visited {{
                font-size: inherit !important;
                line-height: inherit !important;
            }}

            .completion-workflow-note strong {{
                color: #1e293b !important;
                font-weight: 800 !important;
            }}

            .completion-workflow-note a,
            .completion-workflow-note a:visited {{
                color: #8b5e00 !important;
                font-weight: 800 !important;
                text-decoration: none;
            }}

            .completion-workflow-note a:hover {{
                color: #5f4600 !important;
                text-decoration: underline;
            }}

            /* DETAIL TABS */
            .st-key-trial_detail_tabs {{
                margin-top: var(--ui-detail-tabs-offset-y) !important;
            }}

            .st-key-trial_detail_tabs .stTabs [data-baseweb="tab-list"] {{
                gap: 8px !important;
                background: transparent !important;
                border: none !important;
                border-bottom: none !important;
                border-radius: 0 !important;
                padding: 3px 0 0 0 !important;
                margin: -3px 0 0px 0 !important;
                box-shadow: none !important;
            }}

            .st-key-trial_detail_tabs .stTabs [data-baseweb="tab-border"] {{
                display: none !important;
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
                box-shadow: var(--ui-selected-tab-shadow) !important;
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


            /* =========================================================
               RESPONSIVE VIEW-SPECIFIC REFINEMENT PASS
               This block deliberately comes late in the CSS so it can
               override the earlier generic responsive profile cleanly.
               ========================================================= */

            @media (min-width: 1800px) and (min-height: 950px) {{
                :root {{
                    --ui-page-max-w: 2160px;
                    --ui-page-pad-top: 2.4rem;
                    --ui-page-pad-bottom: 2.4rem;
                    --ui-page-pad-x: clamp(3rem, 4.6vw, 7.2rem);

                    --ui-results-top-offset: 18px;
                    --ui-detail-top-offset: 18px;
                    --ui-results-count-size: 0.76rem;

                    --ui-sidebar-w: 18.25rem;

                    --ui-landing-lower-section-min-h: clamp(405px, 39vh, 530px);
                    --ui-landing-filter-header-min-h: 84px;
                    --ui-landing-right-card-min-h: 195px;

                    --ui-treemap-toggle-top: -34px;
                    --ui-treemap-toggle-right: 26px;
                }}

                .st-key-results_shell {{
                    margin-top: var(--ui-results-top-offset) !important;
                }}

                .st-key-detail_shell {{
                    margin-top: var(--ui-detail-top-offset) !important;
                }}

                section[data-testid="stSidebar"],
                section[data-testid="stSidebar"] > div:first-child {{
                    width: var(--ui-sidebar-w) !important;
                    min-width: var(--ui-sidebar-w) !important;
                }}

                .st-key-trial_detail_tabs .stTabs [data-baseweb="tab"] {{
                    height: 40px !important;
                    font-size: 0.90rem !important;
                    padding: 0 17px !important;
                }}

                .top-strip-title-label {{
                    font-size: 0.94rem !important;
                }}

                .st-key-trial_title_shell .stTextArea textarea {{
                    font-size: 0.90rem !important;
                    line-height: 1.36 !important;
                }}

                [class*="st-key-summary_side_inner_"] .stTextArea textarea {{
                    font-size: 0.88rem !important;
                    line-height: 1.46 !important;
                }}
            }}


            @media (min-width: 2250px) and (min-height: 1050px) {{
                :root {{
                    --ui-page-max-w: 2420px;
                    --ui-page-pad-top: 2.8rem;
                    --ui-page-pad-bottom: 2.8rem;
                    --ui-page-pad-x: clamp(4rem, 4.6vw, 8.5rem);

                    --ui-results-top-offset: 28px;
                    --ui-detail-top-offset: 30px;
                    --ui-results-count-size: 0.82rem;

                    --ui-sidebar-w: 19.75rem;

                    --ui-control-h: 46px;
                    --ui-control-font-size: 0.91rem;
                    --ui-field-gap: 14px;

                    --ui-button-h: 46px;
                    --ui-button-font-size: 0.94rem;

                    --ui-logo-size-nonlanding: 58px;
                    --ui-title-size-nonlanding: 3.05rem;
                    --ui-demo-size: 0.82rem;

                    --ui-landing-lower-section-min-h: clamp(465px, 40vh, 610px);
                    --ui-landing-filter-header-min-h: 98px;
                    --ui-landing-right-card-min-h: 220px;

                    --ui-highlight-title-size: 1.36rem;
                    --ui-highlight-text-size: 1.10rem;
                    --ui-label-font-size: 0.96rem;
                    --ui-kicker-font-size: 0.76rem;

                    --ui-summary-tab-top-pad: 10px;
                    --ui-detail-tabs-offset-y: 2px;
                    --ui-summary-row-overlap: -6px;
                    --ui-population-bottom-extension: 172px;

                    --ui-treemap-toggle-top: -32px;
                    --ui-treemap-toggle-right: 28px;
                }}

                .st-key-results_shell {{
                    margin-top: var(--ui-results-top-offset) !important;
                }}

                .st-key-detail_shell {{
                    margin-top: var(--ui-detail-top-offset) !important;
                }}

                section[data-testid="stSidebar"],
                section[data-testid="stSidebar"] > div:first-child {{
                    width: var(--ui-sidebar-w) !important;
                    min-width: var(--ui-sidebar-w) !important;
                }}

                .st-key-sidebar_reset_wrap {{
                    padding-left: 0.35rem !important;
                    padding-right: 0.35rem !important;
                }}

                .st-key-sidebar_filters {{
                    margin-top: 78px !important;
                    padding-left: 0.35rem !important;
                    padding-right: 0.35rem !important;
                }}

                .st-key-sidebar_filters [data-testid="stElementContainer"] {{
                    margin-bottom: 13px !important;
                }}

                .st-key-trial_detail_tabs .stTabs [data-baseweb="tab"] {{
                    height: 43px !important;
                    font-size: 0.96rem !important;
                    padding: 0 19px !important;
                    border-radius: 11px !important;
                }}

                .st-key-header_action_buttons [data-testid="stWidgetLabel"] p,
                .st-key-header_action_buttons label,
                .st-key-header_action_buttons label span,
                .st-key-header_action_buttons label p,
                .st-key-header_action_buttons [data-baseweb="checkbox"] label,
                .st-key-header_action_buttons [data-baseweb="checkbox"] span,
                .st-key-header_action_buttons [data-baseweb="checkbox"] p {{
                    font-size: var(--ui-simulation-toggle-label-size) !important;
                }}

                .top-strip-title-label {{
                    font-size: 0.98rem !important;
                    margin: 9px 0 7px 0 !important;
                }}

                .st-key-trial_title_shell {{
                    padding:
                        var(--ui-meta-shell-pad-top)
                        var(--ui-meta-shell-pad-right)
                        14px
                        var(--ui-meta-shell-pad-left) !important;
                }}

                .st-key-trial_title_shell [data-baseweb="textarea"],
                .st-key-trial_title_shell .stTextArea textarea {{
                    min-height: 92px !important;
                }}

                .st-key-trial_title_shell .stTextArea textarea {{
                    font-size: 0.96rem !important;
                    line-height: 1.38 !important;
                }}

                [class*="st-key-summary_side_inner_"] .stTextArea textarea {{
                    font-size: 0.94rem !important;
                    line-height: 1.48 !important;
                }}

                .st-key-summary_side_inner_study_summary_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_study_summary_block .stTextArea textarea {{
                    min-height: 205px !important;
                }}

                .st-key-summary_side_inner_interventions_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_interventions_block .stTextArea textarea,
                .st-key-summary_side_inner_primary_outcomes_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_primary_outcomes_block .stTextArea textarea {{
                    min-height: 195px !important;
                }}

                .st-key-summary_side_inner_ta_conditions_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_ta_conditions_block .stTextArea textarea {{
                    min-height: 340px !important;
                }}

                .st-key-summary_side_inner_eligibility_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_eligibility_block .stTextArea textarea {{
                    min-height: 430px !important;
                }}
            }}


            @media (min-width: 2700px) and (min-height: 1250px) {{
                :root {{
                    --ui-page-max-w: 2580px;
                    --ui-page-pad-top: 3.0rem;
                    --ui-page-pad-bottom: 3.0rem;
                    --ui-page-pad-x: clamp(4.5rem, 4.4vw, 8.8rem);

                    --ui-results-top-offset: 34px;
                    --ui-detail-top-offset: 38px;
                    --ui-results-count-size: 0.86rem;

                    --ui-sidebar-w: 21rem;

                    --ui-control-h: 48px;
                    --ui-control-radius: 13px;
                    --ui-control-font-size: 0.94rem;
                    --ui-field-gap: 15px;

                    --ui-card-radius: 18px;
                    --ui-card-pad: 34px;
                    --ui-filter-header-pad: 22px 34px 27px 34px;
                    --ui-filter-body-pad: 22px 34px 27px 34px;
                    --ui-card-gap: 1.28rem;

                    --ui-highlight-title-size: 1.42rem;
                    --ui-highlight-text-size: 1.14rem;
                    --ui-label-font-size: 0.98rem;
                    --ui-kicker-font-size: 0.78rem;

                    --ui-button-h: 48px;
                    --ui-button-radius: 11px;
                    --ui-button-pad-x: 1.28rem;
                    --ui-button-font-size: 0.98rem;

                    --ui-logo-size-landing: 100px;
                    --ui-logo-size-nonlanding: 62px;
                    --ui-logo-radius-landing: 24px;
                    --ui-logo-radius-nonlanding: 10px;
                    --ui-logo-gap-landing: 17px;

                    --ui-title-size-landing: 3.75rem;
                    --ui-title-size-nonlanding: 3.18rem;
                    --ui-subtitle-size-landing: 1.96rem;
                    --ui-demo-size: 0.84rem;

                    --ui-landing-header-pad-top: 18px;
                    --ui-mission-top: 1.7rem;
                    --ui-mission-bottom: 1.35rem;

                    --ui-landing-lower-section-min-h: clamp(510px, 40vh, 650px);
                    --ui-landing-filter-header-min-h: 104px;
                    --ui-landing-right-card-min-h: 240px;

                    --ui-detail-tabs-offset-y: 4px;
                    --ui-summary-tab-top-pad: 12px;
                    --ui-summary-row-overlap: -4px;
                    --ui-population-bottom-extension: 190px;

                    --ui-treemap-toggle-top: -30px;
                    --ui-treemap-toggle-right: 30px;
                }}

                .st-key-summary_side_inner_study_summary_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_study_summary_block .stTextArea textarea {{
                    min-height: 220px !important;
                }}

                .st-key-summary_side_inner_interventions_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_interventions_block .stTextArea textarea,
                .st-key-summary_side_inner_primary_outcomes_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_primary_outcomes_block .stTextArea textarea {{
                    min-height: 210px !important;
                }}

                .st-key-summary_side_inner_ta_conditions_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_ta_conditions_block .stTextArea textarea {{
                    min-height: 360px !important;
                }}

                .st-key-summary_side_inner_eligibility_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_eligibility_block .stTextArea textarea {{
                    min-height: 455px !important;
                }}
            }}
            /* =========================================================
               FINAL LARGE-SCREEN TYPOGRAPHY + ALIGNMENT PASS
               Purpose:
               - larger text on large screens
               - less dead empty space
               - stronger alignment between grey and white boxes
               - works from large desktop through ultra-wide screens
               ========================================================= */

            @media (min-width: 1700px) and (min-height: 900px) {{
                :root {{
                    --ui-page-max-w: 2240px;
                    --ui-page-pad-x: clamp(3.2rem, 4.2vw, 7rem);
                    --ui-page-pad-top: 2.25rem;
                    --ui-page-pad-bottom: 2.25rem;

                    --ui-card-radius: 16px;
                    --ui-card-pad: 30px;
                    --ui-card-gap: 1rem;

                    --ui-highlight-title-size: 1.34rem;
                    --ui-highlight-text-size: 1.10rem;
                    --ui-label-font-size: 0.98rem;
                    --ui-kicker-font-size: 0.76rem;

                    --ui-control-h: 45px;
                    --ui-control-radius: 12px;
                    --ui-control-font-size: 0.94rem;
                    --ui-field-gap: 13px;

                    --ui-button-h: 46px;
                    --ui-button-font-size: 0.96rem;
                    --ui-button-radius: 11px;

                    --ui-logo-size-landing: 92px;
                    --ui-logo-size-nonlanding: 58px;
                    --ui-title-size-landing: 3.55rem;
                    --ui-title-size-nonlanding: 3.05rem;
                    --ui-subtitle-size-landing: 1.86rem;
                    --ui-demo-size: 0.82rem;

                    --ui-results-count-size: 0.90rem;
                    --ui-results-table-font-size: 0.92rem;

                    --ui-sidebar-w: 19rem;

                    /* Landing: lower height than before, larger text.
                       This reduces the empty grey/white card interiors. */
                    --ui-landing-lower-section-min-h: clamp(390px, 32vh, 505px);
                    --ui-landing-filter-header-min-h: 90px;
                    --ui-landing-card-min-h: calc((var(--ui-landing-lower-section-min-h) - var(--ui-card-gap)) / 2);

                    /* Detail: coordinated heights for box alignment. */
                    --ui-detail-top-strip-h: 132px;
                    --ui-detail-summary-h: 220px;
                    --ui-detail-bottom-h: 215px;
                    --ui-detail-conditions-h: 385px;
                    --ui-detail-eligibility-h: 500px;
                    --ui-detail-side-min-h: 522px;

                    --ui-meta-top-gap: 28px;
                    --ui-meta-row-gap: 28px;
                    --ui-meta-bottom-gap: 28px;
                    --ui-top-strip-control-h: 30px;

                    --ui-summary-tab-top-pad: 12px;
                    --ui-summary-row-overlap: -2px;
                    --ui-population-bottom-extension: 210px;

                    --ui-treemap-toggle-top: -28px;
                    --ui-treemap-toggle-right: 30px;
                }}

                /* Results page positioning and table readability */
                .st-key-results_shell {{
                    margin-top: 28px !important;
                }}

                section[data-testid="stSidebar"],
                section[data-testid="stSidebar"] > div:first-child {{
                    width: var(--ui-sidebar-w) !important;
                    min-width: var(--ui-sidebar-w) !important;
                }}

                [data-testid="stDataFrame"] div {{
                    font-size: var(--ui-results-table-font-size) !important;
                    line-height: 1.2 !important;
                }}

                .st-key-sidebar_filters {{
                    margin-top: 82px !important;
                    padding-left: 0.35rem !important;
                    padding-right: 0.35rem !important;
                }}

                .st-key-sidebar_filters [data-testid="stElementContainer"] {{
                    margin-bottom: 14px !important;
                }}

                /* Landing page: force a real equal-height visual system */
                .st-key-landing_shell [data-testid="column"] > div > [data-testid="stVerticalBlock"] {{
                    gap: var(--ui-card-gap) !important;
                }}

                .st-key-landing_shell .st-key-filter_header {{
                    min-height: var(--ui-landing-filter-header-min-h) !important;
                    display: flex !important;
                    align-items: center !important;
                }}

                .st-key-landing_shell .st-key-filter_body {{
                    min-height: calc(
                        var(--ui-landing-lower-section-min-h)
                        - var(--ui-landing-filter-header-min-h)
                        - var(--ui-card-gap)
                    ) !important;
                }}

                .st-key-landing_shell .right-column-stack {{
                    min-height: var(--ui-landing-lower-section-min-h) !important;
                    height: var(--ui-landing-lower-section-min-h) !important;
                }}

                .st-key-landing_shell .right-column-stack .highlight-box {{
                    min-height: var(--ui-landing-card-min-h) !important;
                    height: var(--ui-landing-card-min-h) !important;
                    flex: 1 1 0 !important;
                }}

                .st-key-landing_shell .highlight-title {{
                    margin-bottom: 10px !important;
                }}

                .st-key-landing_shell .highlight-text {{
                    line-height: 1.58 !important;
                }}

                .st-key-filter_body label,
                .st-key-filter_body [data-testid="stWidgetLabel"] p,
                .st-key-sidebar_filters label,
                .st-key-sidebar_filters [data-testid="stWidgetLabel"] p {{
                    font-size: var(--ui-label-font-size) !important;
                    line-height: 1.15 !important;
                }}

                /* Detail page: move down slightly and make text readable */
                .st-key-detail_shell {{
                    margin-top: 30px !important;
                }}

                .st-key-trial_top_strip [data-testid="column"] > div {{
                    height: 100% !important;
                }}

                .st-key-trial_title_shell,
                .st-key-trial_meta_shell {{
                    min-height: var(--ui-detail-top-strip-h) !important;
                    box-sizing: border-box !important;
                }}

                .top-strip-title-label {{
                    font-size: 1.02rem !important;
                    margin: 10px 0 8px 0 !important;
                }}

                .st-key-trial_title_shell .stTextArea textarea {{
                    font-size: 1.02rem !important;
                    line-height: 1.42 !important;
                    min-height: 96px !important;
                }}

                .ui-field-label--stack,
                [class*="st-key-meta_native_field_"] [data-testid="stWidgetLabel"] p,
                [class*="st-key-meta_native_field_"] label p {{
                    font-size: 0.92rem !important;
                    line-height: 1.15 !important;
                }}

                [class*="st-key-summary_side_inner_"] .stTextArea textarea {{
                    font-size: 0.98rem !important;
                    line-height: 1.52 !important;
                }}

                .st-key-trial_detail_tabs .stTabs [data-baseweb="tab"] {{
                    height: 44px !important;
                    font-size: 0.98rem !important;
                    padding: 0 20px !important;
                    border-radius: 12px !important;
                }}

                /* Detail page: align the main grey/white boxes vertically */
                .st-key-summary_side_shell_ta_conditions_block,
                .st-key-summary_side_shell_design_block {{
                    min-height: var(--ui-detail-side-min-h) !important;
                    box-sizing: border-box !important;
                }}

                .st-key-summary_side_inner_study_summary_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_study_summary_block .stTextArea textarea {{
                    min-height: var(--ui-detail-summary-h) !important;
                    height: var(--ui-detail-summary-h) !important;
                }}

                .st-key-summary_side_inner_interventions_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_interventions_block .stTextArea textarea,
                .st-key-summary_side_inner_primary_outcomes_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_primary_outcomes_block .stTextArea textarea {{
                    min-height: var(--ui-detail-bottom-h) !important;
                    height: var(--ui-detail-bottom-h) !important;
                }}

                .st-key-summary_side_inner_ta_conditions_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_ta_conditions_block .stTextArea textarea {{
                    min-height: var(--ui-detail-conditions-h) !important;
                    height: var(--ui-detail-conditions-h) !important;
                }}

                .st-key-summary_side_inner_eligibility_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_eligibility_block .stTextArea textarea {{
                    min-height: var(--ui-detail-eligibility-h) !important;
                    height: var(--ui-detail-eligibility-h) !important;
                }}

                .st-key-summary_side_shell_population_block {{
                    min-height: var(--ui-detail-eligibility-h) !important;
                    box-sizing: border-box !important;
                }}

                /* Completion tab: stronger labels and better box relationship */
                .st-key-summary_side_shell_completion_prediction_left_top_block .completion-tier-inline-wrap {{
                    font-size: 1.12rem !important;
                }}

                .st-key-treemap_detailed_drivers_toggle [data-testid="stWidgetLabel"] p,
                .st-key-treemap_detailed_drivers_toggle label p {{
                    font-size: 0.84rem !important;
                }}
            }}


            @media (min-width: 2200px) and (min-height: 1050px) {{
                :root {{
                    --ui-page-max-w: 2460px;
                    --ui-page-pad-x: clamp(4rem, 4.1vw, 8rem);

                    --ui-card-pad: 32px;
                    --ui-highlight-title-size: 1.44rem;
                    --ui-highlight-text-size: 1.17rem;
                    --ui-label-font-size: 1.02rem;
                    --ui-control-font-size: 0.98rem;
                    --ui-button-font-size: 1.00rem;

                    --ui-logo-size-landing: 98px;
                    --ui-logo-size-nonlanding: 62px;
                    --ui-title-size-landing: 3.80rem;
                    --ui-title-size-nonlanding: 3.20rem;
                    --ui-subtitle-size-landing: 2.00rem;

                    --ui-results-count-size: 0.96rem;
                    --ui-results-table-font-size: 0.96rem;
                    --ui-sidebar-w: 20.5rem;

                    --ui-landing-lower-section-min-h: clamp(410px, 32vh, 530px);
                    --ui-landing-filter-header-min-h: 96px;

                    --ui-detail-top-strip-h: 144px;
                    --ui-detail-summary-h: 240px;
                    --ui-detail-bottom-h: 230px;
                    --ui-detail-conditions-h: 420px;
                    --ui-detail-eligibility-h: 540px;
                    --ui-detail-side-min-h: 570px;

                    --ui-meta-top-gap: 30px;
                    --ui-meta-row-gap: 30px;
                    --ui-meta-bottom-gap: 30px;
                    --ui-top-strip-control-h: 32px;
                    --ui-population-bottom-extension: 240px;
                }}

                .st-key-detail_shell {{
                    margin-top: 38px !important;
                }}

                .stTextArea textarea,
                .stTextArea textarea:disabled {{
                    font-size: 0.98rem !important;
                    line-height: 1.52 !important;
                }}

                [class*="st-key-summary_side_inner_"] .stTextArea textarea {{
                    font-size: 1.02rem !important;
                    line-height: 1.54 !important;
                }}

                [class*="st-key-meta_native_field_"] [data-testid="stWidgetLabel"] p,
                [class*="st-key-meta_native_field_"] label p {{
                    font-size: 0.96rem !important;
                }}
            }}


            @media (min-width: 2700px) and (min-height: 1250px) {{
                :root {{
                    --ui-page-max-w: 2580px;
                    --ui-page-pad-x: clamp(4.8rem, 4vw, 8.5rem);

                    --ui-card-pad: 34px;
                    --ui-highlight-title-size: 1.50rem;
                    --ui-highlight-text-size: 1.22rem;
                    --ui-label-font-size: 1.06rem;
                    --ui-control-h: 50px;
                    --ui-control-font-size: 1.02rem;
                    --ui-button-h: 50px;
                    --ui-button-font-size: 1.04rem;

                    --ui-logo-size-landing: 104px;
                    --ui-logo-size-nonlanding: 66px;
                    --ui-title-size-landing: 3.95rem;
                    --ui-title-size-nonlanding: 3.32rem;
                    --ui-subtitle-size-landing: 2.08rem;

                    --ui-results-count-size: 1.00rem;
                    --ui-results-table-font-size: 1.00rem;
                    --ui-sidebar-w: 21.5rem;

                    --ui-landing-lower-section-min-h: clamp(430px, 31vh, 545px);
                    --ui-landing-filter-header-min-h: 100px;

                    --ui-detail-top-strip-h: 152px;
                    --ui-detail-summary-h: 255px;
                    --ui-detail-bottom-h: 245px;
                    --ui-detail-conditions-h: 445px;
                    --ui-detail-eligibility-h: 575px;
                    --ui-detail-side-min-h: 605px;

                    --ui-meta-top-gap: 32px;
                    --ui-meta-row-gap: 32px;
                    --ui-meta-bottom-gap: 32px;
                    --ui-top-strip-control-h: 34px;
                    --ui-population-bottom-extension: 270px;
                }}

                .st-key-results_shell {{
                    margin-top: 36px !important;
                }}

                .st-key-detail_shell {{
                    margin-top: 44px !important;
                }}
            }}

            /* =========================================================
               NON-GRID WIDTH CAP + VERTICAL ESTATE PASS
               Goal:
               - results/grid page can stay wide
               - landing/detail/completion pages stop spreading too much
               - more vertical use on high-resolution screens
               - better grey/white card bottom alignment
               ========================================================= */

            @media (min-width: 1700px) and (min-height: 900px) {{

                /* Default large-screen cap for non-grid views.
                   This affects landing/detail/completion and makes text wrap
                   more naturally instead of becoming long horizontal banners. */
                .block-container {{
                    max-width: 1960px !important;
                }}

                /* Exception: results grid page keeps a much wider canvas. */
                .block-container:has(.st-key-results_shell) {{
                    max-width: 2480px !important;
                }}

                /* Landing page: make the composition taller, not wider. */
                :root {{
                    --ui-landing-lower-section-min-h: clamp(470px, 42vh, 620px);
                    --ui-landing-filter-header-min-h: 96px;
                    --ui-landing-card-min-h: calc((var(--ui-landing-lower-section-min-h) - var(--ui-card-gap)) / 2);

                    --ui-highlight-title-size: 1.40rem;
                    --ui-highlight-text-size: 1.16rem;
                    --ui-label-font-size: 1.00rem;
                    --ui-control-font-size: 0.98rem;
                    --ui-button-font-size: 1.00rem;
                }}

                .st-key-landing_shell {{
                    align-items: center !important;
                }}

                .st-key-landing_shell .st-key-filter_header {{
                    min-height: var(--ui-landing-filter-header-min-h) !important;
                    height: var(--ui-landing-filter-header-min-h) !important;
                    box-sizing: border-box !important;
                    display: flex !important;
                    align-items: center !important;
                }}

                .st-key-landing_shell .st-key-filter_body {{
                    min-height: calc(
                        var(--ui-landing-lower-section-min-h)
                        - var(--ui-landing-filter-header-min-h)
                        - var(--ui-card-gap)
                    ) !important;
                    height: calc(
                        var(--ui-landing-lower-section-min-h)
                        - var(--ui-landing-filter-header-min-h)
                        - var(--ui-card-gap)
                    ) !important;
                    box-sizing: border-box !important;
                }}

                .st-key-landing_shell .right-column-stack {{
                    min-height: var(--ui-landing-lower-section-min-h) !important;
                    height: var(--ui-landing-lower-section-min-h) !important;
                }}

                .st-key-landing_shell .right-column-stack .highlight-box {{
                    min-height: var(--ui-landing-card-min-h) !important;
                    height: var(--ui-landing-card-min-h) !important;
                    flex: 1 1 0 !important;
                    box-sizing: border-box !important;
                }}

                /* Give the landing filter body a less top-heavy distribution. */
                .st-key-landing_shell .st-key-filter_body > div > [data-testid="stVerticalBlock"] {{
                    height: 100% !important;
                    display: flex !important;
                    flex-direction: column !important;
                    justify-content: flex-start !important;
                    gap: 0.35rem !important;
                }}

                .st-key-landing_shell .highlight-text {{
                    line-height: 1.62 !important;
                }}

                .st-key-landing_shell .highlight-title {{
                    margin-bottom: 11px !important;
                }}

                /* Detail/completion pages: lower the content and add vertical body. */
                .st-key-detail_shell {{
                    margin-top: 44px !important;
                }}

                .st-key-trial_title_shell,
                .st-key-trial_meta_shell {{
                    min-height: 152px !important;
                    box-sizing: border-box !important;
                }}

                .st-key-trial_title_shell .stTextArea textarea {{
                    min-height: 104px !important;
                    font-size: 1.04rem !important;
                    line-height: 1.44 !important;
                }}

                .top-strip-title-label {{
                    font-size: 1.04rem !important;
                    margin: 10px 0 8px 0 !important;
                }}

                .st-key-trial_detail_tabs .stTabs [data-baseweb="tab"] {{
                    height: 46px !important;
                    font-size: 1.00rem !important;
                    padding: 0 21px !important;
                    border-radius: 12px !important;
                }}

                [class*="st-key-summary_side_inner_"] .stTextArea textarea {{
                    font-size: 1.02rem !important;
                    line-height: 1.55 !important;
                }}

                /* Trial information tab: make the middle section genuinely taller. */
                .st-key-summary_side_inner_study_summary_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_study_summary_block .stTextArea textarea {{
                    min-height: 275px !important;
                    height: 275px !important;
                }}

                .st-key-summary_side_inner_interventions_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_interventions_block .stTextArea textarea,
                .st-key-summary_side_inner_primary_outcomes_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_primary_outcomes_block .stTextArea textarea {{
                    min-height: 255px !important;
                    height: 255px !important;
                }}

                .st-key-summary_side_inner_ta_conditions_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_ta_conditions_block .stTextArea textarea {{
                    min-height: 560px !important;
                    height: 560px !important;
                }}

                .st-key-summary_side_shell_ta_conditions_block,
                .st-key-summary_side_shell_design_block {{
                    min-height: 650px !important;
                    box-sizing: border-box !important;
                }}

                .st-key-summary_side_inner_eligibility_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_eligibility_block .stTextArea textarea {{
                    min-height: 620px !important;
                    height: 620px !important;
                }}

                /* Completion tab: stronger vertical presence and cleaner alignment. */
                .st-key-summary_side_shell_completion_prediction_left_top_block,
                .st-key-summary_side_shell_completion_prediction_left_bottom_block {{
                    box-sizing: border-box !important;
                }}

                .st-key-summary_side_shell_completion_prediction_left_top_block .completion-tier-inline-wrap {{
                    font-size: 1.18rem !important;
                }}

                .st-key-treemap_detailed_drivers_toggle {{
                    top: -40px !important;
                    right: 32px !important;
                }}

                .st-key-treemap_detailed_drivers_toggle [data-testid="stWidgetLabel"] p,
                .st-key-treemap_detailed_drivers_toggle label p {{
                    font-size: 0.88rem !important;
                }}
            }}


            @media (min-width: 2200px) and (min-height: 1050px) {{

                /* Non-grid views stay readable and more vertical.
                   Grid view remains wide. */
                .block-container {{
                    max-width: 2060px !important;
                }}

                .block-container:has(.st-key-results_shell) {{
                    max-width: 2600px !important;
                }}

                :root {{
                    --ui-landing-lower-section-min-h: clamp(500px, 43vh, 680px);
                    --ui-landing-filter-header-min-h: 104px;

                    --ui-highlight-title-size: 1.48rem;
                    --ui-highlight-text-size: 1.23rem;
                    --ui-label-font-size: 1.06rem;
                    --ui-control-font-size: 1.03rem;
                    --ui-button-font-size: 1.05rem;
                }}

                .st-key-detail_shell {{
                    margin-top: 52px !important;
                }}

                .st-key-trial_title_shell,
                .st-key-trial_meta_shell {{
                    min-height: 164px !important;
                }}

                .st-key-trial_title_shell .stTextArea textarea {{
                    min-height: 114px !important;
                    font-size: 1.08rem !important;
                }}

                .st-key-summary_side_inner_study_summary_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_study_summary_block .stTextArea textarea {{
                    min-height: 305px !important;
                    height: 305px !important;
                }}

                .st-key-summary_side_inner_interventions_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_interventions_block .stTextArea textarea,
                .st-key-summary_side_inner_primary_outcomes_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_primary_outcomes_block .stTextArea textarea {{
                    min-height: 285px !important;
                    height: 285px !important;
                }}

                .st-key-summary_side_inner_ta_conditions_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_ta_conditions_block .stTextArea textarea {{
                    min-height: 620px !important;
                    height: 620px !important;
                }}

                .st-key-summary_side_shell_ta_conditions_block,
                .st-key-summary_side_shell_design_block {{
                    min-height: 720px !important;
                }}

                .st-key-summary_side_inner_eligibility_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_eligibility_block .stTextArea textarea {{
                    min-height: 700px !important;
                    height: 700px !important;
                }}
            }}


            @media (min-width: 2700px) and (min-height: 1250px) {{

                /* At 2880-like screens, non-grid views should not become
                   ultra-wide. Let height do more of the work. */
                .block-container {{
                    max-width: 2140px !important;
                }}

                .block-container:has(.st-key-results_shell) {{
                    max-width: 2680px !important;
                }}

                :root {{
                    --ui-page-pad-top: 3.0rem;
                    --ui-page-pad-bottom: 3.0rem;

                    --ui-landing-lower-section-min-h: clamp(540px, 44vh, 740px);
                    --ui-landing-filter-header-min-h: 112px;

                    --ui-highlight-title-size: 1.56rem;
                    --ui-highlight-text-size: 1.30rem;
                    --ui-label-font-size: 1.10rem;
                    --ui-control-h: 52px;
                    --ui-control-font-size: 1.07rem;
                    --ui-button-h: 52px;
                    --ui-button-font-size: 1.08rem;

                    --ui-logo-size-landing: 106px;
                    --ui-logo-size-nonlanding: 68px;
                    --ui-title-size-landing: 4.00rem;
                    --ui-title-size-nonlanding: 3.35rem;
                    --ui-subtitle-size-landing: 2.10rem;
                }}

                .st-key-detail_shell {{
                    margin-top: 60px !important;
                }}

                .st-key-trial_title_shell,
                .st-key-trial_meta_shell {{
                    min-height: 174px !important;
                }}

                .st-key-summary_side_inner_study_summary_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_study_summary_block .stTextArea textarea {{
                    min-height: 330px !important;
                    height: 330px !important;
                }}

                .st-key-summary_side_inner_interventions_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_interventions_block .stTextArea textarea,
                .st-key-summary_side_inner_primary_outcomes_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_primary_outcomes_block .stTextArea textarea {{
                    min-height: 310px !important;
                    height: 310px !important;
                }}

                .st-key-summary_side_inner_ta_conditions_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_ta_conditions_block .stTextArea textarea {{
                    min-height: 680px !important;
                    height: 680px !important;
                }}

                .st-key-summary_side_shell_ta_conditions_block,
                .st-key-summary_side_shell_design_block {{
                    min-height: 790px !important;
                }}

                .st-key-summary_side_inner_eligibility_block [data-baseweb="textarea"],
                .st-key-summary_side_inner_eligibility_block .stTextArea textarea {{
                    min-height: 760px !important;
                    height: 760px !important;
                }}
            }}


            /* =========================================================
               NON-LANDING TOP ALIGNMENT PASS
               Scope:
               - Results / Grid page
               - Trial Details shell
               - all Trial Details tabs through the shared detail shell

               1440 x 900 remains the untouched baseline.
               Negative --ui-nonlanding-header-y-shift moves the shared
               non-landing top cluster upward.
               Positive --ui-sidebar-reset-y-shift moves Reset Filters downward.
               ========================================================= */

            @media (min-width: 1700px) and (min-height: 900px) {{
                :root {{
                    --ui-nonlanding-header-y-shift: -8px;
                    --ui-sidebar-reset-y-shift: 14px;
                }}

                .st-key-results_shell {{
                    margin-top: calc(28px + var(--ui-nonlanding-header-y-shift)) !important;
                }}

                .st-key-detail_shell {{
                    margin-top: calc(44px + var(--ui-nonlanding-header-y-shift)) !important;
                }}
            }}

            @media (min-width: 2200px) and (min-height: 1050px) {{
                :root {{
                    --ui-nonlanding-header-y-shift: -10px;
                    --ui-sidebar-reset-y-shift: 18px;
                }}

                .st-key-results_shell {{
                    margin-top: calc(28px + var(--ui-nonlanding-header-y-shift)) !important;
                }}

                .st-key-detail_shell {{
                    margin-top: calc(52px + var(--ui-nonlanding-header-y-shift)) !important;
                }}
            }}

            @media (min-width: 2700px) and (min-height: 1250px) {{
                :root {{
                    --ui-nonlanding-header-y-shift: -12px;
                    --ui-sidebar-reset-y-shift: 22px;
                }}

                .st-key-results_shell {{
                    margin-top: calc(36px + var(--ui-nonlanding-header-y-shift)) !important;
                }}

                .st-key-detail_shell {{
                    margin-top: calc(60px + var(--ui-nonlanding-header-y-shift)) !important;
                }}
            }}


            /* =========================================================
               FINAL RESPONSIVE HEIGHT CONTRACTS
               Purpose:
               - one shared visual contract for landing, grid, detail,
                 population, and completion views
               - compact at 1440 x 900
               - reference at 1920 x 1080
               - modestly larger at 2560 / 2880 without infinite width
               ========================================================= */

            /* Completion Score — gauge vertical alignment
               Single manual control for the gauge chart + score + tier label.

               Increase --ui-completion-gauge-up-shift = move the whole gauge group UP.
               Decrease --ui-completion-gauge-up-shift = move the whole gauge group DOWN.

               This is intentionally applied to the Plotly chart and tier label directly,
               because spacer-only controls can be overridden by Streamlit layout wrappers.
            */
            :root {{
                --ui-completion-gauge-up-shift: 18px;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .trial-meta-top-gap {{
                height: 0px !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .trial-meta-bottom-gap {{
                height: 0px !important;
            }}

            .st-key-summary_side_inner_completion_prediction_left_top_block [data-testid="stPlotlyChart"],
            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-tier-row {{
                transform: translateY(calc(-1 * var(--ui-completion-gauge-up-shift))) !important;
            }}

            @media (min-width: 1800px) and (min-height: 950px) {{
                :root {{
                    --ui-completion-gauge-up-shift: 10px;
                }}
            }}

            @media (min-width: 2400px) and (min-height: 1200px) {{
                :root {{
                    --ui-completion-gauge-up-shift: 4px;
                }}
            }}

            @media (min-width: 2700px) and (min-height: 1250px) {{
                :root {{
                    --ui-completion-gauge-up-shift: 0px;
                }}
            }}

            :root {{
                --ui-results-grid-h: 645px;

                --ui-landing-form-gap: 0.55rem;

                /* Trial Information tab height contract.
                   Only --ui-detail-side-min-h controls the shared bottom line.
                   The textarea heights below tune the internal white field fill. */
                --ui-detail-summary-h: 165px;
                --ui-detail-bottom-h: 165px;
                --ui-detail-conditions-h: 325px;
                --ui-detail-side-min-h: 412px;

                --ui-population-card-h: var(--ui-detail-side-min-h);
                --ui-population-eligibility-text-h: calc(var(--ui-detail-side-min-h) - 40px);

                --ui-completion-left-card-h: 265px;
                --ui-completion-right-card-h: 535px;
                --ui-completion-tier-font-size: 1.5rem;
                --ui-completion-tier-overlap: -35px;
            }}

            @media (min-width: 1800px) and (min-height: 950px) {{
                :root {{
                    --ui-results-grid-h: 725px;

                    --ui-landing-form-gap: 0.75rem;

                    --ui-detail-summary-h: 220px;
                    --ui-detail-bottom-h: 215px;
                    --ui-detail-conditions-h: 432px;
                    --ui-detail-side-min-h: 539px;

                    --ui-population-card-h: var(--ui-detail-side-min-h);
                    --ui-population-eligibility-text-h: calc(var(--ui-detail-side-min-h) - 50px);

                    --ui-completion-left-card-h: 275px;
                    --ui-completion-right-card-h: 565px;
                    --ui-completion-tier-font-size: 1.6rem;
                    --ui-completion-tier-overlap: -37px;
                }}
            }}

            @media (min-width: 2400px) and (min-height: 1200px) {{
                :root {{
                    --ui-results-grid-h: 810px;

                    --ui-landing-form-gap: 0.9rem;

                    --ui-detail-summary-h: 275px;
                    --ui-detail-bottom-h: 255px;
                    --ui-detail-conditions-h: 528px;
                    --ui-detail-side-min-h: 644px;

                    --ui-population-card-h: var(--ui-detail-side-min-h);
                    --ui-population-eligibility-text-h: calc(var(--ui-detail-side-min-h) - 50px);

                    --ui-completion-left-card-h: 285px;
                    --ui-completion-right-card-h: 585px;
                    --ui-completion-tier-font-size: 1.70rem;
                    --ui-completion-tier-overlap: -35px;
                }}
            }}

            @media (min-width: 2700px) and (min-height: 1250px) {{
                :root {{
                    --ui-results-grid-h: 875px;

                    --ui-landing-form-gap: 1rem;

                    --ui-detail-summary-h: 330px;
                    --ui-detail-bottom-h: 310px;
                    --ui-detail-conditions-h: 640px;
                    --ui-detail-side-min-h: 761px;

                    --ui-population-card-h: var(--ui-detail-side-min-h);
                    --ui-population-eligibility-text-h: calc(var(--ui-detail-side-min-h) - 55px);

                    --ui-completion-left-card-h: 290px;
                    --ui-completion-right-card-h: 595px;
                    --ui-completion-tier-font-size: 1.66rem;
                    --ui-completion-tier-overlap: -40px;
                }}
            }}


            /* =========================================================
               LANDING PAGE FINAL CONTRACT
               Scope: landing page only.
               Goal:
               - align "Clinical Trial Selection" with the first right-card title
               - keep the search controls grouped and centered
               - prevent large screens from stretching the form rhythm too much
               ========================================================= */

            :root {{
                --ui-landing-lower-section-min-h: 355px;
                --ui-landing-filter-header-min-h: 74px;
                --ui-landing-card-min-h: calc((var(--ui-landing-lower-section-min-h) - var(--ui-card-gap)) / 2);
                --ui-landing-filter-body-h: calc(
                    var(--ui-landing-lower-section-min-h)
                    - var(--ui-landing-filter-header-min-h)
                    - var(--ui-card-gap)
                );

                --ui-landing-form-gap: 0.45rem;
                --ui-landing-form-max-w: 100%;
                --ui-landing-label-control-gap: 5px;
            }}

            @media (min-width: 1800px) and (min-height: 950px) {{
                :root {{
                    --ui-landing-lower-section-min-h: 420px;
                    --ui-landing-filter-header-min-h: 82px;
                    --ui-landing-form-gap: 0.50rem;
                    --ui-landing-form-max-w: 96%;
                    --ui-landing-label-control-gap: 5px;
                }}
            }}

            @media (min-width: 2400px) and (min-height: 1200px) {{
                :root {{
                    --ui-landing-lower-section-min-h: 455px;
                    --ui-landing-filter-header-min-h: 90px;
                    --ui-landing-form-gap: 0.54rem;
                    --ui-landing-form-max-w: 94%;
                    --ui-landing-label-control-gap: 6px;
                }}
            }}

            @media (min-width: 2700px) and (min-height: 1250px) {{
                :root {{
                    --ui-landing-lower-section-min-h: 485px;
                    --ui-landing-filter-header-min-h: 96px;
                    --ui-landing-form-gap: 0.58rem;
                    --ui-landing-form-max-w: 92%;
                    --ui-landing-label-control-gap: 6px;
                }}
            }}

            .st-key-landing_shell [data-testid="column"] > div > [data-testid="stVerticalBlock"] {{
                gap: var(--ui-card-gap) !important;
            }}

            /* Title alignment:
               The grey header title now starts at the same vertical inset
               as the right white-card titles: var(--ui-card-pad). */
            .st-key-landing_shell .st-key-filter_header {{
                min-height: var(--ui-landing-filter-header-min-h) !important;
                height: var(--ui-landing-filter-header-min-h) !important;
                box-sizing: border-box !important;
                display: flex !important;
                align-items: flex-start !important;
                padding:
                    var(--ui-card-pad)
                    var(--ui-card-pad)
                    0
                    var(--ui-card-pad) !important;
            }}

            .st-key-landing_shell .st-key-filter_header .highlight-title {{
                margin: 0 !important;
                line-height: 1.15 !important;
            }}

            .st-key-landing_shell .right-column-stack .highlight-title {{
                margin: 0 0 10px 0 !important;
                line-height: 1.15 !important;
            }}

            .st-key-landing_shell .st-key-filter_body {{
                min-height: var(--ui-landing-filter-body-h) !important;
                height: var(--ui-landing-filter-body-h) !important;
                box-sizing: border-box !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }}

            .st-key-landing_shell .st-key-filter_body > div {{
                width: var(--ui-landing-form-max-w) !important;
                max-width: var(--ui-landing-form-max-w) !important;
                height: auto !important;
                margin: auto !important;
            }}

            .st-key-landing_shell .st-key-filter_body > div > [data-testid="stVerticalBlock"] {{
                height: auto !important;
                min-height: 0 !important;
                display: flex !important;
                flex-direction: column !important;
                justify-content: center !important;
                gap: var(--ui-landing-form-gap) !important;
            }}

            .st-key-landing_shell .st-key-filter_body [data-testid="stVerticalBlock"] > div {{
                margin-bottom: 0 !important;
                padding-bottom: 0 !important;
            }}

            .st-key-landing_shell .st-key-filter_body [data-testid="stHorizontalBlock"] {{
                margin: 0 !important;
            }}

            .st-key-landing_shell .st-key-filter_body [data-testid="stWidgetLabel"] {{
                min-height: 0 !important;
                margin-bottom: 0 !important;
                padding-bottom: 0 !important;
            }}

            .st-key-landing_shell .st-key-filter_body label,
            .st-key-landing_shell .st-key-filter_body [data-testid="stWidgetLabel"] p {{
                margin-bottom: 0 !important;
                line-height: 1.12 !important;
            }}

            .st-key-landing_shell .st-key-filter_body div[data-baseweb="select"] {{
                margin-top: var(--ui-landing-label-control-gap) !important;
            }}

            .st-key-landing_shell .right-column-stack {{
                min-height: var(--ui-landing-lower-section-min-h) !important;
                height: var(--ui-landing-lower-section-min-h) !important;
                gap: var(--ui-card-gap) !important;
            }}

            .st-key-landing_shell .right-column-stack .highlight-box {{
                min-height: var(--ui-landing-card-min-h) !important;
                height: var(--ui-landing-card-min-h) !important;
                flex: 1 1 0 !important;
                box-sizing: border-box !important;
            }}

            /* Results grid: shrink to row count, but never exceed the responsive maximum.
               Python sets the actual row-based height through dynamic_height.
               CSS only caps large result sets; it must not force a fixed height. */
            .st-key-results_shell [data-testid="stDataFrame"] {{
                max-height: var(--ui-results-grid-h) !important;
            }}

            .st-key-results_shell [data-testid="stDataFrame"] > div {{
                max-height: var(--ui-results-grid-h) !important;
            }}

            /* Trial Information: shared outer-shell bottom-line contract. */
            .st-key-summary_side_shell_ta_conditions_block,
            .st-key-summary_side_shell_design_block {{
                min-height: var(--ui-detail-side-min-h) !important;
                height: var(--ui-detail-side-min-h) !important;
                box-sizing: border-box !important;
            }}

            .st-key-summary_side_shell_ta_conditions_block > div,
            .st-key-summary_side_shell_design_block > div {{
                min-height: 100% !important;
            }}

            .st-key-summary_side_inner_study_summary_block [data-baseweb="textarea"],
            .st-key-summary_side_inner_study_summary_block .stTextArea textarea {{
                min-height: var(--ui-detail-summary-h) !important;
                height: var(--ui-detail-summary-h) !important;
            }}

            .st-key-summary_side_inner_interventions_block [data-baseweb="textarea"],
            .st-key-summary_side_inner_interventions_block .stTextArea textarea,
            .st-key-summary_side_inner_primary_outcomes_block [data-baseweb="textarea"],
            .st-key-summary_side_inner_primary_outcomes_block .stTextArea textarea {{
                min-height: var(--ui-detail-bottom-h) !important;
                height: var(--ui-detail-bottom-h) !important;
            }}

            .st-key-summary_side_inner_ta_conditions_block [data-baseweb="textarea"],
            .st-key-summary_side_inner_ta_conditions_block .stTextArea textarea {{
                min-height: var(--ui-detail-conditions-h) !important;
                height: var(--ui-detail-conditions-h) !important;
            }}

            /* Population Details: both cards share one shell height. */
            .st-key-summary_side_shell_eligibility_block,
            .st-key-summary_side_shell_population_block {{
                min-height: var(--ui-population-card-h) !important;
                height: var(--ui-population-card-h) !important;
                box-sizing: border-box !important;
            }}

            .st-key-summary_side_inner_eligibility_block [data-baseweb="textarea"],
            .st-key-summary_side_inner_eligibility_block .stTextArea textarea {{
                min-height: var(--ui-population-eligibility-text-h) !important;
                height: var(--ui-population-eligibility-text-h) !important;
            }}

            /* Completion Score: aligned card shells and stronger score/tier text. */
            .st-key-summary_side_shell_completion_prediction_left_top_block,
            .st-key-summary_side_shell_completion_prediction_left_bottom_block {{
                min-height: var(--ui-completion-left-card-h) !important;
                height: var(--ui-completion-left-card-h) !important;
                box-sizing: border-box !important;
            }}

            .st-key-summary_side_shell_completion_prediction_right_block {{
                min-height: var(--ui-completion-right-card-h) !important;
                height: var(--ui-completion-right-card-h) !important;
                box-sizing: border-box !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-tier-row {{
                text-align: center !important;
                margin-top: var(--ui-completion-tier-overlap) !important;
                margin-bottom: 0 !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .completion-tier-inline-wrap {{
                font-size: var(--ui-completion-tier-font-size) !important;
                font-weight: 800 !important;
                letter-spacing: -0.015em !important;
                line-height: 1 !important;
            }}

            /* Completion Score — gauge vertical alignment
               Single manual control for gauge + score + tier label.
               More negative = move UP.
               More positive = move DOWN. */
            :root {{
                --ui-completion-gauge-y-shift: -30px;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .trial-meta-top-gap {{
                height: 0px !important;
            }}

            .st-key-summary_side_shell_completion_prediction_left_top_block .trial-meta-bottom-gap {{
                height: 0px !important;
            }}

            @media (min-width: 1800px) and (min-height: 950px) {{
                :root {{
                    --ui-completion-gauge-y-shift: -10px;
                }}
            }}

            @media (min-width: 2400px) and (min-height: 1200px) {{
                :root {{
                    --ui-completion-gauge-y-shift: -4px;
                }}
            }}

            @media (min-width: 2700px) and (min-height: 1250px) {{
                :root {{
                    --ui-completion-gauge-y-shift: 0px;
                }}
            }}

            .st-key-summary_side_inner_completion_prediction_left_top_block > div {{
                padding: 0 10px 0 10px !important;
                height: 100% !important;
            }}

            .st-key-summary_side_inner_completion_prediction_left_top_block > div > [data-testid="stVerticalBlock"] {{
                min-height: 100% !important;
                height: 100% !important;
                display: flex !important;
                flex-direction: column !important;
                justify-content: center !important;
                transform: translateY(var(--ui-completion-gauge-y-shift)) !important;
            }}

            .st-key-summary_side_inner_completion_prediction_right_block > div {{
                padding: 0 10px 0 10px !important;
                height: 100% !important;
            }}

            .st-key-summary_side_inner_completion_prediction_right_block > div > [data-testid="stVerticalBlock"] {{
                min-height: 100% !important;
                height: 100% !important;
                display: block !important;
            }}

            .st-key-summary_side_inner_completion_prediction_left_top_block [data-testid="stPlotlyChart"],
            .st-key-summary_side_inner_completion_prediction_right_block [data-testid="stPlotlyChart"] {{
                margin: 0 !important;
            }}

            .st-key-summary_side_inner_completion_prediction_left_top_block [data-testid="stPlotlyChart"] > div,
            .st-key-summary_side_inner_completion_prediction_right_block [data-testid="stPlotlyChart"] > div {{
                padding: 0 !important;
            }}

            /* =========================================================
               FINAL NON-LANDING HEADER / SIDEBAR ALIGNMENT TUNING
               1440 x 900 remains unchanged.
               Negative header shift = move logo/title/buttons UP.
               Positive reset shift = move Reset Filters DOWN.
               ========================================================= */

            :root {{
                --ui-nonlanding-header-y-shift: 0px;
                --ui-sidebar-reset-y-shift: 0px;
            }}

            @media (min-width: 1700px) and (min-height: 900px) {{
                :root {{
                    --ui-nonlanding-header-y-shift: -10px;
                    --ui-sidebar-reset-y-shift: 10px;
                }}
            }}

            @media (min-width: 2200px) and (min-height: 1050px) {{
                :root {{
                    --ui-nonlanding-header-y-shift: -12px;
                    --ui-sidebar-reset-y-shift: 20px;
                }}
            }}

            @media (min-width: 2700px) and (min-height: 1250px) {{
                :root {{
                    --ui-nonlanding-header-y-shift: -16px;
                    --ui-sidebar-reset-y-shift: 30px;
                }}
            }}

            /* =========================================================
               SIDEBAR SECRET FIELDS POSITION
               Controls how far below the visible filter dropdowns the
               Register / Analysis / Values / Scores fields are pushed.

               Higher value = secret fields lower.
               They remain accessible by scrolling the sidebar.
               ========================================================= */

            :root {{
                --ui-sidebar-secret-fields-top-gap: 800px;
            }}

            @media (min-width: 1700px) and (min-height: 900px) {{
                :root {{
                    --ui-sidebar-secret-fields-top-gap: 1000px;
                }}
            }}

            @media (min-width: 2200px) and (min-height: 1050px) {{
                :root {{
                    --ui-sidebar-secret-fields-top-gap: 1500px;
                }}
            }}

            @media (min-width: 2700px) and (min-height: 1250px) {{
                :root {{
                    --ui-sidebar-secret-fields-top-gap: 1500px;
                }}
            }}

            @media (min-width: 3200px) and (min-height: 1700px) {{
                :root {{
                    --ui-sidebar-secret-fields-top-gap: 2500px;
                }}
            }}

            /* =========================================================
               HEADER SIMULATION MODE LABEL SIZE
               Responsive control for "Simulation Mode (Editing Content)".

               1440 x 900 must stay compact.
               Larger screens progressively increase the label size.

               Higher label-size = larger text.
               Positive y-shift = label lower.
               ========================================================= */

            :root {{
                --ui-simulation-toggle-label-size: 0.7rem;
                --ui-simulation-toggle-label-y-shift: 1px;
            }}

            @media (min-width: 1700px) and (min-height: 900px) {{
                :root {{
                    --ui-simulation-toggle-label-size: 1.0rem;
                    --ui-simulation-toggle-label-y-shift: 3px;
                }}
            }}

            @media (min-width: 2200px) and (min-height: 1050px) {{
                :root {{
                    --ui-simulation-toggle-label-size: 1.05rem;
                    --ui-simulation-toggle-label-y-shift: 2px;
                }}
            }}

            @media (min-width: 2700px) and (min-height: 1250px) {{
                :root {{
                    --ui-simulation-toggle-label-size: 1.08rem;
                    --ui-simulation-toggle-label-y-shift: 2px;
                }}
            }}


            /* =========================================================
               FINAL OVERRIDE — COMPLETION INFO ICON COLORS
               Reverses normal / hover colors for:
               - top-right gauge question mark
               - small "i" next to the score/tier label
               ========================================================= */

            html body .completion-gauge-help-anchor {{
                background: #334155 !important;
                color: #ffffff !important;
                border-color: #1e293b !important;
            }}

            html body .completion-gauge-help-anchor:hover,
            html body .completion-gauge-help-wrap:hover .completion-gauge-help-anchor,
            html body .completion-gauge-help-wrap:focus-within .completion-gauge-help-anchor {{
                background: #e2e8f0 !important;
                color: #64748b !important;
                border-color: #cbd5e1 !important;
            }}

            html body .completion-tier-info-anchor {{
                background: #334155 !important;
                color: #ffffff !important;
                border-color: #1e293b !important;
            }}

            html body .completion-tier-info-anchor:hover,
            html body .completion-tier-info-wrap:hover .completion-tier-info-anchor,
            html body .completion-tier-info-wrap:focus-within .completion-tier-info-anchor {{
                background: #eef2f7 !important;
                color: #607083 !important;
                border-color: #cbd5e1 !important;
            }}



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
        "s_detail": "True",
        "s_detail_memory": "True",
        "s_scores": "",
        "global_edit_mode": False,
        "show_detailed_drivers": True,
        "show_detailed_drivers_user_set": False,

        "detail_completion_tab_visible": False,
        "detail_prediction_notice": False,
        "completion_score_tab_jump_nonce": 0,

    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)




FILTER_COL_MAP = {
    "f_sponsor": "lead_sponsor_canonical",
    "f_ta": "therapeutic_area_ui",
    "f_phase": "phase_ui",
    "f_year": "start_year",
    "f_nct_id": "nct_id",
}

FILTER_STATE_KEYS = list(FILTER_COL_MAP.keys()) + [
    "s_registry",
    "s_mode",
    "s_detail",
    "s_detail_memory",
    "s_scores",
]


def keep_filter_state_alive():
    """Keep filter values stable when their widgets are not rendered on the detail page."""
    for key in FILTER_STATE_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def reset_filters(return_to_landing=True):
    preserved_registry = str(st.session_state.get("s_registry", "") or "")
    preserved_mode = str(st.session_state.get("s_mode", "") or "")
    preserved_detail = get_s_detail_value()
    preserved_scores = str(st.session_state.get("s_scores", "") or "")

    for key in FILTER_COL_MAP:
        st.session_state[key] = None

    st.session_state["s_registry"] = preserved_registry
    st.session_state["s_mode"] = preserved_mode
    st.session_state["s_scores"] = preserved_scores
    persist_s_detail_value(preserved_detail)

    st.session_state.selected_nct_id = None
    if return_to_landing:
        st.session_state.search_initiated = False


def handle_sidebar_reset_filters():
    reset_filters(return_to_landing=True)


def start_search():
    persist_s_detail_value(
        st.session_state.get("s_detail_memory", st.session_state.get("s_detail", ""))
    )

    st.session_state.search_initiated = True
    st.session_state.selected_nct_id = None


def reset_detail_prediction_state():
    st.session_state.trigger_prediction = False
    st.session_state.analysis_result = None
    st.session_state.analysis_nct_id = None
    st.session_state.detail_completion_tab_visible = False
    st.session_state.detail_prediction_notice = False


def get_s_detail_value():
    detail_value = str(st.session_state.get("s_detail", "") or "").strip()
    memory_value = str(st.session_state.get("s_detail_memory", "") or "").strip()

    if detail_value:
        return detail_value

    if memory_value:
        return memory_value

    if st.session_state.get("show_detailed_drivers_user_set", False):
        return "False"

    return "True"


def persist_s_detail_value(value=None):
    detail_value = get_s_detail_value() if value is None else str(value or "").strip()

    if detail_value.lower() == "true":
        detail_value = "True"
    elif detail_value.lower() in ("false", "0", "no", "off"):
        detail_value = "False"

    st.session_state["s_detail"] = detail_value
    st.session_state["s_detail_memory"] = detail_value




def is_detailed_values_enabled():
    return get_s_detail_value().strip().lower() == "true"


def sync_detail_toggle_from_values():
    persist_s_detail_value()
    st.session_state["show_detailed_drivers"] = is_detailed_values_enabled()


def sync_values_from_detail_toggle():
    st.session_state["show_detailed_drivers_user_set"] = True
    persist_s_detail_value(
        "True"
        if st.session_state.get("show_detailed_drivers", False)
        else "False"
    )


def sync_s_detail_text_input_to_memory():
    st.session_state["show_detailed_drivers_user_set"] = True
    persist_s_detail_value()



def show_completion_score_tab():
    st.session_state.detail_completion_tab_visible = True
    st.session_state.detail_prediction_notice = False

def handle_predict_trial_completion():
    if st.session_state.get("global_edit_mode", False):
        reset_detail_prediction_state()
        st.session_state.detail_prediction_notice = True
        return

    show_completion_score_tab()
    st.session_state.trigger_prediction = True
    st.session_state.completion_score_tab_jump_nonce += 1


def reset_trial_editor_state():
    selected_id = st.session_state.get("selected_nct_id")
    if not selected_id:
        return

    selected_df = X_ALL[X_ALL[ID_COL] == selected_id]
    if selected_df.empty:
        return

    row = selected_df.iloc[0]
    trial_key = selected_id


    for field_id in TRIAL_EDITOR_FIELD_IDS:
        state_key = f"input_{trial_key}_{field_id}"

        initial_val = _get_initial_field_value(field_id, row)

        st.session_state[state_key] = initial_val

    for suffix, candidates in TRIAL_EDITOR_TEXT_FIELDS.items():
        state_key = f"text_{trial_key}_{suffix}"
        value = trial_val(row, *candidates)
        st.session_state[state_key] = "" if value == "N/A" else str(value)

def handle_global_edit_toggle():
    reset_detail_prediction_state()

    if not st.session_state.get("global_edit_mode", False):
        reset_trial_editor_state()

def go_back_to_results():
    st.session_state.selected_nct_id = None
    st.session_state.global_edit_mode = False
    reset_detail_prediction_state()


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

def render_transition_overlay_hook():
    st.iframe(
        """
        <script>
        (function() {
            const doc = parent.document;
            const win = parent.window;
            const OVERLAY_ID = "ctp-transition-overlay";

            function removeOverlay() {
                const existing = doc.getElementById(OVERLAY_ID);
                if (existing) existing.remove();

                if (win.__ctpOverlayTimer) {
                    win.clearTimeout(win.__ctpOverlayTimer);
                    win.__ctpOverlayTimer = null;
                }
            }

            function showOverlay(message, timeoutMs) {
                removeOverlay();

                const overlay = doc.createElement("div");
                overlay.id = OVERLAY_ID;
                overlay.innerHTML = `
                    <style>
                        @keyframes ctpSpin { to { transform: rotate(360deg); } }
                        @keyframes ctpOverlayIn { from { opacity: 0; } to { opacity: 1; } }

                        #${OVERLAY_ID} {
                            position: fixed;
                            inset: 0;
                            z-index: 999999;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            background: rgba(241, 245, 249, 0.72);
                            backdrop-filter: blur(1.5px);
                            -webkit-backdrop-filter: blur(1.5px);
                            animation: ctpOverlayIn 0.08s ease-out both;
                            cursor: progress;
                        }

                        #${OVERLAY_ID} .ctp-card {
                            display: flex;
                            align-items: center;
                            gap: 10px;
                            padding: 10px 14px;
                            border-radius: 12px;
                            border: 1px solid #e2e8f0;
                            background: rgba(255, 255, 255, 0.92);
                            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.10);
                            font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
                            color: #475569;
                            font-size: 0.82rem;
                            font-weight: 700;
                            letter-spacing: -0.01em;
                        }

                        #${OVERLAY_ID} .ctp-spinner {
                            width: 15px;
                            height: 15px;
                            border-radius: 999px;
                            border: 2px solid rgba(148, 163, 184, 0.30);
                            border-top-color: #52606d;
                            animation: ctpSpin 0.75s linear infinite;
                        }
                    </style>

                    <div class="ctp-card">
                        <div class="ctp-spinner"></div>
                        <div>${message}</div>
                    </div>
                `;

                overlay.addEventListener("click", removeOverlay);
                doc.body.appendChild(overlay);

                win.__ctpOverlayTimer = win.setTimeout(removeOverlay, timeoutMs);
            }

            function getButtonText(button) {
                return (button.innerText || button.textContent || "").trim();
            }

            function getOverlayConfig(text) {
                if (text === "Search Trials") return ["Loading trials...", 1600];
                if (text === "Back to Results") return ["Returning to results...", 1600];
                if (text === "Reset Filters") return ["Resetting filters...", 1600];
                if (text === "Predict Trial Completion") return ["Generating completion score...", 3330];
                return null;
            }

            removeOverlay();

            if (!win.__ctpOverlayListenerInstalled) {
                doc.addEventListener("click", function(event) {
                    const button = event.target.closest("button");

                    if (button) {
                        const config = getOverlayConfig(getButtonText(button));
                        if (!config) return;

                        showOverlay(config[0], config[1]);
                        return;
                    }

                    const dataframe = event.target.closest('[data-testid="stDataFrame"]');

                    if (dataframe) {
                        const rect = dataframe.getBoundingClientRect();
                        const clickX = event.clientX - rect.left;

                        // Native st.dataframe row selection is triggered from the left
                        // selection / checkbox band. Avoid showing a false overlay
                        // when the user clicks normal row cells.
                        const CHECKBOX_BAND_WIDTH = 52;

                        if (clickX >= 0 && clickX <= CHECKBOX_BAND_WIDTH) {
                            showOverlay("Opening trial...", 700);
                        }
                    }
                }, true);

                win.__ctpOverlayListenerInstalled = true;
            }
        })();
        </script>
        """,
        height=1,
    )

def render_header(is_landing=True, show_predict_button=False, show_back_button=False, show_global_edit_toggle=False):
    img_base64 = load_logo_base64()

    t1, t2 = st.columns([3.8, 3.2], vertical_alignment="top")
    with t1:
        shell_key = "app_header_landing" if is_landing else "app_header_nonlanding"
        with st.container(key=shell_key):
            if is_landing:
                logo_size = "var(--ui-logo-size-landing)"
                logo_img_size = "calc(var(--ui-logo-size-landing) - 2px)"
                logo_border = "var(--ui-logo-border-landing)"
                logo_radius = "var(--ui-logo-radius-landing)"
                title_size = "var(--ui-title-size-landing)"
                logo_gap = "var(--ui-logo-gap-landing)"
                title_demo_gap = "0px"
            else:
                logo_size = "var(--ui-logo-size-nonlanding)"
                logo_img_size = "calc(var(--ui-logo-size-nonlanding) - 2px)"
                logo_border = "var(--ui-logo-border-nonlanding)"
                logo_radius = "var(--ui-logo-radius-nonlanding)"
                title_size = "var(--ui-title-size-nonlanding)"
                logo_gap = "var(--ui-logo-gap-nonlanding)"
                title_demo_gap = "8px"

            subtitle_html = (
                "<div style='color: #52606d; font-size: var(--ui-subtitle-size-landing); font-weight: 800; display: flex; align-items: baseline; gap: 15px; margin-top: 0px;'>"
                "<span style='line-height: 1;'>Late-Stage Clinical Trial Predictive Engine</span>"
                "<span style='font-size: var(--ui-demo-size); color: #94a3b8; text-transform: uppercase;'>demo version</span>"
                "</div>"
                if is_landing
                else
                "<span style='font-size: var(--ui-demo-size); font-weight: 800; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; line-height: 1; margin-bottom: 0px;'>Demo Version</span>"
            )

            html = (
                f"<div style='display: flex; align-items: center; gap: {logo_gap};'>"
                f"<div style='background-color: white; border: {logo_border} solid #52606d; padding: var(--ui-logo-pad); border-radius: {logo_radius}; display: flex; align-items: center; justify-content: center; height: {logo_size}; width: {logo_size}; flex-shrink: 0; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-top: 0px;'>"
                f"<img src='data:image/png;base64,{img_base64}' style='height: {logo_img_size}; filter: {BRAND_FILTER};'>"
                f"</div>"
                f"<div style='display: {'block' if is_landing else 'flex'}; align-items: {'stretch' if is_landing else 'flex-end'}; gap: {title_demo_gap};'>"
                f"<div style='font-size: {title_size}; font-weight: 800; color: #52606d; line-height: 1; {'margin-top: 0px;' if is_landing else ''}'>CTPredict</div>"
                f"{subtitle_html}"
                f"</div>"
                f"</div>"
            )

            st.markdown(html, unsafe_allow_html=True)

    with t2:

        if show_back_button or show_predict_button or show_global_edit_toggle:

            with st.container(key="header_action_buttons"):
                c_toggle, c_back, c_predict = st.columns([1.55, 0.95, 1.75], gap="small", vertical_alignment="top")

                with c_toggle:
                    if show_global_edit_toggle:
                        st.toggle(
                            "Simulation Mode (Editing Content)",
                            key="global_edit_mode",
                            on_change=handle_global_edit_toggle
                        )

                with c_back:
                    if show_back_button:
                        st.button(
                            "Back to Results",
                            width="stretch",
                            key="header_back_btn",
                            on_click=go_back_to_results
                        )

                with c_predict:
                    if show_predict_button:
                        predict_btn_type = (
                            "secondary"
                            if st.session_state.get("detail_completion_tab_visible", False)
                            else "primary"
                        )

                        st.button(
                            "Predict Trial Completion",
                            width="stretch",
                            type=predict_btn_type,
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
            st.button("Reset", width="stretch", on_click=reset_filters)
        with r3_c3:
            st.button(
                "Search Trials",
                width="stretch",
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
    show_score = str(st.session_state.get("s_scores", "")).strip().lower() == "true"

    grid_cols = [
        "nct_id",
        "ui_search_label",
        "lead_sponsor_canonical",
        "therapeutic_area_ui",
        "phase_ui",
        "start_year",
    ]
    grid_labels = ["NCT ID", "Identity", "Sponsor", "Area", "Phase", "Start Year"]

    if show_score:
        grid_cols.append("Clinical_Score")
        grid_labels.append("Score")

    grid_df = df[grid_cols].copy()
    grid_df.columns = grid_labels

    if show_score and "Score" in grid_df.columns:
        grid_df["Score"] = pd.to_numeric(grid_df["Score"], errors="coerce").map(
            lambda x: "" if pd.isna(x) else f"{x:.1f}".replace(".", ",")
        )

    grid_df = grid_df.sort_values("NCT ID", ascending=True, kind="stable").reset_index(drop=True)

    # Keep row mechanics Python-side, while the final CSS block clamps
    # the visual dataframe height per viewport profile.
    row_h = 34
    header_h = 40
    grid_max_h = 920
    dynamic_height = min(grid_max_h, header_h + (len(grid_df) * row_h) + 2)

    if show_score:
        column_config = {
            "NCT ID": st.column_config.TextColumn("NCT ID", width=92),
            "Identity": st.column_config.TextColumn("Identity", width=476),
            "Sponsor": st.column_config.TextColumn("Sponsor", width=166),
            "Area": st.column_config.TextColumn("Area", width=100),
            "Phase": st.column_config.TextColumn("Phase", width=60),
            "Start Year": st.column_config.NumberColumn("Start Year", width=60, format="%d"),
            "Score": st.column_config.TextColumn("Score", width=40),
        }
    else:
        column_config = {
            "NCT ID": st.column_config.TextColumn("NCT ID", width=92),
            "Identity": st.column_config.TextColumn("Identity", width=490),
            "Sponsor": st.column_config.TextColumn("Sponsor", width=170),
            "Area": st.column_config.TextColumn("Area", width=100),
            "Phase": st.column_config.TextColumn("Phase", width=60),
            "Start Year": st.column_config.NumberColumn("Start Year", width=60, format="%d"),
        }

    table_event = st.dataframe(
        grid_df,
        hide_index=True,
        width="stretch",
        height=dynamic_height,
        column_config=column_config,
        selection_mode="single-row",
        on_select="rerun",
        row_height=row_h,
        key=f"trials_table_{'scores' if show_score else 'base'}",
    )

    selected_rows = table_event.selection.rows

    if selected_rows:
        selected_idx = selected_rows[0]
        if 0 <= selected_idx < len(grid_df):
            return grid_df.iloc[selected_idx]["NCT ID"]

    return None

def open_trial_third_ui(selected_id):
    if not selected_id:
        return

    selected_id = str(selected_id)
    if st.session_state.get("selected_nct_id") == selected_id:
        return

    st.session_state.selected_nct_id = selected_id
    reset_detail_prediction_state()
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


def _get_initial_field_value(field_id, row):
    if field_id in {"has_placebo_ml", "has_dmc_ml"}:
        return _coerce_checkbox_value(
            trial_val(row, field_id.replace("_ml", "_ui"), field_id, default=False)
        )

    display_col = field_id.replace("_ml", "_ui") if "_ml" in field_id else f"{field_id}_ui"
    return trial_val(row, display_col, field_id)


def _init_trial_field_state(field_id, row):
    trial_key = st.session_state.get("selected_nct_id", "no_trial")
    state_key = f"input_{trial_key}_{field_id}"

    initial_val = _get_initial_field_value(field_id, row)

    if state_key not in st.session_state:
        st.session_state[state_key] = initial_val

    meta = TAXONOMY.get(field_id, {})
    options = meta.get("ui", {}).get("options")

    if not options:
        options = _get_dynamic_field_options(field_id)

    return state_key, initial_val, options


def _resolve_field_labels(state_key, initial_val, options):
    labels = [opt[1] for opt in options]
    current_value = st.session_state.get(state_key, initial_val)

    if current_value not in labels and current_value not in (None, "", "N/A"):
        labels = [current_value] + labels

    selected_index = labels.index(current_value) if current_value in labels else 0
    return labels, selected_index

def _readonly_widget_key(state_key, key_suffix=""):
    return f"{state_key}__readonly_{key_suffix}" if key_suffix else f"{state_key}__readonly"


def _safe_set_session_value(key, value):
    try:
        if st.session_state.get(key) != value:
            st.session_state[key] = value
    except st.errors.StreamlitAPIException:
        pass


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
            labels, selected_index = _resolve_field_labels(state_key, initial_val, options)

            if is_edit:
                st.selectbox(
                    label,
                    options=labels,
                    index=selected_index,
                    key=f"{state_key}_{key_suffix}" if key_suffix else state_key,
                    label_visibility="collapsed"
                )
            else:
                readonly_key = _readonly_widget_key(state_key, key_suffix)
                readonly_value = labels[selected_index] if labels else ""

                _safe_set_session_value(readonly_key, readonly_value)

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
            labels, selected_index = _resolve_field_labels(state_key, initial_val, options)

            if is_edit:
                st.selectbox(
                    label,
                    options=labels,
                    index=selected_index,
                    key=f"{state_key}_{key_suffix}" if key_suffix else state_key
                )
            else:
                readonly_key = _readonly_widget_key(state_key, key_suffix)
                readonly_value = labels[selected_index] if labels else ""

                _safe_set_session_value(readonly_key, readonly_value)

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

def render_completion_workflow_note():
    st.markdown(COMPLETION_WORKFLOW_INFO_HTML, unsafe_allow_html=True)

def render_trial_detail_tabs_refined(row):
    render_trial_top_strip_refined(row)

    score_visible = st.session_state.get("detail_completion_tab_visible", False)
    score_requested = score_visible and st.session_state.get("trigger_prediction", False)

    show_completion_workflow_note = (
        st.session_state.get("global_edit_mode", False)
        and st.session_state.get("detail_prediction_notice", False)
    )

    if show_completion_workflow_note:
        render_completion_workflow_note()

    with st.container(key="trial_detail_tabs"):
        if score_visible:
            tab1, tab2, tab3 = st.tabs(
                [DETAIL_TAB_INFO, DETAIL_TAB_POPULATION, DETAIL_TAB_SCORE],
                default=DETAIL_TAB_SCORE,
                key=f"trial_detail_tabs_with_score_{st.session_state.get('completion_score_tab_jump_nonce', 0)}"
            )
        else:
            tab1, tab2 = st.tabs(
                [DETAIL_TAB_INFO, DETAIL_TAB_POPULATION],
                default=DETAIL_TAB_INFO,
                key="trial_detail_tabs_base"
            )

        with tab1:
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

        with tab2:
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

        if score_visible:
            with tab3:
                render_completion_prediction_tab(row)

def get_edited_row(row: pd.Series) -> pd.Series:
    edited_row = row.copy()
    trial_key = st.session_state.get("selected_nct_id", "no_trial")

    # 1. Update from Smart Info Boxes: inputs, selectboxes, toggles
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
                # Map UI label back to ML code when needed
                for opt in options:
                    if opt[1] == val:
                        if field_id.endswith("_ml"):
                            edited_row[field_id] = opt[0]
                        else:
                            edited_row[field_id] = opt[1]
                        break
            else:
                edited_row[field_id] = val

    # 2. Update from large text areas
    for panel_key, candidates in TRIAL_EDITOR_TEXT_FIELDS.items():
        text_key = f"text_{trial_key}_{panel_key}"
        target_col = candidates[0]

        if text_key in st.session_state:
            edited_row[target_col] = st.session_state[text_key]

    return edited_row

def get_analysis_result_for_selected_trial(row):
    if not (st.session_state.trigger_prediction or st.session_state.get("analysis_result")):
        return None

    if (
        not st.session_state.get("analysis_result")
        or st.session_state.get("analysis_nct_id") != st.session_state.selected_nct_id
    ):
        with st.spinner("Analyzing signals..."):
            try:
                row_to_predict: pd.Series = get_edited_row(row)
                prediction_payload = row_to_predict.replace({np.nan: None}).to_dict()

                res = requests.post(
                    API_URL,
                    json=prediction_payload,
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

    if st.session_state.get("trigger_prediction", False):
        st.session_state.trigger_prediction = False

    return st.session_state.get("analysis_result")


def render_completion_prediction_tab(row):
    res = get_analysis_result_for_selected_trial(row)

    # Completion tab visual profile.
    # Keep the gauge visually lighter, give the tier label more room,
    # and lift the treemap slightly inside its shell.
    left_box_h = TEXTAREA_HEIGHTS["completion_prediction_left"]
    right_box_h = TEXTAREA_HEIGHTS["completion_prediction_right"]

    gauge_plot_h = 250
    bar_plot_h = 238
    treemap_plot_h = 530

    left_col, right_col = st.columns([3.25, 3.75], gap="xsmall")

    with left_col:
        with st.container(key="completion_prediction_top_row"):

            def _render_gauge_panel():
                if not res:
                    render_box_spacer(left_box_h)
                    return

                score = res.get("score", 0)
                tier = get_risk_tier(score)

                st.markdown(
                    (
                        '<div class="completion-gauge-help-wrap">'
                        '<div class="completion-gauge-help-anchor" '
                        'aria-label="Completion score help" tabindex="0">?</div>'
                        '<div class="completion-gauge-help-tooltip">'
                        f'{COMPLETION_GAUGE_HELP_TOOLTIP}'
                        '</div>'
                        '</div>'
                    ),
                    unsafe_allow_html=True
                )

                st.plotly_chart(
                    plot_success_gauge(score, height=gauge_plot_h),
                    width="stretch",
                    config={"displayModeBar": False}
                )

                st.markdown(
                    (
                        '<div class="completion-tier-row">'
                            '<div class="completion-tier-inline-wrap">'
                                f'<span class="completion-tier-text">{tier}</span>'
                                '<span class="completion-tier-info-wrap">'
                                    '<span class="completion-tier-info-anchor" '
                                    'aria-label="Completion score scale" tabindex="0">i</span>'
                                    '<span class="completion-tier-info-tooltip">'
                                        f'{COMPLETION_TIER_SCALE_TOOLTIP}'
                                    '</span>'
                                '</span>'
                            '</div>'
                        '</div>'
                    ),
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
                width="stretch",
                config={
                    "displayModeBar": False,
                    "staticPlot": True
                }

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

            sync_detail_toggle_from_values()

            with st.container(key="treemap_zoom_hint"):
                st.markdown(
                    "Click to zoom in, click a title to zoom out",
                    unsafe_allow_html=False
                )

            with st.container(key="treemap_detailed_drivers_toggle"):
                st.toggle(
                    "Detailed View Mode",
                    key="show_detailed_drivers",
                    on_change=sync_values_from_detail_toggle
                )

            show_detailed = is_detailed_values_enabled()

            st.plotly_chart(
                plot_treemap(
                    res["subcat_impacts"],
                    res["pillar_impacts"],
                    show_values=show_detailed,
                    height=treemap_plot_h
                ),
                width="stretch",
                config={"displayModeBar": False}
            )

        render_summary_plot_shell_panel(
            panel_suffix="completion_prediction_right_block",
            body_renderer=_render_treemap_panel
        )

# ==========================
# 5. PAGE RENDERERS
# ==========================

def render_empty_results_message():
    st.markdown(
        """
        <div class="highlight-box" style="margin-top: 0.4rem;">
            <div class="highlight-title" style="margin-bottom: 6px;">No trials match these filters</div>
            <div class="highlight-text">
                Adjust the sidebar filters or use <b>Reset Filters</b> to return to the full result set.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_landing_page(x_base):
    with st.container(key="landing_shell"):
        render_header(is_landing=True)

        st.markdown(
            '''
            <div class="highlight-box mission-box landing-mission-box">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div class="highlight-title">Operational Success & Risk Stratification</div>
                    <div class="highlight-kicker">Core Mission</div>
                </div>
                <div class="highlight-text">This predictive engine estimates the <b>likelihood of operational completion</b> and the <b>risk of early termination</b> using only data available at clinical trial initiation. Each trial is systematically evaluated and classified into <b>four distinct tiers</b> - High Risk, Watchlist, Favorable, and Low Risk - providing a clear and actionable risk profile.</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

        cl, cr = st.columns([1, 1], gap="small")

        with cl:
            with st.container(key="filter_header"):
                st.markdown(
                    '<div class="highlight-title" style="margin:0;">Clinical Trial Selection</div>',
                    unsafe_allow_html=True
                )

            with st.container(key="filter_body"):
                render_filters(x_base)

        with cr:
            st.markdown(
                '''
                <div class="right-column-stack">
                    <div class="highlight-box">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div class="highlight-title">Industry-Scale Public Clinical Data</div>
                            <div class="highlight-kicker">Intelligence Source</div>
                        </div>
                        <div class="highlight-text">Built on the publicly available <b>AACT registry</b>, this machine learning system leverages execution patterns from <b>30,000+ Phase II and III trials</b> since 2005. The analytical scope focuses on <b>late-stage studies</b>, where strategic and financial stakes are highest.</div>
                    </div>
                    <div class="highlight-box">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div class="highlight-title">Predictive Power & Benchmarking</div>
                            <div class="highlight-kicker">Engine Accuracy</div>
                        </div>
                        <div class="highlight-text">When comparing a completed trial with one that terminated early, the system assigns a <b>higher risk score</b> to the failed trial in <b>75% of cases</b>. It outperforms the 50% random baseline and traditional approaches built on publicly available data (<b>ROC AUC ≈ 0.75</b> vs. 0.50 baseline).</div>
                    </div>
                </div>
                ''',
                unsafe_allow_html=True
            )

def render_results_page(x_base):
    with st.sidebar:
        with st.container(key="sidebar_reset_wrap"):
            st.button(
                "Reset Filters",
                width="stretch",
                on_click=handle_sidebar_reset_filters
            )

        with st.container(key="sidebar_filters"):
            filtered_df = render_filters(x_base, is_sidebar=True)

        with st.container(key="sidebar_secret_fields"):
            st.text_input("Register", key="s_registry")
            st.text_input("Analysis", key="s_mode")
            st.text_input(
                "Values",
                key="s_detail",
                on_change=sync_s_detail_text_input_to_memory
            )
            st.text_input("Scores", key="s_scores")

    with st.container(key="results_shell"):
        render_header(is_landing=False)

        st.markdown(
            f"<div style='text-align:left; margin:var(--ui-nonlanding-body-gap) 0 8px 0; color:#94a3b8; font-weight:600; font-size:var(--ui-results-count-size); line-height:1;'>{len(filtered_df):,} trials matching criteria</div>",
            unsafe_allow_html=True
        )

        if filtered_df.empty:
            render_empty_results_message()
            return

        selected_id = render_trials_grid(filtered_df)

        if selected_id:
            open_trial_third_ui(selected_id)


def render_detail_page():
    selected_df = X_ALL[X_ALL[ID_COL] == st.session_state.selected_nct_id]

    with st.container(key="detail_shell"):
        render_header(
            is_landing=False,
            show_predict_button=True,
            show_back_button=True,
            show_global_edit_toggle=True
        )

        if selected_df.empty:
            st.warning("Selected trial not found.")
            return

        row = selected_df.iloc[0]
        render_trial_detail_tabs_refined(row)


def route_app():
    x_base = X_ALL

    if st.session_state.selected_nct_id:
        render_detail_page()
        return

    if st.session_state.search_initiated:
        render_results_page(x_base)
        return

    render_landing_page(x_base)


# ==========================
# 6. MAIN UI FLOW
# ==========================

init_session_state()
keep_filter_state_alive()
inject_custom_styles()
render_transition_overlay_hook()
route_app()
