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
    page_title="ClinTrialPredict | Forensic RAW Evidence",
    layout="wide",
)

# ==========================
# 1. SETUP & PATHS
# ==========================
CURRENT_DIR = Path(__file__).resolve().parent
# Pointing to the synced search registry for efficiency
DATA_PATH = CURRENT_DIR / "data" / "search_registry.csv"
TAXONOMY_PATH = CURRENT_DIR.parent / "models" / "taxonomy_01.json"
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
ID_COL = "nct_id"

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"Data file not found at {DATA_PATH}")
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH, dtype={ID_COL: str})
    return df

@st.cache_data
def load_taxonomy():
    if not TAXONOMY_PATH.exists():
        st.error(f"Taxonomy file not found at {TAXONOMY_PATH}")
        return {}
    with open(TAXONOMY_PATH, "r") as f:
        return json.load(f).get("FIELDS", {})

X_ALL = load_data()
TAXONOMY = load_taxonomy()

def get_risk_tier(score):
    if score >= 70: return "HIGH", "High probability of clinical success.", "#dcfce7", "#166534"
    if score >= 50: return "MODERATE", "Success is possible but carries risks.", "#fef9c3", "#854d0e"
    return "LOW", "Significant forensic indicators of trial failure.", "#fee2e2", "#991b1b"

# ==========================
# GLOBAL STYLES
# ==========================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        .block-container { padding-top: 0rem !important; padding-bottom: 0rem !important; }
        [data-testid="stSidebarUserContent"] { padding-top: 0rem !important; }
        [data-testid="stDecoration"] { display: none !important; }
        [data-testid="stHeader"] { background-color: rgba(0,0,0,0) !important; }
        
        html, body, [class*="css"], .stMarkdown {
            font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }
        
        .main > div { max-width: 1400px; margin: 0 auto; }
        
        section[data-testid="stSidebar"] {
            background-color: #334155 !important;
            border-right: 1px solid #d1d5db;
        }
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] label {
            color: #f8fafc !important;
            font-weight: 500;
        }
        
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] input {
            background-color: white !important;
            color: #1e293b !important;
            border: 1px solid #cbd5e1 !important;
        }

        div[data-testid="stExpander"] {
            border: none !important;
            border-radius: 8px !important;
            margin-bottom: 15px;
            overflow: hidden;
            box-shadow: none !important;
        }
        
        div[data-testid="stExpander"] summary {
            background-color: #cbd5e1 !important;
            padding: 10px 15px !important;
            border-radius: 8px 8px 0 0 !important;
            border: none !important;
            transition: background-color 0.2s;
        }
        div[data-testid="stExpander"] summary:hover {
            background-color: #94a3b8 !important;
        }
        div[data-testid="stExpander"] summary span p {
            color: #0f172a !important;
            font-weight: 700 !important;
        }

        div[data-testid="stExpander"] > div[role="region"] {
            border: none !important;
            padding: 20px !important;
            background-color: white !important;
        }

        .title-box-container {
            background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px;
            padding: 15px 18px; min-height: 100px; max-height: 250px; overflow-y: auto; 
            font-size: 0.95rem; color: #1e293b; margin-top: 10px; margin-bottom: 20px;
            line-height: 1.5; font-weight: 500; white-space: pre-wrap;
        }

        .raw-text-block {
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            line-height: 1.4;
            color: #334155;
            background: #ffffff;
            border-left: 3px solid #64748b;
            padding: 8px 12px;
            margin-bottom: 12px;
        }

        .risk-box {
            padding: 20px; border-radius: 10px; margin-top: 20px;
            border: 1px solid transparent;
        }
        .risk-title { font-size: 1.2rem; font-weight: 800; margin-bottom: 5px; }
        .risk-desc { font-size: 0.95rem; font-weight: 500; }
    </style>
    """, unsafe_allow_html=True
)

# ==========================
# 2. SIDEBAR FILTERS
# ==========================
with st.sidebar:
    st.title("Forensic RAW View")
    st.markdown("---")
    
    search_query = st.text_input("Search NCT ID or Title", placeholder="NCT0...")
    
    tas = sorted(X_ALL["therapeutic_area_ui"].dropna().unique())
    selected_ta = st.selectbox("Therapeutic Area", ["All Areas"] + tas)
    
    phases = sorted(X_ALL["phase_ui"].dropna().unique())
    selected_phase = st.multiselect("Phase", phases, default=phases)
    
    if st.button("Reset Filters"):
        st.rerun()

# Filtering logic
filtered_df = X_ALL.copy()
if search_query:
    filtered_df = filtered_df[
        filtered_df[ID_COL].str.contains(search_query, case=False, na=False) |
        filtered_df["official_title"].str.contains(search_query, case=False, na=False)
    ]
if selected_ta != "All Areas":
    filtered_df = filtered_df[filtered_df["therapeutic_area_ui"] == selected_ta]
filtered_df = filtered_df[filtered_df["phase_ui"].isin(selected_phase)]

# ==========================
# 3. MAIN CONTENT
# ==========================
if "selected_nct_id" not in st.session_state:
    st.session_state.selected_nct_id = None

if st.session_state.selected_nct_id is None:
    st.title("Clinical Universe | Raw Evidence Audit")
    st.subheader(f"Analyzing {len(filtered_df):,} Industry-Led Trials")
    
    display_cols = [ID_COL, "ui_title", "therapeutic_area_ui", "phase_ui", "overall_status"]
    display_df = filtered_df[display_cols].copy()
    display_df.columns = ["NCT ID", "Official Title", "Therapeutic Area", "Phase", "Status"]
    
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=600
    )

    if event and event.selection and event.selection.rows:
        selected_row_idx = event.selection.rows[0]
        st.session_state.selected_nct_id = display_df.iloc[selected_row_idx]["NCT ID"]
        st.rerun()

else:
    # --- DETAIL VIEW ---
    row = X_ALL[X_ALL[ID_COL] == st.session_state.selected_nct_id].iloc[0]
    
    if st.button("← Back to Registry"):
        st.session_state.selected_nct_id = None
        st.session_state.analysis_result = None
        st.rerun()

    st.title(f"Forensic Audit: {row[ID_COL]}")
    
    # 1. IDENTITY BOX
    with st.expander("Study Identity & Raw AACT Narrative", expanded=True):
        st.markdown(f"### {row['official_title']}")
        
        st.markdown("**Brief Summary (Sanitized Raw)**")
        st.markdown(f'<div class="title-box-container">{row.get("ui_summary", "No summary available.")}</div>', unsafe_allow_html=True)
        
        st.markdown("**Eligibility Criteria (Sanitized Raw)**")
        st.markdown(f'<div class="title-box-container">{row.get("ui_criteria", "No criteria available.")}</div>', unsafe_allow_html=True)

    # 2. PILLAR GRID (3x2)
    def render_pillar_raw(pillar_ui_name, taxonomy_pillar_name, row_data):
        features = [
            (feat_id, feat_meta) for feat_id, feat_meta in TAXONOMY.items() 
            if feat_meta.get("pillar") == taxonomy_pillar_name
        ]
        features.sort(key=lambda x: (x[1].get("subgroup", ""), x[1].get("priority", 99)))

        with st.expander(pillar_ui_name, expanded=True):
            for feat_id, meta in features:
                label = meta.get("label", feat_id)
                # Check for UI column first, then raw
                ui_col = feat_id.replace('_ml', '') + '_ui'
                val = row_data.get(ui_col, row_data.get(feat_id))
                
                if pd.isna(val) or val == "UNKNOWN": val = "Not Specified"
                
                # Format multiline strings if they contain the separator
                display_val = str(val).replace(" || ", "\n• ")
                if "\n• " in display_val: display_val = "• " + display_val

                st.markdown(f"**{label}**")
                st.markdown(f'<div class="raw-text-block">{display_val}</div>', unsafe_allow_html=True)

    col_r1_1, col_r1_2 = st.columns(2)
    with col_r1_1: render_pillar_raw("Therapeutic Context", "Therapeutic Context", row)
    with col_r1_2: render_pillar_raw("Execution Framework", "Execution Framework", row)

    col_r2_1, col_r2_2 = st.columns(2)
    with col_r2_1: render_pillar_raw("Scientific Attempts", "Scientific Attempt", row)
    with col_r2_2: render_pillar_raw("Patient Profile", "Patient Profile", row)

    col_r3_1, col_r3_2 = st.columns(2)
    with col_r3_1: render_pillar_raw("AACT Raw Evidence", "AACT Raw Data", row)
    with col_r3_2: render_pillar_raw("System Metadata", "Metadata", row)

    # SCORING SECTION (Optional Prediction)
    if st.button("Run Forensic Probability Analysis"):
        row_dict = row.replace({np.nan: None}).to_dict()
        with st.spinner("Analyzing signals..."):
            try:
                response = requests.post(API_URL, json=row_dict)
                if response.status_code == 200:
                    st.session_state.analysis_result = response.json()
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"System Error: {e}")

    if st.session_state.get("analysis_result"):
        res = st.session_state.analysis_result
        score = res.get('score', 0)
        tier, desc, bg_color, t_color = get_risk_tier(score)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.plotly_chart(plot_success_gauge(score), use_container_width=True)
            st.markdown(f'<div class="risk-box" style="background:{bg_color}; color:{t_color}"><div class="risk-title">{tier}</div>{desc}</div>', unsafe_allow_html=True)
        with c2:
            st.plotly_chart(plot_treemap(res.get('subcat_impacts', []), res.get('pillar_impacts', [])), use_container_width=True)
