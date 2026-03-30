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
# 2. STYLES (Consolidated)
# ==========================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        /* LOCK TO LIGHT MODE & GLOBAL FONT CONSISTENCY */
        :root { color-scheme: light !important; }
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background-color: #f8fafc !important; 
            color: #334155 !important;
        }

        .block-container { 
            max-width: 1400px !important; 
            padding-top: 1rem !important; 
            padding-bottom: 2rem !important; 
            margin: auto !important;
        }
        [data-testid="stSidebarUserContent"] { padding-top: 0rem !important; }
        [data-testid="stDecoration"] { display: none !important; }
        [data-testid="stHeader"] { background-color: rgba(0,0,0,0) !important; }
        html, body, [class*="css"], .stMarkdown { font-family: 'Inter', sans-serif; }
        
        /* Main Container Match */
        .main > div { max-width: 1400px; margin: 0 auto; }
        
        /* Sidebar Styling - The "Anchor" Grey (#52606d) */
        section[data-testid="stSidebar"] { 
            background-color: #52606d !important; 
            border-right: 1px solid #e2e8f0; 
        }
        
        /* High Contrast Inputs - Precision Border */
        div[data-baseweb="select"] > div, input { 
            background-color: white !important;
            border: 1.5px solid #94a3b8 !important; 
            border-radius: 8px !important;
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.02) !important;
            transition: all 0.2s;
        }
        div[data-baseweb="select"]:focus-within > div {
            border-color: #52606d !important;
            box-shadow: 0 0 0 1px #52606d !important;
        }

        /* Hub Container - Shaded and Framed to stand out */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.search-anchor) {
            background-color: #f1f5f9 !important;
            border: 2px solid #cbd5e1 !important;
            border-radius: 16px !important;
            padding: 30px !important; 
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
            border-left: 5px solid #cbd5e1 !important;
            height: 100% !important;
        }
        
        /* SYMMETRY ENGINE: Uniform gaps for all blocks */
        [data-testid="stHorizontalBlock"] {
            gap: 1.5rem !important; 
        }

        /* Ensure the vertical stack of Box 2 and Box 3 has a gap */
        .right-column-stack {
            display: flex;
            flex-direction: column;
            gap: 1.5rem; /* Matches the horizontal gap exactly */
            height: 100%;
        }

        /* Adjust Box 1 bottom margin to match the gap */
        .top-box-margin {
            margin-bottom: 1.5rem !important;
        }

        /* Context Area Styling - Sophisticated Slate Shades */
        .context-area {
            padding: 0px;
            background-color: transparent;
            height: 100%;
            font-family: 'Inter', sans-serif !important;
        }
        .context-title { 
            font-size: 2.2rem; 
            font-weight: 800; 
            color: #1e293b; 
            margin-bottom: 25px; 
            letter-spacing: -0.04em;
            line-height: 1.1;
            font-family: 'Inter', sans-serif !important;
        }
        .context-text { 
            font-size: 1.1rem; 
            line-height: 1.6; 
            color: #475569; 
            margin-bottom: 35px;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Tweak Highlight Box for consistent padding */
        .highlight-box {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 24px; /* Slightly more breathable padding */
            border-left: 4px solid #cbd5e1;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .highlight-box:hover {
            border-left-color: #52606d;
            background-color: #fcfdfe;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05);
            transform: translateX(5px);
        }
        .highlight-title {
            font-weight: 800;
            color: #334155;
            font-size: 1.15rem;
            margin-bottom: 8px;
            letter-spacing: -0.02em;
            font-family: 'Inter', sans-serif !important;
        }
        .highlight-text {
            color: #64748b;
            font-size: 0.95rem;
            line-height: 1.55;
            font-weight: 450;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Multiselect Tag - High Contrast Grey */
        span[data-baseweb="tag"] {
            background-color: #f1f5f9 !important;
            color: #1e293b !important;
            border-radius: 4px !important;
            font-weight: 600 !important;
        }

        /* Executive Overrides */
        label, strong { color: #475569 !important; font-weight: 600 !important; font-size: 0.85rem !important; letter-spacing: -0.01em; }
        
        /* Buttons - Lean & Matching Sidebar Grey */
        .stButton > button { 
            border-radius: 8px; 
            font-weight: 700; 
            padding: 0.6rem 1rem;
            transition: all 0.2s; 
            border: 1px solid #cbd5e1;
            background-color: white;
            color: #52606d;
        }
        .stButton > button:hover {
            border-color: #52606d;
            background-color: #f8fafc;
        }
        
        /* Primary Action Button (Search) */
        div.stButton > button[kind="primary"] { 
            background-color: #52606d !important; 
            color: white !important; 
            border: none !important;
            height: 42px; 
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #404b56 !important;
        }

        /* Identity Content */
        .identity-header-text { font-size: 1.2rem; font-weight: 600; color: #0f172a; margin-right: 15px; }
        .title-box-container { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 15px 18px; margin-top: 15px; margin-bottom: 25px; line-height: 1.6; font-weight: 500; }

        /* Mobile Adjustments */
        @media (max-width: 768px) {
            .stButton > button { height: 50px; font-size: 1rem; }
            h1 { font-size: 1.8rem !important; }
        }
    </style>
""", unsafe_allow_html=True)

# CSS to hide sidebar when search not initiated
if not st.session_state.get("search_initiated", False):
    st.markdown("""
        <style>
            section[data-testid="stSidebar"] { display: none !important; }
        </style>
    """, unsafe_allow_html=True)

# ==========================
# 3. DATA LOADING
# ==========================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else pd.DataFrame()
    if 'start_year' in df.columns: df['start_year'] = pd.to_numeric(df['start_year'], errors='coerce').fillna(0).astype(int)
    tax = json.load(open(TAXONOMY_PATH)) if TAXONOMY_PATH.exists() else {}
    return df, tax.get("FIELDS", tax)

X_ALL, TAXONOMY = load_data()

# ==========================
# 4. STATE & HELPERS
# ==========================
for key, val in {"search_initiated": False, "selected_nct_id": None, "trigger_prediction": False}.items():
    if key not in st.session_state: st.session_state[key] = val

def reset_filters():
    for key in ["f_sponsor", "f_ta", "f_indication", "f_phase", "f_year", "f_nct_id", "s_registry", "s_mode"]:
        if key in st.session_state: 
            st.session_state[key] = [] if (key.startswith("f_") and key != "f_nct_id") else ""
    st.session_state.selected_nct_id = None
    st.session_state.search_initiated = False

def handle_filter_change():
    st.session_state.selected_nct_id = None

def get_risk_tier(score: float):
    if score >= 75: return "Robust", "Strong success patterns detected.", "#f0fdf4", "#166534"
    if score >= 50: return "Favorable", "Favorable historical indicators.", "#eff6ff", "#1e40af"
    if score >= 25: return "Watchlist", "Mixed signals; mitigation required.", "#fff7ed", "#9a3412"
    return "High Risk", "Significant attrition patterns.", "#fde8e8", "#991b1b"

# ==========================
# 5. FILTER COMPONENT
# ==========================
def render_filter_fields(df, is_sidebar=False):
    curr_df = df.copy()
    
    if is_sidebar:
        containers = [st.container() for _ in range(5)]
        with containers[0]:
            st.multiselect("Company / Sponsor", sorted(curr_df["lead_sponsor_canonical"].dropna().unique()), key="f_sponsor", on_change=handle_filter_change)
            if st.session_state.f_sponsor: curr_df = curr_df[curr_df["lead_sponsor_canonical"].isin(st.session_state.f_sponsor)]
        with containers[1]:
            st.multiselect("Therapeutic Area", sorted(curr_df["therapeutic_area"].dropna().unique()), key="f_ta", on_change=handle_filter_change)
            if st.session_state.f_ta: curr_df = curr_df[curr_df["therapeutic_area"].isin(st.session_state.f_ta)]
        with containers[2]:
            st.multiselect("Indication (GBD)", sorted(curr_df["gbd_indication_name"].dropna().unique()), key="f_indication", on_change=handle_filter_change)
            if st.session_state.f_indication: curr_df = curr_df[curr_df["gbd_indication_name"].isin(st.session_state.f_indication)]
        with containers[3]:
            st.multiselect("Trial Phase", sorted(curr_df["phase"].dropna().unique()), key="f_phase", on_change=handle_filter_change)
            if st.session_state.f_phase: curr_df = curr_df[curr_df["phase"].isin(st.session_state.f_phase)]
        with containers[4]:
            years = sorted([y for y in curr_df["start_year"].unique() if y > 0], reverse=True)
            st.multiselect("Start Year", years, key="f_year", on_change=handle_filter_change)
            if st.session_state.f_year: curr_df = curr_df[curr_df["start_year"].isin(st.session_state.f_year)]
    else:
        # Row 1
        c1, c2 = st.columns(2)
        with c1: st.multiselect("Company / Sponsor", sorted(curr_df["lead_sponsor_canonical"].dropna().unique()), key="f_sponsor")
        with c2: st.multiselect("Therapeutic Area", sorted(curr_df["therapeutic_area"].dropna().unique()), key="f_ta")
        # Row 2
        c3, c4 = st.columns(2)
        with c3: st.text_input("Clinical trial number (AACT)", key="f_nct_id", placeholder="e.g. NCT01234567")
        with c4: st.multiselect("Trial Phase", sorted(curr_df["phase"].dropna().unique()), key="f_phase")
        # Row 3 - Horizontal alignment logic
        c5, c6 = st.columns(2)
        with c5:
            st.multiselect("Start Year", sorted([y for y in curr_df["start_year"].unique() if y > 0], reverse=True), key="f_year")
        with c6:
            # Spacer to match the multiselect label height perfectly
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            bc1, bc2 = st.columns([1.5, 1])
            with bc1:
                if st.button("Search Trials", use_container_width=True, type="primary"):
                    st.session_state.search_initiated = True
                    st.rerun()
            with bc2:
                if st.button("Reset", use_container_width=True, key="btn_hub_reset"):
                    reset_filters()
                    st.rerun()

        if st.session_state.f_sponsor: curr_df = curr_df[curr_df["lead_sponsor_canonical"].isin(st.session_state.f_sponsor)]
        if st.session_state.f_ta: curr_df = curr_df[curr_df["therapeutic_area"].isin(st.session_state.f_ta)]
        if st.session_state.f_nct_id: curr_df = curr_df[curr_df["nct_id"].str.contains(st.session_state.f_nct_id, case=False, na=False)]
        if st.session_state.f_phase: curr_df = curr_df[curr_df["phase"].isin(st.session_state.f_phase)]
        if st.session_state.f_year: curr_df = curr_df[curr_df["start_year"].isin(st.session_state.f_year)]
        st.markdown(f"<div style='text-align:right; font-size:0.8rem; color:#64748b; margin-top:10px;'>{len(curr_df):,} trials matching criteria</div>", unsafe_allow_html=True)
    return curr_df

# ==========================
# 6. FOCUS TRAP & KEYBOARD
# ==========================
st.components.v1.html("""
    <script>
    const doc = window.parent.document;
    doc.addEventListener('keydown', function(e) {
        if (e.key === 'Tab' || e.key === 'Shift') {
            const active = doc.activeElement;
            if (active && active.tagName === 'INPUT' && active.getAttribute('role') === 'combobox') {
                if (e.key === 'Tab') active.blur();
                setTimeout(() => {
                    const container = doc.querySelector('.block-container');
                    const inputs = Array.from(container.querySelectorAll('input[role="combobox"], button')).filter(el => !el.closest('[data-testid="stSidebar"]'));
                    const index = inputs.indexOf(active);
                    if (index > -1) {
                        const nextIdx = (e.shiftKey && e.key === 'Tab') ? index - 1 : index + 1;
                        if (nextIdx >= 0 && nextIdx < inputs.length) { inputs[nextIdx].focus(); if (e.key === 'Tab') e.preventDefault(); }
                    }
                }, 180);
            }
        }
    });
    </script>
""", height=0)

# ==========================
# 7. MAIN LOGIC
# ==========================
x_base = X_ALL.copy()
if st.session_state.get("s_mode", "").lower() != "all":
    x_base = x_base[(x_base["is_correct"] == True) | (x_base["trial_segment"] == "ONGOING")]
if st.session_state.get("s_registry", "").lower() != "all":
    x_base = x_base[x_base["trial_segment"] == "HISTORICAL"]

if st.session_state.search_initiated:
    with st.sidebar:
        st.markdown("<h2 style='color:#f8fafc; font-weight:800;'>Clinical Trials Filter</h2>", unsafe_allow_html=True)
        if st.button("Reset Filter", use_container_width=True, key="btn_sidebar_reset"): 
            reset_filters()
            st.rerun()
        filtered_df = render_filter_fields(x_base, is_sidebar=True)
        st.markdown("<div style='height: 400px;'></div>---", unsafe_allow_html=True)
        st.text_input("Register", key="s_registry", placeholder="all", on_change=handle_filter_change)
        st.text_input("Analysis", key="s_mode", placeholder="all", on_change=handle_filter_change)
else:
    filtered_df = x_base.copy()

t1, t2 = st.columns([3, 1])
with t1: st.markdown("<h1 style='font-size: 3rem; margin-bottom: 0;'>ClinTrialPred <span style='font-size: 1rem; color: #94a3b8; font-weight: normal; vertical-align: middle; margin-left: 15px;'>demo version</span></h1>", unsafe_allow_html=True)
with t2:
    if st.session_state.selected_nct_id:
        if st.button("Predict Completion"): st.session_state.trigger_prediction = True

if not st.session_state.selected_nct_id:
    if not st.session_state.search_initiated:
        # ROW 1: TOP BOX with explicit margin class
        st.markdown('''
            <div class="highlight-box top-box-margin">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <div class="highlight-title">Operational Success & Risk Stratification</div>
                    <div style="font-size:0.65rem; font-weight:800; color:#94a3b8; text-transform:uppercase; letter-spacing:0.1em;">Core Mission</div>
                </div>
                <div class="highlight-text">This predictive engine estimates the likelihood of operational completion and the risk of early termination using only data available at clinical trial initiation. Each trial is systematically evaluated and classified into four distinct tiers—High Risk, Watchlist, Favorable, and Robust—providing a clear and actionable risk profile.</div>
            </div>
        ''', unsafe_allow_html=True)

        # ROW 2: Use gap="large" for horizontal spacing
        col_left, col_right = st.columns([1, 1], gap="large")
        
        with col_left:
            # We keep the container for the shaded background
            with st.container(border=True):
                st.markdown('<div class="search-anchor"></div>', unsafe_allow_html=True)
                st.markdown('''
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 20px;">
                        <div class="highlight-title" style="margin-bottom: 0;">Clinical Trial Portfolio Selection Criteria</div>
                    </div>
                ''', unsafe_allow_html=True)
                render_filter_fields(x_base, is_sidebar=False)
        
        with col_right:
            # We wrap the two boxes in a div that enforces the vertical gap
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
        st.markdown(f"<div style='margin-top: 20px;'></div><div class='result-count-label'>Showing {len(filtered_df):,} trials matching criteria</div>", unsafe_allow_html=True)
        grid_df = filtered_df[["nct_id", "ui_search_label", "lead_sponsor_canonical", "therapeutic_area", "phase", "start_year", "Clinical_Score", "is_correct"]].copy()
        grid_df["Clinical_Score"] = grid_df["Clinical_Score"].round(1)
        grid_df["is_correct"] = grid_df["is_correct"].map({True: "Yes", False: "No"})
        grid_df.columns = ["NCT ID", "Identity", "Sponsor", "Area", "Phase", "Year", "Score", "Accurate"]
        event = st.dataframe(grid_df, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", height=min(450, len(grid_df)*35 + 40))
        if event and event.selection and event.selection.rows:
            st.session_state.selected_nct_id = grid_df.iloc[event.selection.rows[0]]["NCT ID"]
            st.rerun()
else:
    row = X_ALL[X_ALL[ID_COL] == st.session_state.selected_nct_id].iloc[0]
    if st.button("← Back"):
        st.session_state.selected_nct_id = None
        st.rerun()
    with st.expander("Identity", expanded=True):
        st.markdown(f'<div style="display: flex; align-items: baseline; margin-bottom: 10px;"><span class="identity-header-text">{row[ID_COL]}</span><span style="font-size: 1.2rem; color: #475569; font-weight: 600;">{row.get("ui_search_label", "N/A")}</span></div><div class="title-box-container"><span style="color: #64748b; font-size: 0.75rem; text-transform: uppercase; font-weight: 800; display: block; margin-bottom: 8px;">Title</span>{row.get("title", "No title available.")}</div>', unsafe_allow_html=True)
    def render_pillar(title, taxonomy_name, data):
        feats = sorted([ (f_id, f_m) for f_id, f_m in TAXONOMY.items() if f_m.get("ui", {}).get("pillar") == taxonomy_name ], key=lambda x: (x[1].get("ui", {}).get("subgroup", ""), x[1].get("ui", {}).get("priority", 99)))
        with st.expander(title, expanded=False):
            for i in range(0, len(feats), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i+j < len(feats):
                        f_id, f_m = feats[i+j]
                        ui = f_m.get("ui", {})
                        val = data.get(f_id)
                        with cols[j]:
                            if ui.get("options"):
                                labels = [o[1] for o in ui["options"]]
                                map_val = f_m.get("mapping", {}).get(str(val).upper(), [None, "N/A"])[1]
                                st.selectbox(ui.get("label", f_id), labels, index=labels.index(map_val) if map_val in labels else 0, disabled=True, key=f"d_{f_id}")
                            else:
                                st.markdown(f"**{ui.get('label', f_id)}**")
                                st.markdown(f"<div style='background:#ffffff; padding:10px; border:1px solid #cbd5e1; border-radius:6px; font-size:0.9rem; color:#0f172a; min-height:40px; margin-bottom:15px;'>{val if not pd.isna(val) else 'N/A'}</div>", unsafe_allow_html=True)
    c_r1_1, c_r1_2 = st.columns(2)
    with c_r1_1: render_pillar("Therapeutic Context", "Therapeutic Context", row)
    with c_r1_2: render_pillar("Execution Framework", "Execution Framework", row)
    c_r2_1, c_r2_2 = st.columns(2)
    with c_r2_1: render_pillar("Scientific Attempt", "Scientific Attempt", row)
    with c_r2_2: render_pillar("Patient Profile", "Patient Profile", row)
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
            if "error" in res: st.error(f"Audit Unavailable: {res['error']}")
            else:
                score = res.get('score', 0)
                tier, desc, bg, tc = get_risk_tier(score)
                st.components.v1.html("<script>window.parent.document.querySelector('section.main').scrollTo({top: 1300, behavior: 'smooth'});</script>", height=0)
                st.markdown("<hr style='margin: 40px 0;'>", unsafe_allow_html=True)
                cl, cr = st.columns([1.0, 1.4])
                with cl:
                    st.plotly_chart(plot_success_gauge(score), use_container_width=True, config={'displayModeBar': False})
                    st.markdown(f"<div style='background:{bg}; color:{tc}; padding:20px; border-radius:12px; border:1px solid {tc}22;'><div style='font-size:1.4rem; font-weight:800;'>{tier}</div><div>{desc}</div></div>", unsafe_allow_html=True)
                    if res.get('pillar_impacts'): st.plotly_chart(plot_impact_bar(pd.DataFrame(res['pillar_impacts'])), use_container_width=True)
                with cr:
                    if res.get('subcat_impacts'): st.plotly_chart(plot_treemap(res['subcat_impacts'], res.get('pillar_impacts', [])), use_container_width=True)
