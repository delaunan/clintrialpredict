import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import requests

# IMPORT PLOTTING UTILS
# (This assumes utils/plot.py exists in the frontend folder)
from utils.plot import plot_success_gauge, plot_impact_bar, plot_treemap

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Clinical Trial Completion Predictor",
    page_icon="🧪",
    layout="wide",
)

# ==========================
# 1. SETUP & PATHS
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data_predict.csv"
#HISTORICAL_PATH = PROJECT_ROOT / "project_data.csv"

# *** CRITICAL: PASTE YOUR GOOGLE CLOUD URL HERE ***
API_URL = "https://clintrialpredict-835962039082.europe-west1.run.app/predict"

PHASE_COL = "phase"
TA_COL = "therapeutic_area"
OUTCOME_COL = "completed"
ID_COL = "nct_id"
LOW_RISK_MAX = 0.33
MEDIUM_RISK_MAX = 0.66

# ==========================
# GLOBAL STYLES
# ==========================
st.markdown(
    """
    <style>
        .main > div { max-width: 1200px; margin: 0 auto; }
        .stButton > button {
            background: #3a3a3a; color: white; padding: 0.55rem 1.2rem;
            border-radius: 8px; font-size: 0.9rem; border: none; font-weight: 500;
        }
        .stButton > button:hover {
            background: #2f2f2f; transform: translateY(-1px);
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.2);
        }
        h1 { font-size: 2.3rem !important; font-weight: 800 !important; letter-spacing: -0.03em; }
        div.streamlit-expander { max-width: 50% !important; margin-left: 0 !important; margin-right: auto !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================
# LOADING DATA (UI ONLY)
# ==========================

@st.cache_data
def load_predict_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Prediction data not found at {DATA_PATH}")
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)

#historical_df = load_historical_data()
X = load_predict_data()

# ==========================
# UI HELPER FUNCTIONS
# ==========================
def get_risk_tier(p_fail: float):
    if p_fail <= LOW_RISK_MAX:
        return "Low", "Model sees few patterns associated with withdrawn/suspended trials."
    elif p_fail <= MEDIUM_RISK_MAX:
        return "Medium", "Mixed signals; trial resembles both successful and failed studies."
    else:
        return "High", "Trial shares strong characteristics with previously halted trials."


# ==========================
# UI: MAIN APPLICATION
# ==========================
st.markdown("# 🧪 ClinTrialPredict")
st.markdown("### Trial selection")

if X.empty:
    st.stop()

# Build display labels
X["short_label"] = X[ID_COL].astype(str) + " — " + X["brief_title"].astype(str)
all_labels = X["short_label"].tolist()
label_to_nct = dict(zip(X["short_label"], X[ID_COL]))

# --- Dropdown ---
selected_label = st.selectbox("Trial (NCT ID — brief title)", all_labels, index=None, placeholder="Select a trial…", key="trial_select")

# Only show details AFTER a trial is selected
if selected_label is not None:
    trial_id = label_to_nct[selected_label]
    selected_trial = X[X[ID_COL] == trial_id].iloc[[0]]
    row = selected_trial.iloc[0]

    # --- TRIAL IDENTITY CARD ---
    st.markdown("## Trial overview")
    st.markdown(f"""
    **{row['official_title']}**
    - **NCT ID:** {row['nct_id']}
    - **Phase:** {row['phase']}
    - **Therapeutic area:** {row['therapeutic_area']}
    - **Pathology:** {row['best_pathology']}
    """)

    left_col, right_col = st.columns(2)

    with left_col:
        with st.expander("Patient & criteria"):
            st.write(f"**Gender:** {row['gender']}")
            st.write(f"**Population flags:** Child={row['child']}, Adult={row['adult']}, Older={row['older_adult']}")
            st.write(f"**Healthy volunteers:** {row['healthy_volunteers']}")
            with st.expander("Full criteria text"): st.write(row.get("txt_criteria", "N/A"))

    with right_col:
        with st.expander("Therapeutic landscape"):
            st.write(f"**Therapeutic area:** {row['therapeutic_area']}")
            st.write(f"**Competition:** {row['competition_broad']} (Broad), {row['competition_niche']} (Niche)")
            st.write(f"**Agent:** {row['agent_category']}")

    with left_col:
        with st.expander("Protocol design"):
            st.write(f"**Phase:** {row['phase']} | **Type:** {row['study_type']}")
            st.write(f"**Allocation:** {row['allocation']} | **Masking:** {row['masking']}")
            st.write(f"**Arms:** {row['number_of_arms']}")

    with right_col:
        with st.expander("Sponsor & operational factors"):
            st.write(f"**Sponsor:** {row['lead_sponsor']} ({row['agency_class']})")
            st.write(f"**USA Sites:** {row['includes_us']}")

    st.markdown("<br>", unsafe_allow_html=True)
    run_prediction = st.button("Make prediction")

    # ==========================
    # PREDICTION DASHBOARD (CLIENT-SIDE)
    # ==========================
    if run_prediction:
        # Prepare Data (Handle NaNs for JSON)
        row_dict = selected_trial.iloc[0].replace({np.nan: None}).to_dict()

        with st.spinner("Consulting the API..."):
            try:
                # 2. Call the API
                response = requests.post(API_URL, json=row_dict)

                if response.status_code == 200:
                    result = response.json()

                    p_comp = result.get('prediction_success', 0)
                    impacts_data = result.get('impacts', [])

                    # Reconstruct DataFrames for Plots
                    if impacts_data:
                        df_impacts = pd.DataFrame(impacts_data)
                        df_pillars = df_impacts.groupby('Pillar', as_index=False)['Impact_Pct'].sum()
                    else:
                        df_impacts = None
                        df_pillars = None

                    # Benchmarking
                    p_fail = 1.0 - p_comp
                    tier, desc = get_risk_tier(p_fail)
                    #bench = compute_benchmarks(historical_df, row, p_comp)

                    # Draw Dashboard
                    st.markdown("## Prediction dashboard")
                    st.markdown("---")

                    col1, col2 = st.columns([1.0, 1.2])

                    with col1:
                        st.markdown("#### Completion & risk")
                        st.plotly_chart(plot_success_gauge(p_comp), use_container_width=True)

                        if tier == "Low": st.success(f"**{tier} risk** – {desc}")
                        elif tier == "Medium": st.warning(f"**{tier} risk** – {desc}")
                        else: st.error(f"**{tier} risk** – {desc}")

                        if df_pillars is not None:
                            st.markdown("#### Pillar impact overview")
                            st.plotly_chart(plot_impact_bar(df_pillars), use_container_width=True)

                    with col2:
                        if df_impacts is not None:
                            st.markdown("#### Drivers map")
                            st.write("High-level view of feature influence.")
                            st.plotly_chart(plot_treemap(df_impacts, df_pillars), use_container_width=True)
                        else:
                            st.info("Visual explanations not available.")

                else:
                    st.error(f"API Error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the API. Is the Google Cloud URL correct?")
            except Exception as e:
                st.error(f"An error occurred: {e}")
