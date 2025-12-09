import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ==========================
# CONFIG
# ==========================

from pathlib import Path
import sys

st.set_page_config(
    page_title="Clinical Trial Completion Predictor",
    page_icon="🧪",
    layout="wide",
)
# ============================================================
# 1. Locate the Project Root
# ============================================================
# app.py sits in the top-level project folder
PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# 2. Add "src" to Python Path
# ============================================================
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))


# ============================================================
# 3. Define All Important Paths (portable!)
# ============================================================
MODEL_PATH = PROJECT_ROOT / "models" / "ctp_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "data_predict.csv"
HISTORICAL_PATH = PROJECT_ROOT / "data" / "project_data.csv"


# ============================================================
# 4. Example usage (optional)
# ============================================================
def main():
    print(f"Project Root:    {PROJECT_ROOT}")
    print(f"Model Path:      {MODEL_PATH}")
    print(f"Predict Data:    {DATA_PATH}")
    print(f"Historical Data: {HISTORICAL_PATH}")

    # Example import from src (works because we added src to sys.path)
    # from inference import run_prediction
    # run_prediction(MODEL_PATH, DATA_PATH)


if __name__ == "__main__":
    main()

PHASE_COL = "phase"
TA_COL = "therapeutic_area"
OUTCOME_COL = "completed"

ID_COL = "nct_id"

from src.prep.preprocessing import preprocessor

# Risk thresholds (op basis van p_fail)
LOW_RISK_MAX = 0.25      # p_fail <= 25%  -> Low risk
MEDIUM_RISK_MAX = 0.50   # 25–50%        -> Medium risk
# p_fail > 50%           -> High risk
# --------------------------------------------------
# GLOBAL STYLES
# --------------------------------------------------
# ---------- Global style tweaks ----------
st.markdown(
    """
    <style>
        /* Center the main block and limit width */
        .main > div {
            max-width: 900px;
            margin: 0 auto;
        }

        .stButton > button {
    background: #3a3a3a;
    color: white;
    padding: 0.55rem 1.2rem;
    border-radius: 8px;
    font-size: 0.9rem;
    border: none;
    font-weight: 500;
}
.stButton > button:hover {
    background: #2f2f2f;
}


        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.2);
        }

        /* Card look for sections */
        .card {
            padding: 1.5rem 1.75rem;
            border-radius: 1rem;
            background: #FFFFFF;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
            border: 1px solid #E5E7EB;
        }

        h1 {
            font-size: 2.6rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Shrink Streamlit Expanders to Half Width ----
st.markdown("""
<style>

    /* Target the actual Streamlit expander container */
    div.streamlit-expander {
        max-width: 50% !important;     /* Make it 50% width */
        margin-left: 0 !important;     /* Align left */
        margin-right: auto !important; /* Prevent centering */
    }

</style>
""", unsafe_allow_html=True)



# ==========================
# Loading Data and Model
# ==========================

@st.cache_data
def load_historical_data(path: str = HISTORICAL_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # failed = 1  -> did NOT complete
    # failed = 0  -> completed
    df[OUTCOME_COL] = 1 - df["scientific_success"]

    return df

historical_df = load_historical_data()

# Titels
st.markdown("### 🧪 Clinical Trial Completion Predictor")
st.write("Select a clinical trial to generate a prediction on its completion.")

# Load the model
model = joblib.load(MODEL_PATH)

# Load dataset with all ongoing trials
X = pd.read_csv(DATA_PATH)

ID_COL = "nct_id"  # adjust if needed

# -----------------------------------
# TRIAL SELECTION DROPDOWN
# -----------------------------------


# Build nice readable label: "NCTID — Brief Title"
X["short_label"] = (
    X[ID_COL].astype(str)
    + " — " + X["brief_title"].astype(str)
)

# Map pretty label → NCT ID
label_to_nct = dict(zip(X["short_label"], X[ID_COL]))

# Dropdown
selected_label = st.selectbox(
    "Select a trial",
    X["short_label"].tolist()
)

# Convert back to NCT ID
trial_id = label_to_nct[selected_label]

# Pull selected trial row
selected_trial = X[X[ID_COL] == trial_id].iloc[[0]]
row = selected_trial.iloc[0]
# ==========================
# HELPERS
# ==========================

def get_risk_tier(p_fail: float):
    """Geef (tier, beschrijving) terug op basis van kans op falen."""
    if p_fail <= LOW_RISK_MAX:
        return (
            "Low",
            "Model sees few patterns associated with withdrawn/suspended trials."
        )
    elif p_fail <= MEDIUM_RISK_MAX:
        return (
            "Medium",
            "Mixed signals; trial resembles both successful and failed studies."
        )
    else:
        return (
            "High",
            "Trial shares strong characteristics with previously halted trials."
        )

def compute_benchmarks(
    historical_df: pd.DataFrame,
    current_row: pd.Series,
    p_comp: float,
) -> dict:
    """Compare predicted completion probability to historical data."""

    # Overall completion rate across all historical trials
    overall_rate = historical_df[OUTCOME_COL].mean()

    # Similar = same phase + same therapeutic area
    mask_similar = (
        (historical_df[PHASE_COL] == current_row[PHASE_COL]) &
        (historical_df[TA_COL] == current_row[TA_COL])
    )
    similar_df = historical_df[mask_similar]

    if not similar_df.empty:
        similar_rate = similar_df[OUTCOME_COL].mean()
        n_similar = len(similar_df)
    else:
        similar_rate = np.nan
        n_similar = 0

    return {
        "overall_rate": overall_rate,
        "similar_rate": similar_rate,
        "n_similar": n_similar,
    }

def build_summary(row: pd.Series, p_comp: float, tier: str, bench: dict) -> str:
    phase_val = str(row.get(PHASE_COL, "unknown phase"))
    ta_val = str(row.get(TA_COL, "this therapeutic area"))

    base = (
        f"This Phase {phase_val} trial in {ta_val} has an estimated "
        f"completion probability of {p_comp:.1%}"
    )

    if bench is not None and not np.isnan(bench.get("similar_rate", np.nan)):
        diff = p_comp - bench["similar_rate"]
        direction = "higher" if diff >= 0 else "lower"
        base += (
            f", which is {abs(diff):.1%} {direction} than the historical "
            f"completion rate for {bench['n_similar']} similar trials"
        )

    if tier:
        base += f" and is classified as {tier.lower()} risk."

    return base

# ==========================
# Prediction and Scores
# ==========================

if st.button("Make prediction"):

    # ---- Compute failure probability ----
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(selected_trial)[0]
        idx_fail = list(model.classes_).index(1)
        p_fail = float(proba[idx_fail])
    else:
        pred = int(model.predict(selected_trial)[0])
        p_fail = 1.0 if pred == 1 else 0.0

    # ---- Compute completion probability ----
    p_comp = 1.0 - p_fail

    # ---- OUTCOME SUMMARY ----
    st.subheader("1. Outcome Summary")

    st.markdown("#### Completion Probability Gauge")
    st.write("Predicted probability that the trial will be **completed**:")
    st.write(f"**{p_comp:.1%}**")

    st.progress(int(p_comp * 100))

    # ---- RISK LEVEL ----
    st.markdown("#### Risk Level (Low / Medium / High)")

    tier, desc = get_risk_tier(p_fail)

    if tier == "Low":
        box = st.success
    elif tier == "Medium":
        box = st.warning
    else:
        box = st.error

    box(f"**{tier} risk** – {desc}")

    # =========================
    # BENCHMARKS & COMPARISON
    # =========================
    bench = compute_benchmarks(historical_df, row, p_comp)

    st.markdown("#### How does this trial compare?")

    c1, c2 = st.columns(2)

    # Overall historical completion rate
    c1.metric(
        "Overall completion rate (historical)",
        f"{bench['overall_rate']:.1%}"
    )

    # Similar trials: same phase & therapeutic area
    if not np.isnan(bench["similar_rate"]):
        label = f"Similar trials (Phase {row[PHASE_COL]}, {row[TA_COL]})"
        c2.metric(
            label,
            f"{bench['similar_rate']:.1%}",
            help=f"Based on n = {bench['n_similar']} historical trials."
        )
    else:
        c2.info("No similar historical trials found for this phase/therapeutic area.")

    # Short narrative summary
    summary_text = build_summary(row, p_comp, tier, bench)
    st.markdown("#### Summary")
    st.markdown(summary_text)

# -----------------------------------
# TRIAL IDENTITY CARD
# -----------------------------------
st.subheader("Selected Trial Overview")

st.markdown(f"""
**{row['official_title']}**

- **NCT ID:** {row['nct_id']}
- **Phase:** {row['phase']}
- **Therapeutic Area:** {row['therapeutic_area']}
""")

# 👇 create two columns, use only the left one
col1, col2 = st.columns([1, 1])

with col1:
    with st.expander("Protocol Design", expanded=False):
        st.write(f"**Primary Purpose:** {row['primary_purpose']}")
        st.write(f"**Intervention Model:** {row['intervention_model']}")
        st.write(f"**Allocation:** {row['allocation']}")
        st.write(f"**Masking:** {row['masking']}")
        st.write(f"**Number of Arms:** {row['number_of_arms']}")
        st.write(f"**# Primary Endpoints:** {row['num_primary_endpoints']}")
        st.write(f"**Data Monitoring Committee (DMC):** {row['has_dmc']}")

    with st.expander("Eligibility & Population", expanded=False):
        st.write(f"**Eligibility Strictness Score:** {row['eligibility_strictness_score']}")
        st.write(f"**Criteria Length (log):** {row['criteria_len_log']}")
        st.write(f"**Gender Restriction:** {row['gender']} (restricted: {row['is_gender_restricted']})")
        st.write(f"**Healthy Volunteers:** {row['healthy_volunteers']}")
        st.write(
            "**Population Flags:** "
            f"Child={row['child']}, Adult={row['adult']}, Older Adult={row['older_adult']}"
        )

        with st.expander("Full Criteria Text", expanded=False):
            st.write(row["txt_criteria"])

    with st.expander("Sponsor & Operational Factors", expanded=False):
        st.write(f"**Lead Sponsor:** {row['lead_sponsor']}")
        st.write(f"**Sponsor Tier:** {row['sponsor_tier']}")
        st.write(f"**Sponsor Class:** {row['agency_class']}")
        st.write(f"**Includes U.S. Sites:** {row['includes_us']}")
        st.write(f"**Study Start Year:** {row['start_year']}")
        st.write(f"**COVID Exposure:** {row['covid_exposure']}")

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
