import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ==========================
# CONFIG
# ==========================

MODEL_PATH = "/Users/juliusvlassenroot/code/delaunan/clintrialpredict/models/ctp_model.joblib"
DATA_PATH = "data_predict.csv"
HISTORICAL_PATH = "project_data.csv"

PHASE_COL = "phase"
TA_COL = "therapeutic_area"
OUTCOME_COL = "completed"

ID_COL = "nct_id"       # ID van de trial in je CSV

from src.prep.preprocessing import preprocessor

# Risk thresholds (op basis van p_fail)
LOW_RISK_MAX = 0.25      # p_fail <= 25%  -> Low risk
MEDIUM_RISK_MAX = 0.50   # 25–50%        -> Medium risk
# p_fail > 50%           -> High risk

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
st.title("Clinical Trial Completion Predictor")
st.write("Select a trial and get a prediction.")

# Load the model
model = joblib.load(MODEL_PATH)

# Load dataset with all ongoing trials
X = pd.read_csv(DATA_PATH)

# Kolommen die je als overzicht wilt tonen
display_cols = ["nct_id", "phase", "txt_tags", "txt_criteria", "therapeutic_area"]

# Featurekolommen voor het model
# (ID en target eruit; voeg hier evt. meer kolommen toe die je NIET als feature wilt)
#feature_cols = [c for c in trials_df.columns if c not in [ID_COL, FAIL_COL]]

import streamlit as st

ID_COL = "nct_id"  # adjust if needed

# -----------------------------------
# TRIAL SELECTION DROPDOWN
# -----------------------------------

st.subheader("Choose Clinical Trial")

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




# def completion_rate(df: pd.DataFrame, mask: pd.Series):
#     """
#     Bereken completion rate in een subset.
#     We gaan ervan uit dat FAIL_COL 1 = fail, 0 = geen fail is.
#     completion_rate = 1 - gemiddelde(failure).
#     """
#     subset = df[mask]
#     if subset.empty:
#         return None

#     fail_rate = subset[FAIL_COL].mean()
#     return 1.0 - fail_rate


# # ==========================
# # 2. BENCHMARKS
# # ==========================

# st.header("2. Benchmarks")

# phase = selected_trial["phase"].iloc[0]
# ta = selected_trial["therapeutic_area"].iloc[0]

# phase_rate = completion_rate(X, X["phase"] == phase)
# ta_rate = completion_rate(X, X["therapeutic_area"] == ta)

# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("**Completion rate – same phase**")
#     st.write(f"{phase_rate:.1%}" if phase_rate is not None else "N/A")

# with col2:
#     st.markdown("**Completion rate – same therapeutic area**")
#     st.write(f"{ta_rate:.1%}" if ta_rate is not None else "N/A")
