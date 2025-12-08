import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ==========================
# CONFIG
# ==========================

#MODEL_PATH = "model.pkl"
DATA_PATH = "project_data.csv"

ID_COL = "nct_id"       # ID van de trial in je CSV
FAIL_COL = "target"     # kolom waar 1 = fail, 0 = geen fail

# Risk thresholds (op basis van p_fail)
LOW_RISK_MAX = 0.25      # p_fail <= 25%  -> Low risk
MEDIUM_RISK_MAX = 0.50   # 25–50%        -> Medium risk
# p_fail > 50%           -> High risk


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


def completion_rate(df: pd.DataFrame, mask: pd.Series):
    """
    Bereken completion rate in een subset.
    We gaan ervan uit dat FAIL_COL 1 = fail, 0 = geen fail is.
    completion_rate = 1 - gemiddelde(failure).
    """
    subset = df[mask]
    if subset.empty:
        return None

    fail_rate = subset[FAIL_COL].mean()
    return 1.0 - fail_rate


# ==========================
# DATA & MODEL LADEN
# ==========================

# Titels
st.title("Clinical Trial Completion Predictor")
st.write("Select a trial and get a prediction.")

# Load the model
model = joblib.load(MODEL_PATH)

# Load dataset with all ongoing trials
trials_df = pd.read_csv(DATA_PATH)

# Kolommen die je als overzicht wilt tonen
display_cols = ["nct_id", "phase", "txt_tags", "txt_criteria", "therapeutic_area"]

# Featurekolommen voor het model
# (ID en target eruit; voeg hier evt. meer kolommen toe die je NIET als feature wilt)
feature_cols = [c for c in trials_df.columns if c not in [ID_COL, FAIL_COL]]


# ==========================
# TRIAL SELECTIE
# ==========================

trial_id = st.selectbox(
    "Choose Clinical Trial",
    trials_df[ID_COL].values
)

# 1-rijige DataFrame met de gekozen trial
selected_trial = trials_df[trials_df[ID_COL] == trial_id].iloc[[0]]

# Give an overview of the selected trial
st.subheader("Selected Trial Overview")
st.write(selected_trial[display_cols])


# ==========================
# VOORSPELLING + UI
# ==========================

if st.button("Make prediction"):

    # Input voor model
    X = selected_trial[feature_cols]

    # ---- kans op falen (1 = fail) uit het model ----
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        # Pak expliciet de kolom voor klasse 1 (fail)
        idx_fail = list(model.classes_).index(1)
        p_fail = float(proba[idx_fail])
    else:
        # Geen predict_proba -> terugvallen op label
        pred = int(model.predict(X)[0])
        p_fail = 1.0 if pred == 1 else 0.0

    st.success(f"Predicted probability of failing: **{p_fail:.2%}**")

    # ==========================
    # 1. OUTCOME SUMMARY
    # ==========================

    st.header("1. Outcome Summary")

    # Completion probability (1 - p_fail)
    p_comp = 1.0 - p_fail

    st.subheader("Completion Probability Gauge")
    st.write("Predicted probability that the trial will be **completed**:")
    st.write(f"**{p_comp:.1%}**")
    st.progress(int(p_comp * 100))

    # Risk level
    st.subheader("Risk Level (Low / Medium / High)")


    # Tiers with colors
    tier, desc = get_risk_tier(p_fail)

    if tier == "Low":
        box = st.success
    elif tier == "Medium":
        box = st.warning
    else:
        box = st.error

    box(f"**{tier} risk** – {desc}")


    # ==========================
    # 2. BENCHMARKS
    # ==========================

    st.header("2. Benchmarks")

    phase = selected_trial["phase"].iloc[0]
    ta = selected_trial["therapeutic_area"].iloc[0]

    phase_rate = completion_rate(trials_df, trials_df["phase"] == phase)
    ta_rate = completion_rate(trials_df, trials_df["therapeutic_area"] == ta)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Completion rate – same phase**")
        st.write(f"{phase_rate:.1%}" if phase_rate is not None else "N/A")

    with col2:
        st.markdown("**Completion rate – same therapeutic area**")
        st.write(f"{ta_rate:.1%}" if ta_rate is not None else "N/A")
