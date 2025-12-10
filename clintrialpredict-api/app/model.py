import pandas as pd
import joblib
import os

# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
#MODEL_PATH = "models/ctp_model.joblib"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "apimodel", "ctp_model.joblib")
model = joblib.load(MODEL_PATH)

# ------------------------------------------------------------------
# Features expected by the trained pipeline (ORDER MATTERS)
# ------------------------------------------------------------------
MODEL_FEATURES = ['nct_id', 'start_date', 'study_type', 'overall_status', 'phase',
    'number_of_arms', 'why_stopped', 'has_dmc', 'is_fda_regulated_drug',
    'start_year', 'target', 'covid_exposure', 'includes_us', 'allocation',
    'intervention_model', 'primary_purpose', 'masking',
    'num_primary_endpoints', 'lead_sponsor', 'agency_class',
    'sponsor_clean', 'sponsor_tier', 'gender', 'healthy_volunteers',
    'criteria_len_log', 'child', 'adult', 'older_adult', 'best_pathology',
    'therapeutic_area', 'tree_number', 'therapeutic_subgroup_name',
    'competition_broad', 'competition_niche', 'txt_tags', 'txt_criteria',
    'name', 'agent_category', 'score_masking', 'score_allocation',
    'score_model', 'design_rigor_score', 'is_gender_restricted',
    'is_sick_only', 'eligibility_strictness_score',
    'emb_0', 'emb_1', 'emb_2', 'emb_3', 'emb_4',
    'emb_5', 'emb_6', 'emb_7', 'emb_8', 'emb_9',
    'emb_10', 'emb_11', 'emb_12', 'emb_13', 'emb_14',
    'emb_15', 'emb_16', 'emb_17', 'emb_18', 'emb_19',
    'min_p_value'
]

# ------------------------------------------------------------------
# Prediction function
# ------------------------------------------------------------------
def predict(features: dict):
    """
    Predict for a single clinical trial input.

    Parameters
    ----------
    features : dict
        Keys must exactly match MODEL_FEATURES

    Returns
    -------
    list
        Model prediction
    """

    # ---- Validate keys strictly ----
    missing = set(MODEL_FEATURES) - set(features.keys())
    extra = set(features.keys()) - set(MODEL_FEATURES)

    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected features: {sorted(extra)}")

    # ---- Build DataFrame in correct order ----
    X = pd.DataFrame([features], columns=MODEL_FEATURES)

    # ---- CRITICAL FIX: force numeric columns to float ----
    # This avoids sklearn imputer / scaler dtype crashes
    for col in X.columns:
        if X[col].dtype != object:
            X[col] = X[col].astype(float)

    # ---- Predict ----
    preds = model.predict(X)
    return preds.tolist()
