import pandas as pd
import joblib
from pathlib import Path

# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "ctp_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# ------------------------------------------------------------------
# Feature schema (MUST match training exactly)
# ------------------------------------------------------------------

MODEL_FEATURES = [
    'nct_id', 'start_date', 'study_type', 'overall_status', 'phase',
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
    Make a prediction from a dictionary of named features.

    Parameters
    ----------
    features : dict
        Keys must match MODEL_FEATURES

    Returns
    -------
    list
        Model predictions
    """

    # Check for missing features
    missing = set(MODEL_FEATURES) - set(features.keys())
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    # Build DataFrame in correct column order
    X = pd.DataFrame(
        [[features[feat] for feat in MODEL_FEATURES]],
        columns=MODEL_FEATURES
    )

    # Run model prediction (pipeline-safe)
    preds = model.predict(X)

    return preds.tolist()
