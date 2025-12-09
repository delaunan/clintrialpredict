import pandas as pd
import joblib

# Path to your saved XGBoost model
MODEL_PATH = "models/ctp_model.joblib"

# Load the model
model = joblib.load(MODEL_PATH)

# List of features expected by the model
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
    'is_sick_only', 'eligibility_strictness_score', 'emb_0', 'emb_1',
    'emb_2', 'emb_3', 'emb_4', 'emb_5', 'emb_6', 'emb_7', 'emb_8', 'emb_9',
    'emb_10', 'emb_11', 'emb_12', 'emb_13', 'emb_14', 'emb_15', 'emb_16',
    'emb_17', 'emb_18', 'emb_19', 'min_p_value'
]

def predict(features: list):
    """
    Make prediction for a single row of features.
    :param features: list of 60+ features in the same order as MODEL_FEATURES
    :return: list of predictions
    """
    if len(features) != len(MODEL_FEATURES):
        raise ValueError(f"Expected {len(MODEL_FEATURES)} features, got {len(features)}")

    # Convert input list into a DataFrame with column names
    X = pd.DataFrame([features], columns=MODEL_FEATURES)

    # Make prediction
    return model.predict(X).tolist()
