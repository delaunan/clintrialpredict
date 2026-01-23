import nbformat as nbf
import os

# 1. Define the code for the new cells
markdown_source = r"""#### <REF:PROD_PREDICT_UNIVERSE>
> #### **12. Predict Universe Generation: Full Portfolio Scoring**

This final phase utilizes the `ClinicalTrialLoaderPredict` to ingest the expanded clinical universe (2005–2026), including ongoing trials. We then apply the calibrated production model to generate success scores and SHAP explainability for the entire searchable registry, ensuring 'Audit Mode' is ready for the Streamlit UI."""

ingestion_code = r"""# --- STEP 1: LOAD FULL PREDICT UNIVERSE ---
from src.prep.data_loader_predict import ClinicalTrialLoaderPredict

print(">>> Initializing Predict Loader for expanded universe (2005-2026)...")
predict_loader = ClinicalTrialLoaderPredict(str(DATA_PATH))

# Load and Engineer
df_universe = predict_loader.load_and_clean()
df_universe = predict_loader.add_features(df_universe)

n_hist = len(df_universe[df_universe.trial_segment == 'HISTORICAL'])
n_ongo = len(df_universe[df_universe.trial_segment == 'ONGOING'])

print(f"\n>>> Full Predict Universe: {len(df_universe):,} trials engineered.")
print(f"    - Historical (Labeled): {n_hist:,}")
print(f"    - Ongoing (Predictive):  {n_ongo:,}")"""

scoring_code = r"""# --- STEP 2: SCORING & ARTIFACT EXPORT ---
print(">>> Generating Scores for the full universe...")

# 1. Prepare Features
X_universe = df_universe.drop(columns=cols_to_drop, errors='ignore')

# 2. Calculate SHAP for the entire universe 
# (Re-calculating ensures feature alignment with the final model)
X_universe_trans = model.named_steps['prep'].transform(X_universe)
shap_values_universe = explainer.shap_values(X_universe_trans)

# 3. Generate Scorecard
df_universe_scores = generate_production_scorecard(
    X_universe, shap_values_universe, feature_names, RISK_TAXONOMY,
    base_value=model_base_value,
    thresholds_logit_map=final_threshold_logits,
    global_thresh_logit=global_logit
)

# 4. Merge Metadata for Dashboard
app_data = df_universe[['nct_id', 'brief_title', 'overall_status', 'trial_segment', 
                        'enrollment', 'number_of_facilities', 'start_year', 'therapeutic_area']].copy()
app_data = app_data.merge(df_universe_scores[['Clinical_Score', 'Zone']], left_index=True, right_index=True)

# 5. Add Demo Quality Flag
# Criteria: High Score (>85) or Low Score (<15) + Tier 1 Giant Sponsor
tier_1_mask = df_universe['sponsor_tier'] == 'TIER_1_GIANT'
polarized_mask = (app_data['Clinical_Score'] > 85) | (app_data['Clinical_Score'] < 15)
app_data['is_demo_quality'] = (tier_1_mask & polarized_mask).astype(int)

# 6. Save Artifacts
SEARCH_DATA_PATH = MODELS_PATH / "app_search_data_01.csv"
SHAP_VALS_PATH = MODELS_PATH / "shap_values_01.joblib"

app_data.to_csv(SEARCH_DATA_PATH, index=False)
joblib.dump(shap_values_universe, SHAP_VALS_PATH)

print(f"\n[SUCCESS] Production artifacts exported to /models")
print(f"          - Search Registry: {len(app_data):,} rows")
print(f"          - SHAP Matrix:     {shap_values_universe.shape}")""

# 2. Update the notebook
b_path = 'notebooks/production_01.ipynb'
b = nbf.read(nb_path, as_version=4)

# Create new cells
b.cells.append(nbf.v4.new_markdown_cell(markdown_source))
b.cells.append(nbf.v4.new_code_cell(ingestion_code))
b.cells.append(nbf.v4.new_code_cell(scoring_code))

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully appended Predict Universe cells to {nb_path}")