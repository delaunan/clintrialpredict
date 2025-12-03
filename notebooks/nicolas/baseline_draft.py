import pandas as pd
import numpy as np
import os
import csv

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & ROBUST PATH SETUP
# -----------------------------------------------------------------------------

# Dynamic Path Finding: Looks for the 'data' folder relative to where you are

DATA_PATH = "/home/delaunan/code/delaunan/clintrialpredict/data"

print(f">>> DATA_PATH set to: {os.path.abspath(DATA_PATH)}")

OUTPUT_FILE = 'project_data.csv'

# ROBUST LOADING PARAMETERS
AACT_LOAD_PARAMS = {
    "sep": "|",
    "dtype": str,
    "header": 0,
    "quotechar": '"',
    "quoting": csv.QUOTE_MINIMAL,
    "low_memory": False,
    "on_bad_lines": "warn"
}

print(">>> Setup Complete. Ready to process.")

# -----------------------------------------------------------------------------
# 2. THE FUNNEL: LOADING & FILTERING (Updated with Year Filter)
# -----------------------------------------------------------------------------
print(">>> Loading Studies & Applying Filters...")

# A. Load Studies
cols_studies = [
    'nct_id', 'overall_status', 'study_type', 'phase',
    'start_date', 'start_date_type',
    'number_of_arms', 'official_title', 'why_stopped'
]
df = pd.read_csv(os.path.join(DATA_PATH, 'studies.txt'), usecols=cols_studies, **AACT_LOAD_PARAMS)

# B. Filter: Interventional Only
df = df[df['study_type'] == 'INTERVENTIONAL'].copy()

# C. Filter: Drugs Only
df_int = pd.read_csv(os.path.join(DATA_PATH, 'interventions.txt'), usecols=['nct_id', 'intervention_type'], **AACT_LOAD_PARAMS)
drug_ids = df_int[df_int['intervention_type'].str.upper().isin(['DRUG', 'BIOLOGICAL'])]['nct_id'].unique()
df = df[df['nct_id'].isin(drug_ids)]

# D. Filter: Closed Statuses Only
allowed_statuses = ['COMPLETED', 'TERMINATED', 'WITHDRAWN', 'SUSPENDED']
df = df[df['overall_status'].isin(allowed_statuses)]

# E. Filter: Exclude Phase 0 and Phase 4 (Refined Scope)
excluded_phases = ['EARLY_PHASE1', 'PHASE4', 'NA']
df = df[~df['phase'].isin(excluded_phases)]

# F. Create Target & Fix Dates
df['target'] = df['overall_status'].apply(lambda x: 0 if x == 'COMPLETED' else 1)
df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
df['start_year'] = df['start_date'].dt.year

# --- NEW FILTER: VALID YEARS ONLY ---
# Drop 1900 (Errors) and Future Dates (Invalid for training)
current_year = pd.Timestamp.now().year
df = df[df['start_year'].between(2000, current_year-1)]

print(f"   - Core Cohort Size (Phases 1-3, Years 2000-{current_year-1}): {len(df)} trials")


# -----------------------------------------------------------------------------
# 3. MEDICAL HIERARCHY & SUBGROUPS
# -----------------------------------------------------------------------------
print(">>> Attaching Medical Hierarchy...")

# A. Load Smart Lookup (Best Term per Trial)
df_smart = pd.read_csv(os.path.join(DATA_PATH, 'smart_pathology_lookup.csv'))
df = df.merge(df_smart, on='nct_id', how='left')

# B. Fill Missing
df['therapeutic_area'] = df['therapeutic_area'].fillna('Other/Unclassified')
df['best_pathology'] = df['best_pathology'].fillna('Unknown')

# C. Create Subgroup Code (Level 2 Hierarchy)
df['therapeutic_subgroup'] = df['tree_number'].astype(str).apply(
    lambda x: x[:7] if pd.notna(x) and len(x) >= 7 else 'Unknown'
)

# D. Map Subgroup Code to Name
df_lookup = pd.read_csv(os.path.join(DATA_PATH, 'mesh_lookup.csv'), sep='|')
code_to_name = pd.Series(df_lookup.mesh_term.values, index=df_lookup.tree_number).to_dict()
df['therapeutic_subgroup_name'] = df['therapeutic_subgroup'].map(code_to_name).fillna('Unknown Subgroup')


# -----------------------------------------------------------------------------
# 4. DUAL-LEVEL CROWDING (Niche vs Broad)
# -----------------------------------------------------------------------------
print(">>> Calculating Competition Intensity (Dual Level)...")

# A. Standardize Phase for Grouping
phase_group_map = {
    'PHASE1': 'PHASE1', 'PHASE1/PHASE2': 'PHASE2',
    'PHASE2': 'PHASE2', 'PHASE2/PHASE3': 'PHASE3', 'PHASE3': 'PHASE3'
}
df['phase_group'] = df['phase'].map(phase_group_map).fillna('UNKNOWN')

# --- LEVEL 1: BROAD COMPETITION (Area + Phase) ---
grid_broad = df.groupby(['start_year', 'therapeutic_area', 'phase_group']).size().reset_index(name='count')
dict_broad = dict(zip(zip(grid_broad['start_year'], grid_broad['therapeutic_area'], grid_broad['phase_group']), grid_broad['count']))

def get_broad_crowding(row):
    y, area, ph = row['start_year'], row['therapeutic_area'], row['phase_group']
    if pd.isna(y): return 0
    return dict_broad.get((y, area, ph), 0) + dict_broad.get((y+1, area, ph), 0) + dict_broad.get((y+2, area, ph), 0)

df['competition_broad'] = df.apply(get_broad_crowding, axis=1)

# --- LEVEL 2: NICHE COMPETITION (Subgroup + Phase) ---
grid_niche = df.groupby(['start_year', 'therapeutic_subgroup', 'phase_group']).size().reset_index(name='count')
dict_niche = dict(zip(zip(grid_niche['start_year'], grid_niche['therapeutic_subgroup'], grid_niche['phase_group']), grid_niche['count']))

def get_niche_crowding(row):
    y, sub, ph = row['start_year'], row['therapeutic_subgroup'], row['phase_group']
    if pd.isna(y) or sub == 'Unknown': return 0
    return dict_niche.get((y, sub, ph), 0) + dict_niche.get((y+1, sub, ph), 0) + dict_niche.get((y+2, sub, ph), 0)

df['competition_niche'] = df.apply(get_niche_crowding, axis=1)

df.drop(columns=['phase_group'], inplace=True)
print("   - Created 'competition_broad' and 'competition_niche'")


# -----------------------------------------------------------------------------
# 5. PROTOCOL DETAILS (Eligibility, Endpoints) - UPDATED
# -----------------------------------------------------------------------------
print(">>> Extracting Eligibility & Endpoints...")

# A. Load Eligibility Fields (No Age Parsing)
# We rely on the pre-calculated flags (adult/child/older_adult) instead of parsing numbers.
cols_elig = [
    'nct_id',
    'criteria',
    'gender', 'healthy_volunteers',
    'adult', 'child', 'older_adult'
]

df_elig = pd.read_csv(os.path.join(DATA_PATH, 'eligibilities.txt'),
                      usecols=cols_elig,
                      **AACT_LOAD_PARAMS)

# B. Merge into Main DataFrame
df = df.merge(df_elig, on='nct_id', how='left')

# C. Endpoint Counts
df_calc = pd.read_csv(os.path.join(DATA_PATH, 'calculated_values.txt'),
                      usecols=['nct_id', 'number_of_primary_outcomes_to_measure'],
                      **AACT_LOAD_PARAMS)
df = df.merge(df_calc, on='nct_id', how='left')
df['num_primary_endpoints'] = pd.to_numeric(df['number_of_primary_outcomes_to_measure'], errors='coerce').fillna(1)

# D. P-Values (Analysis Only)
df_outcomes = pd.read_csv(os.path.join(DATA_PATH, 'outcomes.txt'), usecols=['id', 'nct_id', 'outcome_type'], **AACT_LOAD_PARAMS)
prim_ids = df_outcomes[df_outcomes['outcome_type'] == 'PRIMARY']['id'].unique()

df_an = pd.read_csv(os.path.join(DATA_PATH, 'outcome_analyses.txt'), usecols=['outcome_id', 'p_value'], **AACT_LOAD_PARAMS)
df_an = df_an[df_an['outcome_id'].isin(prim_ids)]
df_an['p_value_num'] = pd.to_numeric(df_an['p_value'], errors='coerce')

min_p = df_an.groupby('outcome_id')['p_value_num'].min().reset_index()
min_p = min_p.merge(df_outcomes[['id', 'nct_id']], left_on='outcome_id', right_on='id')
trial_p = min_p.groupby('nct_id')['p_value_num'].min().reset_index(name='min_p_value')

df = df.merge(trial_p, on='nct_id', how='left')


# -----------------------------------------------------------------------------
# 6. OPERATIONAL PROXIES, SPONSORS & EXTERNAL FACTORS (Updated)
# -----------------------------------------------------------------------------
print(">>> Merging Operational Features & Calculating COVID Exposure...")

# A. Phase Ordinal
phase_map = {'PHASE1': 1, 'PHASE1/PHASE2': 1.5, 'PHASE2': 2, 'PHASE2/PHASE3': 2.5, 'PHASE3': 3}
df['phase_ordinal'] = df['phase'].map(phase_map).fillna(0)
df = df[df['phase_ordinal'] > 0] # Drop unknown phases

# B. COVID Exposure
df['covid_exposure'] = df['start_year'].between(2019, 2021).astype(int)

# C. Geography (International Flag + US Flag)
# Logic:
# 1. International = Sites in >1 unique country.
# 2. Includes US = 'United States' is listed as a location.
df_countries = pd.read_csv(os.path.join(DATA_PATH, 'countries.txt'), usecols=['nct_id', 'name'], **AACT_LOAD_PARAMS)

country_stats = df_countries.groupby('nct_id')['name'].agg(
    cnt='nunique',
    includes_us=lambda x: 1 if 'United States' in x.values else 0
).reset_index()

df = df.merge(country_stats, on='nct_id', how='left')

# Create the flags
df['is_international'] = (df['cnt'] > 1).astype(int)
df['includes_us'] = df['includes_us'].fillna(0).astype(int)

# Drop the raw count (leakage prevention)
df.drop(columns=['cnt'], inplace=True)

# D. Sponsors & Design
df_sponsors = pd.read_csv(os.path.join(DATA_PATH, 'sponsors.txt'), **AACT_LOAD_PARAMS)
df_lead = df_sponsors[df_sponsors['lead_or_collaborator'] == 'lead'][['nct_id', 'agency_class']].drop_duplicates('nct_id')
df = df.merge(df_lead, on='nct_id', how='left')

cols_des = ['nct_id', 'allocation', 'intervention_model', 'masking', 'primary_purpose']
df_des = pd.read_csv(os.path.join(DATA_PATH, 'designs.txt'), usecols=cols_des, **AACT_LOAD_PARAMS)
df = df.merge(df_des, on='nct_id', how='left')

print("   - Created 'is_international' and 'includes_us' flags.")
print("   - Dropped leaky counts (facilities/countries).")


# -----------------------------------------------------------------------------
# 7. TEXT FEATURE ENGINEERING (Corrected: Golden Trio + Keep Criteria)
# -----------------------------------------------------------------------------
print(">>> Engineering Text Features (High-Signal Only)...")

# A. Load Components
# 1. Keywords
df_keys = pd.read_csv(os.path.join(DATA_PATH, 'keywords.txt'), usecols=['nct_id', 'name'], **AACT_LOAD_PARAMS)
keys_grouped = df_keys.groupby('nct_id')['name'].apply(lambda x: " ".join(x.dropna().astype(str))).reset_index(name='txt_keywords')

# 2. Intervention Names (DRUGS ONLY - NO DESCRIPTIONS)
df_int = pd.read_csv(os.path.join(DATA_PATH, 'interventions.txt'), usecols=['nct_id', 'intervention_type', 'name'], **AACT_LOAD_PARAMS)
df_int = df_int[df_int['intervention_type'].str.upper().isin(['DRUG', 'BIOLOGICAL'])]
int_names = df_int.groupby('nct_id')['name'].apply(lambda x: " ".join(x.dropna().astype(str))).reset_index(name='txt_int_names')

# B. Merge Components
df = df.merge(keys_grouped, on='nct_id', how='left')
df = df.merge(int_names, on='nct_id', how='left')

# Fill NaNs
text_cols = ['official_title', 'txt_keywords', 'txt_int_names', 'criteria']
df[text_cols] = df[text_cols].fillna("")

# C. Create Final Features

# 1. Stream A: TAGS (The "What") -> TF-IDF
# We join them with spaces. The Title + Keywords + Drug Name is 95% Nouns.
print("   - Creating 'txt_tags' (Title + Keywords + Drug Names)...")
df['txt_tags'] = (
    df['official_title'] + " " +
    df['txt_keywords'] + " " +
    df['txt_int_names']
)

# 2. Stream B: COMPLEXITY (The "How Hard") -> BERT
# We explicitly rename 'criteria' to 'txt_criteria' to mark it for the next pipeline.
print("   - Preserving 'txt_criteria' for BERT processing...")
df['txt_criteria'] = df['criteria']

# D. Cleanup
# We DROP the components we just used, but we KEEP the final results.
cols_to_drop = [
    'start_date', 'start_date_type', 'tree_number', 'number_of_primary_outcomes_to_measure',
    'official_title', 'txt_keywords', 'txt_int_names', 'criteria'
]
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# E. Save
df.to_csv(os.path.join(DATA_PATH, OUTPUT_FILE), index=False, quoting=csv.QUOTE_MINIMAL)

print(f"\\n>>> SUCCESS: Final Dataset saved to {OUTPUT_FILE}")
print(f"    Rows: {len(df)}")
print(f"    Columns: {len(df.columns)}")
print(f"    Text Columns Present: 'txt_tags', 'txt_criteria'")



numeric_df = data.select_dtypes(include=['number'])
numeric_df.columns


categorical_cols = data.columns.difference(numeric_df.columns)
categorical_cols

numeric_df.hist(figsize=(12, 10), bins=30)
plt.tight_layout()
plt.show()


import seaborn as sns
sns.boxplot(x=data["number_of_arms"])
plt.show()

import numpy as np
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

#These are the different numeric columns that need to pass through different branches of the pipeline
log_trans_cols = ['competition_niche', 'competition_broad', 'num_primary_endpoints', 'number_of_arms']
stand_scal_cols = ["start_year"]
min_max_cols = ["phase_ordinal"]

cat_binary_cols = ['is_international', 'covid_exposure', 'healthy_volunteers',
              'adult', 'child', 'older_adult', 'includes_us']

cat_nominal_cols = ['gender', 'agency_class', 'masking', 'intervention_model',
               'primary_purpose', 'therapeutic_area', 'allocation']

cat_high_card_cols = ['therapeutic_subgroup_name', 'best_pathology']

pipe_bin = Pipeline([
    ("imputer",SimpleImputer(strategy='most_frequent')),
    ("encoder",OneHotEncoder(drop='if_binary', dtype=int, handle_unknown='ignore'))]
)

pipe_nom = Pipeline([
    ("imputer",SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
    ("hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int))]
)

pipe_high = Pipeline([
    ("imputer",SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
    ("target",TargetEncoder(target_type='binary', smooth=10.0, random_state=42))]
)

#Pipeline for highly skewed columns
log_std_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("log1p", FunctionTransformer(np.log1p, validate=False)),
    ("scaler", StandardScaler())])

#Pipeline for StandardScaling
std_pipeline = Pipeline([
     ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())])

#Pipeline for MinMax Scaling
minmax_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("scaler", MinMaxScaler())])

scaler = ColumnTransformer(
    transformers=[
        ("log_std", log_std_pipeline, log_trans_cols),
        ("std", std_pipeline, stand_scal_cols),
        ("minmax", minmax_pipeline, min_max_cols),
        ("cat_binary", pipe_bin,cat_binary_cols),
        ("nominal", pipe_nom,cat_nominal_cols),
        ("high_card", pipe_high,cat_high_card_cols)],remainder="passthrough")

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

log_reg_model = Pipeline(steps=[
    ("preprocess", scaler),
    ("model", LogisticRegression(class_weight='balanced', solver='liblinear', penalty='l2', C=0.01,max_iter=1000, random_state=42))])


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

log_reg_model = Pipeline(steps=[
    ("preprocess", scaler),
    ("model", LogisticRegression(class_weight='balanced', solver='liblinear', penalty='l2', C=0.01,max_iter=1000, random_state=42))])

log_reg_model.fit(X_train, y_train)
log_reg_model.predict(X_test)

import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# 1. Get Predictions (Probabilities are crucial for Risk Scoring)
y_pred = log_reg_model.predict(X_test)
y_prob = log_reg_model.predict_proba(X_test)[:, 1]  # Probability of Class 1 (Failure)

# 2. Calculate Key Metrics
roc_score = roc_auc_score(y_test, y_prob)
pr_score = average_precision_score(y_test, y_prob)

print(f"--- MODEL PERFORMANCE METRICS ---")
print(f"ROC-AUC Score:      {roc_score:.4f}  (0.5 = Random, 1.0 = Perfect)")
print(f"PR-AUC Score:       {pr_score:.4f}   (Baseline: {y_test.mean():.4f})")
print("-" * 40)
print("CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# 3. Visualizations (The "Truth" Charts)
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# A. Confusion Matrix (Normalized to show percentages)
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    normalize='true',
    cmap='Blues',
    display_labels=['Completed', 'Failed'],
    ax=ax[0]
)
ax[0].set_title("Confusion Matrix (Normalized)")

# B. ROC Curve (Trade-off between TP and FP)
RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax[1], name='LogReg')
ax[1].set_title(f"ROC Curve (AUC = {roc_score:.2f})")
ax[1].plot([0, 1], [0, 1], "k--", label="Chance") # Add diagonal line

# C. Precision-Recall Curve (Best for Imbalanced Data)
PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax[2], name='LogReg')
ax[2].set_title(f"PR Curve (Avg Prec = {pr_score:.2f})")
ax[2].plot([0, 1], [y_test.mean(), y_test.mean()], "k--", label="Baseline") # Add baseline

plt.tight_layout()
plt.show()


from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Create the "Dumb" Model
# strategy='most_frequent' means "Always predict the majority class (0)"
dummy = DummyClassifier(strategy='most_frequent')

# 2. Fit (It just looks at the counts of 0 vs 1)
dummy.fit(X_train, y_train)

# 3. Predict
y_pred_dummy = dummy.predict(X_test)

# 4. Evaluate
print(f"Dumb Model Accuracy: {accuracy_score(y_test, y_pred_dummy):.4f}")
print("\nClassification Report (Notice Class 1 is 0.00):")
print(classification_report(y_test, y_pred_dummy, zero_division=0))
