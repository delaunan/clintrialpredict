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


import os

# Define the absolute path to the data directory
# Update this string if running on a different machine
DATA_PATH = "/home/delaunan/code/delaunan/clintrialpredict/data"

# Verify that the directory exists
if os.path.exists(DATA_PATH):
    print(f"Data Path configured: {DATA_PATH}")
else:
    print(f"Error: Path not found: {DATA_PATH}")

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Construct File Path
file_path = os.path.join(DATA_PATH, 'project_data.csv')

# 2. Load Data
try:
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Data Loaded Successfully: {len(df)} rows.")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")

# 3. Sort by Time
# Essential for temporal splitting.
if 'df' in locals():
    df = df.sort_values(by='start_year').reset_index(drop=True)
    print(f"Date Range: {df['start_year'].min()} to {df['start_year'].max()}")


# 1. Define Split Point (80% Train / 20% Test)
split_idx = int(len(df) * 0.8)

# 2. Separate Features and Target
X = df.drop(columns=['target', 'overall_status'])
y = df['target']

# 3. Perform Temporal Split
X_train = X.iloc[:split_idx]
X_test  = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

# 4. Extract Years for Post-Hoc Analysis
train_years = X_train['start_year'].values
test_years  = X_test['start_year'].values

# 5. Verification Report
print(f"TRAIN Set: {train_years.min()} - {train_years.max()} (Rows: {len(X_train)})")
print(f"TEST Set:  {test_years.min()} - {test_years.max()} (Rows: {len(X_test)})")

# Check for COVID coverage in Training
if train_years.max() >= 2019:
    print("Status: Training set includes COVID-era data (2019+).")
else:
    print("Status: Training set ends before COVID era.")


from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import re

# --- CONFIGURATION ---
#N_TEXT_FEATURES = 100

# 1. Define Feature Groups
cat_binary = ['is_international', 'covid_exposure', 'healthy_volunteers',
              'adult', 'child', 'older_adult', 'includes_us']

cat_nonminal = ['gender', 'agency_class', 'masking', 'intervention_model',
               'primary_purpose', 'therapeutic_area', 'allocation']

cat_high_card = ['therapeutic_subgroup_name', 'best_pathology']

# --- TEXT FEATURE (DISABLED FOR BASELINE) ---
# TEXT_TAGS = 'txt_tags'

# 2. Define Stop Words (DISABLED)
# clinical_stop_words = [
#     'study', 'trial', 'clinical', 'phase', 'group', 'cohort', 'arm',
#     'randomized', 'randomised', 'controlled', 'double', 'blind', 'open', 'label',
#     'safety', 'efficacy', 'comparison', 'evaluation',
#     'patient', 'subject', 'participant', 'volunteer'
# ]
# final_stop_words = list(ENGLISH_STOP_WORDS) + clinical_stop_words

# 3. Define Text Cleaning Logic (DISABLED)
# def clean_text_logic(series):
#     s = series.fillna('').str.lower()
#     s = s.str.replace(r'\d+', '', regex=True)
#     s = s.str.replace(r'(\w{2,})ies\b', r'\1y', regex=True)
#     s = s.str.replace(r'(\w{3,})s\b', r'\1', regex=True)
#     return s

# 4. Define Pipelines
pipe_bin = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(drop='if_binary', dtype=int, handle_unknown='ignore')
)

pipe_nom = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='UNKNOWN'),
    OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int)
)

pipe_high = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='UNKNOWN'),
    TargetEncoder(target_type='binary', smooth=10.0, random_state=42)
)

# Text Pipeline (DISABLED)
# pipe_txt = make_pipeline(
#     FunctionTransformer(clean_text_logic, validate=False, feature_names_out='one-to-one'),
#     TfidfVectorizer(max_features=N_TEXT_FEATURES, stop_words=final_stop_words)
# )

# 5. Assemble ColumnTransformer
encoder_step = ColumnTransformer(
    transformers=[
        ('binary', pipe_bin, cat_binary),
        ('nominal', pipe_nom, cat_nonminal),
        ('high_card', pipe_high, cat_high_card),
        # ('text', pipe_txt, TEXT_TAGS)  <-- DISABLED HERE
    ],
    remainder='paththrough',
    verbose_feature_names_out=False
)


print("Processing Categorical Features...")

# 1. Fit on Train, Transform Train
X_train_cat = encoder_step.fit_transform(X_train, y_train)

# 2. Transform Test (No Fitting)
X_test_cat = encoder_step.transform(X_test)

# 3. Output Verification
print(f"Encoded Train Shape: {X_train_cat.shape}")
print(f"Encoded Test Shape:  {X_test_cat.shape}")

# Verify Feature Names
# This should now work because FunctionTransformer passes the names correctly
new_features = encoder_step.get_feature_names_out()
print(f"Total Features: {len(new_features)}")
print(f"Sample Features: {new_features[:10]}")


import numpy as np

# 1. Settings
n_samples = 20
feature_names = encoder_step.get_feature_names_out()

# 2. Pick Random Indices
# We use a fixed seed (42) so you see the same rows every time you run this
rng = np.random.RandomState(42)
random_indices = rng.choice(X_train_cat.shape[0], size=n_samples, replace=False)

# 3. Slice the Data
# We grab the specific rows corresponding to the random indices
sample_raw = X_train_cat[random_indices]

# 4. Handle Sparse Matrix
# TF-IDF often creates a "Sparse Matrix" to save memory.
# We must convert it to a standard "Dense" array to put it in a DataFrame.
try:
    sample_raw = sample_raw.toarray()
except AttributeError:
    pass # It is already a standard array

# 5. Create and Display DataFrame
df_sample = pd.DataFrame(sample_raw, columns=feature_names)

print(f"Displaying {n_samples} Random Rows from the Processed Training Set:")
display(df_sample)

# 6. Quick Stats Check
print("\n--- Data Integrity Check ---")
print(f"Min Value: {df_sample.min().min()}")
print(f"Max Value: {df_sample.max().max()}")
print("If Max > 1, it means Target Encoding is working (probabilities) or TF-IDF is working (scores).")
print("If Max == 1, it might be only Binary/One-Hot features.")

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/home/delaunan/code/delaunan/clintrialpredict/data/project_data.csv")
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

cols_to_drop = ["target", "covid_exposure", "is_international", "min_p_value"]

num_df = numeric_df.drop(columns=cols_to_drop)

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


# 1. Define Split Point (80% Train / 20% Test)
# 1. Define Split Point (80% Train / 20% Test)
split_idx = int(len(data) * 0.8)

# 2. Separate Features and Target
X = data.drop(columns=['target', 'overall_status','nct_id'])
y = data['target']

# 3. Perform Temporal Split
X_train = X.iloc[:split_idx]
X_test  = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

# 4. Extract Years for Post-Hoc Analysis
train_years = X_train['start_year'].values
test_years  = X_test['start_year'].values


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
        ("high_card", pipe_high,cat_high_card_cols)],remainder="drop")

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

import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# -----------------------------------------------------------------------------
# 1. TRAIN DUMMY CLASSIFIER
# -----------------------------------------------------------------------------
# Strategy 'most_frequent' always predicts Class 0 (Completed)
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)

# Get Predictions
y_pred_dummy = dummy.predict(X_test)
y_prob_dummy = dummy.predict_proba(X_test)[:, 1] # Usually all zeros

# Get MVP Predictions (from your existing log_reg_model)
y_pred_mvp = log_reg_model.predict(X_test)
y_prob_mvp = log_reg_model.predict_proba(X_test)[:, 1]

# -----------------------------------------------------------------------------
# 2. VISUAL COMPARISON
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(20, 6))

# --- Plot A: ROC Curves ---
fpr_d, tpr_d, _ = roc_curve(y_test, y_prob_dummy)
fpr_m, tpr_m, _ = roc_curve(y_test, y_prob_mvp)

ax[0].plot(fpr_d, tpr_d, linestyle='--', label=f'Dummy (AUC = 0.50)')
ax[0].plot(fpr_m, tpr_m, linewidth=2, color='orange', label=f'LogReg MVP (AUC = {roc_auc_score(y_test, y_prob_mvp):.2f})')
ax[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax[0].set_title("ROC Curve Comparison")
ax[0].set_xlabel("False Positive Rate")
ax[0].set_ylabel("True Positive Rate")
ax[0].legend()

# --- Plot B: Dummy Confusion Matrix ---
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_dummy,
    normalize='true', cmap='Greys', colorbar=False, ax=ax[1]
)
ax[1].set_title(f"Dummy Model\n(Accuracy: {dummy.score(X_test, y_test):.2f})")

# --- Plot C: MVP Confusion Matrix ---
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_mvp,
    normalize='true', cmap='Blues', colorbar=False, ax=ax[2]
)
ax[2].set_title(f"LogReg MVP\n(Accuracy: {log_reg_model.score(X_test, y_test):.2f})")

plt.tight_layout()
plt.show()
