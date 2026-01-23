
import nbformat as nbf
import os

def create_production_notebook():
    nb = nbf.v4.new_notebook()

    # --- CELL 1: TITLE & INTRODUCTION ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "# **Clinical Trial Success Prediction: Production Pipeline v01**\n\n" \
        "This notebook serves as the **Official Production Engine**. It builds the final predictive model used to feed the " \
        "Streamlit application with XGBoost weights, Therapeutic Area-specific thresholds, and explainability artifacts.\n\n" \
        "### **Key Production Objectives:**\n" \
        "1. **Unified Dataset (2009–2022)**: Training on the full historical cohort to maximize contemporary signal.\n" \
        "2. **Recency-Weighted Calibration**: Optimizing decision boundaries with priority on recent trial outcomes.\n" \
        "3. **Full Explainability Export**: Generating SHAP values for the entire portfolio to support search and 'What-If' analysis.\n" \
        "4. **Artifact Persistence**: Saving the model, thresholds, and search data for the Streamlit app."
    ))

    # --- CELL 2: ENV CONFIG MARKDOWN ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "#### **1. Environment Configuration**\n" \
        "Configuring dynamic module reloading and precision-focused filters to ensure a seamless production workflow."
    ))

    # --- CELL 3: ENV CONFIG CODE ---
    nb.cells.append(nbf.v4.new_code_cell(
        r"""# --- AUTO-RELOAD MAGIC COMMANDS ---
%load_ext autoreload
%autoreload 2

import warnings
from tqdm.auto import tqdm
tqdm.pandas()

# Silence Warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)"""
    ))

    # --- CELL 4: PATH RESOLUTION MARKDOWN ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "#### **2. Project Path Resolution**\n" \
        "Automatically identifies the project root by searching for the `src` directory to ensure portability."
    ))

    # --- CELL 5: PATH RESOLUTION CODE ---
    nb.cells.append(nbf.v4.new_code_cell(
        r"""import sys
from pathlib import Path

# 1. Define the Project Root
current_dir = Path.cwd()
project_root = current_dir

while not (project_root / 'src').exists():
    if project_root == project_root.parent:
        raise FileNotFoundError("Could not find project root containing 'src'")
    project_root = project_root.parent

# 2. Add Project Root to System Path
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 3. Define Key Paths
DATA_PATH = project_root / "data"
MODELS_PATH = project_root / "models"

print(f"Project Root: {project_root}")
print(f"Data Path:    {DATA_PATH}")
print(f"Models Path:  {MODELS_PATH}")"""
    ))

    # --- CELL 6: LIB INIT MARKDOWN ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "#### **3. Library and Custom Module Initialization**\n" \
        "Importing foundational analytical tools and the custom clinical trial pipeline components."
    ))

    # --- CELL 7: LIB INIT CODE ---
    nb.cells.append(nbf.v4.new_code_cell(
        r"""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib

# Custom Modules
from src.prep.data_loader import ClinicalTrialLoader
from src.prep.preprocessing import preprocessor"""
    ))

    # --- CELL 8: DATA PIPELINE MARKDOWN ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "#### **4. Unified Data Pipeline (2009–2022)**\n" \
        "Ingests the analytical cohort. This process strictly filters for trials with a confirmed binary outcome (target)."
    ))

    # --- CELL 9: DATA PIPELINE CODE ---
    nb.cells.append(nbf.v4.new_code_cell(
        r"""# --- DATA LOADING & FEATURE STORE CHECK ---
PROJECT_DATA_PATH = DATA_PATH / "project_data.csv"

if PROJECT_DATA_PATH.exists():
    df = pd.read_csv(PROJECT_DATA_PATH, low_memory=False)
    print(f">>> Loaded existing project_data.csv | Shape: {df.shape}")
else:
    print(">>> project_data.csv not found. Initializing full ingestion pipeline...")
    loader = ClinicalTrialLoader(str(DATA_PATH))
    
    # Cohort Generation: 2009-2022 Industry Phase 2/3 trials with valid target status
    df = loader.load_and_clean()
    
    # Feature Enrichment: Tabular + BioBERT merge
    df = loader.add_features(df)
    
    # Persistence
    loader.check_and_export_nlp(df)
    print(f"\n>>> Data Loaded. Final Shape: {df.shape}")

# Ensure target is strictly valid (Known outcomes only)
df = df.dropna(subset=['target']).copy()
df['target'] = df['target'].astype(int)
print(f">>> Final Production Cohort (Known Targets): {df.shape[0]:,} trials")"""
    ))

    # --- CELL 10: DATA AUDIT MARKDOWN ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "#### **5. Forensic Data Audit**\n" \
        "Verifying dataset completeness and embedding density before entering the training phase."
    ))

    # --- CELL 11: DATA AUDIT CODE ---
    nb.cells.append(nbf.v4.new_code_cell(
        r"""def audit_production_data(df):
    print("="*80)
    print("PRODUCTION DATA AUDIT")
    print("="*80)
    
    # Check for Critical Columns
    critical = ['start_year', 'target', 'therapeutic_area']
    for col in critical:
        status = "[PASS]" if col in df.columns else "[FAIL]"
        print(f"{status} {col:<20}")
        
    # Embedding Presence
    emb_cols = [c for c in df.columns if c.startswith('crit_')]
    if len(emb_cols) > 0:
        print(f"[INFO] Detected {len(emb_cols)} NLP dimensions.")
        if df[emb_cols[0]].sum() != 0:
            print("[PASS] Embeddings contain non-zero signal.")
    else:
        print("[WARN] No Embeddings found. Ensure NLP vectors are attached.")

audit_production_data(df)"""
    ))

    # --- CELL 12: MODEL TRAINING MARKDOWN ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "#### **6. Production Model Configuration: XGBoost**\n" \
        "Training the unified model on the full 2009–2022 cohort using a `random_state=42` for total reproducibility."
    ))

    # --- CELL 13: MODEL TRAINING CODE ---
    nb.cells.append(nbf.v4.new_code_cell(
        r"""from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

print('>>> Preparing Production Training Matrix...')

# Define columns to exclude from the training matrix
cols_meta = ['target', 'nct_id', 'start_date', 'start_year', 'official_title', 'why_stopped']
cols_text = [c for c in df.columns if c.startswith('txt_')]
cols_to_drop = cols_meta + cols_text

X_train = df.drop(columns=cols_to_drop, errors='ignore')
y_train = df['target']

# Imbalance Handling
n_pos = y_train.sum()
ratio = (len(y_train) - n_pos) / n_pos

# Official Production Hyperparameters
xgb = XGBClassifier(
    n_estimators=495, learning_rate=0.02, max_depth=3,
    min_child_weight=25, gamma=10, subsample=0.728,
    colsample_bytree=0.30, reg_alpha=30, reg_lambda=30,
    scale_pos_weight=ratio, tree_method='hist', eval_metric='logloss',
    n_jobs=-1, random_state=42, enable_categorical=False
)

model = Pipeline([('prep', preprocessor()), ('clf', xgb)])

print(f'>>> Fitting Production Pipeline on {X_train.shape[0]:,} trials...')
model.fit(X_train, y_train)
print('    [DONE] Model 01 training complete.')"""
    ))

    # --- CELL 14: CALIBRATION MARKDOWN ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "#### **7. Unified Dynamic Threshold Calibration**\n" \
        "Applying **Recency-Weighted** logic to the full dataset. Trials from 2022 carry the highest weight (1.0), " \
        "while historical data from 2009 carries a lower anchor weight (0.1)."
    ))

    # --- CELL 15: CALIBRATION CODE ---
    nb.cells.append(nbf.v4.new_code_cell(
        r"""from sklearn.metrics import precision_recall_curve
from scipy.special import logit

def find_optimal_threshold(y_true, y_probs, beta=1.15, weights=None):
    p, r, t = precision_recall_curve(y_true, y_probs, sample_weight=weights)
    f = (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-9)
    return t[np.argmax(f[:-1])] if len(t) > 0 else 0.5

print('>>> Starting Production Calibration (2009-2022)...')

# 1. Recency Weighting
y_prob_train = model.predict_proba(X_train)[:, 1]
train_years = df.loc[X_train.index, 'start_year']
y_min, y_max = train_years.min(), train_years.max()
w_floor, w_ceil = 0.1, 1.0
train_weights = w_floor + (w_ceil - w_floor) * (train_years - y_min) / (y_max - y_min)

df_cal = pd.DataFrame({
    'TA': X_train['therapeutic_area'],
    'true': y_train.values,
    'prob': y_prob_train,
    'weight': train_weights
})

# 2. Threshold Optimization
global_thresh = find_optimal_threshold(df_cal['true'], df_cal['prob'], weights=df_cal['weight'])
global_logit = float(logit(np.clip(global_thresh, 1e-5, 1-1e-5)))

final_thresholds = {}
final_threshold_logits = {}
min_samples_policy = 500

ta_counts = df_cal['TA'].value_counts()
for ta, count in ta_counts.items():
    df_ta = df_cal[df_cal['TA'] == ta]
    if count < min_samples_policy or df_ta['true'].nunique() <= 1:
        final_thresholds[ta] = global_thresh
        final_threshold_logits[ta] = global_logit
    else:
        t_ta = find_optimal_threshold(df_ta['true'], df_ta['prob'], weights=df_ta['weight'])
        final_thresholds[ta] = t_ta
        final_threshold_logits[ta] = float(logit(np.clip(t_ta, 1e-5, 1-1e-5)))

print(f"🌍 Global Threshold: {global_thresh:.4f} | Calibrated {len(final_thresholds)} TAs.")"""
    ))

    # --- CELL 16: SHAP MARKDOWN ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "#### **8. Explainability Engine: Full Dataset SHAP Decomposition**\n" \
        "Calculating feature attribution for all production trials to facilitate searching and comparative analysis."
    ))

    # --- CELL 17: SHAP CODE ---
    nb.cells.append(nbf.v4.new_code_cell(
        r"""import shap

print('>>> Initializing SHAP TreeExplainer...')
prep_step = model.named_steps['prep']
# FIX: Enabling verbose feature names to ensure uniqueness across PCA pillars
prep_step.verbose_feature_names_out = True 
X_trans = prep_step.transform(X_train)
feature_names = prep_step.get_feature_names_out()

explainer = shap.TreeExplainer(model.named_steps['clf'])
print('>>> Calculating SHAP values for all trials (this may take a few minutes)...')
shap_values = explainer.shap_values(X_trans)

model_base_value = explainer.expected_value
if isinstance(model_base_value, (list, np.ndarray)):
    model_base_value = model_base_value[0]

print(f"    [DONE] Matrix Shape: {shap_values.shape} | Intercept: {model_base_value:.4f}")"""
    ))

    # --- CELL 18: SCORECARD MARKDOWN ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "#### **9. Clinical Success Scoring Engine**\n"
        "Transforming log-odds and SHAP values into the Success-Oriented Clinical Score (0–100). "
        "We invert the risk signal so that higher scores represent higher success potential."
    ))

    # --- CELL 19: SCORECARD CODE ---
    nb.cells.append(nbf.v4.new_code_cell(
        r"""GAIN_FACTOR = 25.0

def generate_production_scorecard(X, shap_vals, feature_names, base_val, thresh_map, global_logit):
    # Success Potential = Boundary - Model_Risk
    row_threshold_logits = X['therapeutic_area'].map(thresh_map).fillna(global_logit)
    model_pred_logit = base_val + np.sum(shap_vals, axis=1)
    
    delta_logit = row_threshold_logits - model_pred_logit
    scores = (50 + (delta_logit * GAIN_FACTOR)).clip(1, 99)
    
    df_s = pd.DataFrame(index=X.index)
    df_s['Clinical_Score'] = scores
    df_s['Zone'] = pd.cut(df_s['Clinical_Score'], [0, 25, 50, 75, 100], 
                          labels=["High Risk", "Watchlist", "Good", "Robust"])
    
    # Calculate Pillar Offsets for UI Consistency
    df_s['CALIBRATION_OFFSET'] = (row_threshold_logits - base_val) * GAIN_FACTOR
    return df_s

df_scores = generate_production_scorecard(
    X_train, shap_values, feature_names, model_base_value, final_threshold_logits, global_logit
)
print(f">>> Mean Production Success Score: {df_scores['Clinical_Score'].mean():.1f}")"""
    ))

    # --- CELL 20: AUDIT MARKDOWN ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "#### **10. Production Performance Audit**\n"
        "Verifying the model's ranking power and ROI potential on the unified training distribution."
    ))

    # --- CELL 21: CODE (AUDIT + ROI) ---
    nb.cells.append(nbf.v4.new_code_cell(
        r"""from sklearn.metrics import roc_auc_score, average_precision_score

auc_val = roc_auc_score(y_train, y_prob_train)
pr_auc_val = average_precision_score(y_train, y_prob_train)

print(f"Production ROC-AUC: {auc_val:.4f}")
print(f"Production PR-AUC:  {pr_auc_val:.4f}")

# ROI Plot (Failure Capture)
data_roi = pd.DataFrame({'target': y_train, 'prob': y_prob_train}).sort_values('prob', ascending=False)
data_roi['cum_pop'] = np.arange(1, len(data_roi) + 1) / len(data_roi)
data_roi['cum_gain'] = data_roi['target'].cumsum() / data_roi['target'].sum()

plt.figure(figsize=(10, 6))
plt.plot(data_roi['cum_pop'], data_roi['cum_gain'], color='#A83232', lw=2.5, label='Production Capture')
plt.plot([0, 1], [0, 1], '--', color='#555555', alpha=0.6)
plt.title("Production ROI: Portfolio Failure Interception")
plt.xlabel("% of Portfolio Audited")
plt.ylabel("% of Failures Caught")
plt.grid(alpha=0.3)
plt.legend()
plt.show()"""
    ))

    # --- CELL 22: UI SNAPSHOT MARKDOWN ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "#### **11. Production UI Verification Snapshots**\n"
        "Verifying the Gauge, Impact Bar Chart, and Treemap logic for a sample trial. "
        "These visualizations ensure that the math behind the Clinical Success Score is correctly decomposed into the R&D pillars."
    ))

    # --- CELL 23: UI SNAPSHOT CODE ---
    nb.cells.append(nbf.v4.new_code_cell(
        r"""import plotly.graph_objects as go
import re
import textwrap

# 1. RISK TAXONOMY (FULL ACCOUNTING)
RISK_TAXONOMY = {
    "1. Therapeutic Context": {
        "Indication Risk Profile": {"features": ["cat_onehot__therapeutic_area", "cat_target__therapeutic_subgroup_name", "CALIBRATION_OFFSET"]},
        "Development Phase": {"features": ["cat_onehot__phase", "cat_onehot__phase_group"]}
    },
    "2. Scientific Design": {
        "Scientific Rationale": {"features": ["pca_sci"]},
        "Endpoint Strategy": {"features": ["pca_endp"]},
        "Trial Complexity": {"features": ["num_log__number_of_arms", "cat_onehot__primary_purpose"]}
    },
    "3. Execution Framework": {
        "Quality & Bias Control": {"features": ["bin_flags__has_dmc", "num_std__design_rigor_score", "cat_onehot__masking"]},
        "Time Horizon": {"features": ["num_log__duration_months"]},
        "Sponsor Profile": {"features": ["cat_onehot__sponsor_tier"]}
    },
    "4. Patient Profile": {
        "Inclusion Criteria": {"features": ["pca_crit", "num_std__criteria_len_log"]},
        "Patient Acuity": {"features": ["bin_flags__is_sick_only", "bin_flags__is_severe", "bin_flags__is_acute", "bin_flags__is_refractory"]}
    }
}

# 2. UI FUNCTIONS (Simplified Snapshots)
def plot_verif_gauge(val):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=val, title={'text': "Clinical Success Score"}))
    fig.update_layout(height=250); fig.show()

def get_pillar_impacts(df_s, trial_idx, shap_vals, feat_names, taxonomy):
    pos_idx = df_s.index.get_loc(trial_idx)
    row_shap = shap_vals[pos_idx]
    
    impacts = {}
    for pillar, topics in taxonomy.items():
        p_val = 0.0
        for topic, details in topics.items():
            for feat_prefix in details['features']:
                if feat_prefix == "CALIBRATION_OFFSET":
                    p_val += df_s.loc[trial_idx, 'CALIBRATION_OFFSET']
                else:
                    p_val += sum(row_shap[i] * -25.0 for i, f in enumerate(feat_names) if f.startswith(feat_prefix))
        impacts[pillar] = p_val
    return pd.Series(impacts)

# EXECUTE VERIFICATION
trial_idx = df_scores.index[40]
print(f">>> UI Check for NCT ID: {df.loc[trial_idx, 'nct_id']}")
plot_verif_gauge(df_scores.loc[trial_idx, 'Clinical_Score'])

impact_series = get_pillar_impacts(df_scores, trial_idx, shap_values, feature_names, RISK_TAXONOMY)
fig_bar = go.Figure(go.Bar(x=impact_series.values, y=impact_series.index, orientation='h'))
fig_bar.update_layout(title="Pillar Impact Verification", height=300); fig_bar.show()"""
    ))

    # --- CELL 24: EXPORT MARKDOWN ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "#### **12. Production Artifact Export**\n" \
        "Saving the final model pipeline, threshold map, and the application search database."
    ))

    # --- CELL 25: EXPORT CODE ---
    nb.cells.append(nbf.v4.new_code_cell(
        r"""# 1. Save Model & Thresholds
joblib.dump(model, MODELS_PATH / 'model_01.joblib')

cal_export = {
    'global_threshold': global_thresh,
    'ta_thresholds': {ta: float(t) for ta, t in final_thresholds.items()},
    'ta_threshold_logits': final_threshold_logits,
    'base_value': float(model_base_value),
    'gain_factor': GAIN_FACTOR
}
with open(MODELS_PATH / 'thresholds_01.json', 'w') as f:
    json.dump(cal_export, f, indent=4)

# 2. Create Search Database for Streamlit
cols_to_keep = [
    'nct_id', 'brief_title', 'official_title', 'therapeutic_area', 
    'therapeutic_subgroup_name', 'lead_sponsor', 'phase', 'start_year', 
    'target', 'why_stopped'
]
app_df = df[[c for c in cols_to_keep if c in df.columns]].copy()
app_df['Clinical_Score'] = df_scores['Clinical_Score']
app_df['Zone'] = df_scores['Zone']

# Add "Demo Quality" Flag: Prediction matches outcome
row_thresholds = app_df['therapeutic_area'].map(final_thresholds).fillna(global_thresh)
y_pred = (y_prob_train >= row_thresholds).astype(int)
app_df['is_demo_quality'] = (y_pred == app_df['target']).astype(int)

app_df.to_csv(MODELS_PATH / 'app_search_data_01.csv', index=False)

print(">>> ALL PRODUCTION ARTIFACTS EXPORTED SUCCESSFULLY.")"""
    ))

    # --- FINAL WRITE ---
    notebook_path = "notebooks/production_01.ipynb"
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"Successfully created {notebook_path}")

if __name__ == "__main__":
    create_production_notebook()
