
import os
import sys
import pandas as pd
import numpy as np
import shap
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from pathlib import Path

# Add project root to sys.path
project_root = Path("/home/delaunan/code/delaunan/clintrialpredict")
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.prep.pipeline import preprocessor

# Load Data
DATA_PATH = project_root / "data"
DATA_CLINPRED_PATH = DATA_PATH / "data_clinpred.csv"

print(f"Loading data from {DATA_CLINPRED_PATH}...")
df = pd.read_csv(DATA_CLINPRED_PATH, low_memory=False)

# Filter for Historical Outcomes
df = df[df['target'].notna()].copy()
print(f"Filtered Shape: {df.shape}")

# Temporal Split
TRAIN_START_YEAR = 2009
TRAIN_END_YEAR = 2020
TEST_START_YEAR = 2021
TEST_END_YEAR = 2022

df_split = df.set_index('nct_id')
cols_to_keep = [c for c in df_split.columns if c.endswith('_ml') or
                c.startswith(('crit_', 'sci_', 'endp_')) or
                c == 'therapeutic_area']

mask_train = df_split['start_year'].between(TRAIN_START_YEAR, TRAIN_END_YEAR)
mask_test  = df_split['start_year'].between(TEST_START_YEAR, TEST_END_YEAR)

X_train = df_split.loc[mask_train, cols_to_keep]
y_train = df_split.loc[mask_train, 'target']
X_test = df_split.loc[mask_test, cols_to_keep]
y_test = df_split.loc[mask_test, 'target']

print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

# Imbalance Handling
n_pos = y_train.sum()
ratio = (len(y_train) - n_pos) / n_pos

# XGBoost Model with parameters from notebook
xgb = XGBClassifier(
    n_estimators=500, learning_rate=0.02, max_depth=4,
    min_child_weight=15, gamma=4, subsample=0.8,
    colsample_bytree=0.50, reg_alpha=15, reg_lambda=15,
    scale_pos_weight=ratio, tree_method='hist', eval_metric='logloss',
    n_jobs=-1, random_state=42, enable_categorical=False
)

model = Pipeline([('prep', preprocessor()), ('clf', xgb)])

print("Fitting model...")
model.fit(X_train, y_train)

# Performance Check
y_prob_train = model.predict_proba(X_train)[:, 1]
auc_train = roc_auc_score(y_train, y_prob_train)
y_prob_test = model.predict_proba(X_test)[:, 1]
auc_test = roc_auc_score(y_test, y_prob_test)
print(f"TRAIN AUC: {auc_train:.4f}")
print(f"TEST  AUC: {auc_test:.4f}")

# SHAP Calculation
print("Calculating SHAP values...")
prep_step = model.named_steps['prep']
X_train_trans = prep_step.transform(X_train)
X_test_trans = prep_step.transform(X_test)
feature_names = prep_step.get_feature_names_out()

explainer = shap.TreeExplainer(model.named_steps['clf'])
shap_values_train = explainer.shap_values(X_train_trans)
shap_values_test = explainer.shap_values(X_test_trans)

# SHAP values might be a list of two for binary classification in some versions
if isinstance(shap_values_train, list):
    shap_values_train = shap_values_train[0]
if isinstance(shap_values_test, list):
    shap_values_test = shap_values_test[0]

# Average Absolute SHAP
avg_abs_shap_train = np.abs(shap_values_train).mean(axis=0)
avg_abs_shap_test = np.abs(shap_values_test).mean(axis=0)

shap_df = pd.DataFrame({
    'feature': feature_names,
    'avg_abs_shap_train': avg_abs_shap_train,
    'avg_abs_shap_test': avg_abs_shap_test
})

# Identify Top 10 for each
top_10_train = shap_df.sort_values('avg_abs_shap_train', ascending=False).head(10).copy()
top_10_test = shap_df.sort_values('avg_abs_shap_test', ascending=False).head(10).copy()

# Ranks
shap_df['rank_train'] = shap_df['avg_abs_shap_train'].rank(ascending=False)
shap_df['rank_test'] = shap_df['avg_abs_shap_test'].rank(ascending=False)
shap_df['rank_diff'] = shap_df['rank_test'] - shap_df['rank_train'] # positive means rank dropped (became less important)
shap_df['mag_ratio'] = shap_df['avg_abs_shap_test'] / shap_df['avg_abs_shap_train']

print("\nTOP 10 FEATURES - TRAINING SET (2009-2020):")
print(top_10_train[['feature', 'avg_abs_shap_train']].to_string(index=False))

print("\nTOP 10 FEATURES - TEST SET (2021-2022):")
print(top_10_test[['feature', 'avg_abs_shap_test']].to_string(index=False))

print("\nCOMPARISON OF TOP FEATURES (Train vs Test):")
# Join top 10s to see overlap and differences
comparison = shap_df.sort_values('avg_abs_shap_train', ascending=False)
print(comparison[['feature', 'avg_abs_shap_train', 'avg_abs_shap_test', 'rank_train', 'rank_test', 'rank_diff', 'mag_ratio']].to_string(index=False))

print("\nSignificant Rank or Magnitude Differences:")
# Look for features that changed rank significantly or magnitude changed significantly
sig_diff = shap_df[(np.abs(shap_df['rank_diff']) >= 3) | (shap_df['mag_ratio'] > 1.5) | (shap_df['mag_ratio'] < 0.67)]
print(sig_diff.sort_values('rank_diff', ascending=False).to_string(index=False))
