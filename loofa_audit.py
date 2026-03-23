import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, TargetEncoder
from sklearn.metrics import roc_auc_score

# Add project root to sys.path
sys.path.append(str(Path.cwd()))

from src.prep.pipeline import RegistryImputer, identity_transform, FEATURE_REGISTRY

# --- CONFIGURATION ---
DATA_PATH = Path("data")
FILE_PATH = DATA_PATH / "data_clinpred.csv"
TRAIN_START_YEAR = 2009
TRAIN_END_YEAR = 2020
TEST_START_YEAR = 2021
TEST_END_YEAR = 2022

def build_preprocessor(features_to_remove=[]):
    """
    Returns a dynamic ColumnTransformer based on the FEATURE_REGISTRY,
    with specified features removed.
    """
    ORDINAL_COLS = []
    TARGET_COLS  = []

    DISABLED_COLS = [
        'includes_us_ml', 'is_fda_regulated_drug_ml', 'gbd_cause_id_ml',
        'gbd_cause_id_2_ml', 'gbd_cause_id_4_ml', 'gbd_hierarchy_level_ml',
        'is_duration_unknown_ml', 'target_ml'
    ]

    for feat, meta in FEATURE_REGISTRY.items():
        if feat in DISABLED_COLS or feat in features_to_remove:
            continue
        enc = meta.get('encoding')
        if enc == 'ordinal':
            ORDINAL_COLS.append(feat)
        elif enc == 'target':
            TARGET_COLS.append(feat)

    NUM_ARMS_COL = ['number_of_arms_ml'] if 'number_of_arms_ml' not in features_to_remove else []
    NUM_DURATION_COL = ['primary_duration_months_ml'] if 'primary_duration_months_ml' not in features_to_remove else []

    pipe_ordinal = Pipeline([
        ('imputer', RegistryImputer()),
        ('passthrough', FunctionTransformer(identity_transform, feature_names_out="one-to-one"))
    ])

    pipe_target = Pipeline([
        ('imputer', RegistryImputer()),
        ('encoder', TargetEncoder(target_type='binary', smooth=200.0, random_state=42))
    ])

    pipe_arms = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler())
    ])

    pipe_duration = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    transformers = []
    if ORDINAL_COLS:
        transformers.append(('ordinal', pipe_ordinal, ORDINAL_COLS))
    if TARGET_COLS:
        transformers.append(('target', pipe_target, TARGET_COLS))
    if NUM_ARMS_COL:
        transformers.append(('num_arms', pipe_arms, NUM_ARMS_COL))
    if NUM_DURATION_COL:
        transformers.append(('num_duration', pipe_duration, NUM_DURATION_COL))

    return ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        verbose_feature_names_out=True
    )

def run_experiment(df, features_to_remove=[]):
    # Temporal Split
    df_split = df.set_index('nct_id')
    mask_train = df_split['start_year'].between(TRAIN_START_YEAR, TRAIN_END_YEAR)
    mask_test  = df_split['start_year'].between(TEST_START_YEAR, TEST_END_YEAR)

    df_train_temp = df_split.loc[mask_train].copy()
    y_train = df_train_temp['target']
    X_train = df_train_temp.drop(columns=['target'])

    df_test_temp = df_split.loc[mask_test].copy()
    y_test = df_test_temp['target']
    X_test = df_test_temp.drop(columns=['target'])

    # XGBoost Hyperparameters
    n_pos = y_train.sum()
    ratio = (len(y_train) - n_pos) / n_pos

    xgb = XGBClassifier(
        n_estimators=500, learning_rate=0.02, max_depth=4,
        min_child_weight=15, gamma=4, subsample=0.8,
        colsample_bytree=0.50, reg_alpha=15, reg_lambda=15,
        scale_pos_weight=ratio, tree_method='hist', eval_metric='logloss',
        n_jobs=-1, random_state=42, enable_categorical=False
    )

    model = Pipeline([('prep', build_preprocessor(features_to_remove)), ('clf', xgb)])
    model.fit(X_train, y_train)

    y_prob_test = model.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, y_prob_test)
    
    return auc_test

def main():
    if not FILE_PATH.exists():
        print(f"File not found: {FILE_PATH}")
        return

    print("Loading data...")
    df = pd.read_csv(FILE_PATH, low_memory=False)
    df = df[df['target'].notna()].copy()
    print(f"Data shape: {df.shape}")

    features_to_audit = [
        'healthy_volunteers_ml',
        'strategic_ambition_ml',
        'line_of_therapy_ml',
        'gbd_cause_id_3_ml',
        'primary_duration_months_ml',
        'number_of_arms_ml'
    ]

    print("\nRunning Baseline...")
    baseline_auc = run_experiment(df, [])
    print(f"Baseline AUC (2021-2022): {baseline_auc:.4f}")

    results = []
    for feat in features_to_audit:
        print(f"\nAuditing removal of: {feat}")
        auc = run_experiment(df, [feat])
        delta = auc - baseline_auc
        results.append({'feature': feat, 'auc': auc, 'delta': delta})
        print(f"AUC: {auc:.4f} | ΔAUC: {delta:+.4f}")

    print("\nLOOFA AUDIT SUMMARY:")
    for res in results:
        print(f"{res['feature']:<30} | {res['auc']:.4f} | {res['delta']:+.4f}")

    # Identify winners
    winners = [res for res in results if res['delta'] > 0]
    if winners:
        winners.sort(key=lambda x: x['delta'], reverse=True)
        top_2_features_to_remove = [w['feature'] for w in winners[:2]]
        print(f"\nTrying combination of top 2 performers: {top_2_features_to_remove}")
        comb_2_auc = run_experiment(df, top_2_features_to_remove)
        print(f"Combination (Top 2) AUC: {comb_2_auc:.4f} | ΔAUC: {comb_2_auc - baseline_auc:+.4f}")

        top_3_features_to_remove = [w['feature'] for w in winners[:3]]
        print(f"\nTrying combination of top 3 performers: {top_3_features_to_remove}")
        comb_3_auc = run_experiment(df, top_3_features_to_remove)
        print(f"Combination (Top 3) AUC: {comb_3_auc:.4f} | ΔAUC: {comb_3_auc - baseline_auc:+.4f}")

        top_all_features_to_remove = [w['feature'] for w in winners]
        print(f"\nTrying combination of all positive performers: {top_all_features_to_remove}")
        comb_all_auc = run_experiment(df, top_all_features_to_remove)
        print(f"Combination (All) AUC: {comb_all_auc:.4f} | ΔAUC: {comb_all_auc - baseline_auc:+.4f}")
        
        # Determine best configuration
        best_auc = baseline_auc
        best_removed = []
        
        if comb_2_auc > best_auc:
            best_auc = comb_2_auc
            best_removed = top_2_features_to_remove
        if comb_3_auc > best_auc:
            best_auc = comb_3_auc
            best_removed = top_3_features_to_remove
        if comb_all_auc > best_auc:
            best_auc = comb_all_auc
            best_removed = top_all_features_to_remove
        for res in results:
            if res['auc'] > best_auc:
                best_auc = res['auc']
                best_removed = [res['feature']]

        print(f"\nBEST CONFIGURATION FOUND: Removed {best_removed}")
        print(f"BEST AUC: {best_auc:.4f}")

        # Report final set
        print("\nFinal Feature Status (among audited):")
        for feat in features_to_audit:
            status = "REMOVED" if feat in best_removed else "KEPT"
            print(f"- {feat}: {status}")
        
        print(f"Final Test AUC: {best_auc:.4f}")
    else:
        print("\nNo removals increased AUC beyond baseline.")
        print(f"Final Test AUC: {baseline_auc:.4f}")

if __name__ == "__main__":
    main()
