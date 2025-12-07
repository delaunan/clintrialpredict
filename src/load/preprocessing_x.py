import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, TargetEncoder
from sklearn.decomposition import PCA  # <--- NEW IMPORT

def smart_preprocessor(X):
    """
    Builds a ColumnTransformer pipeline dynamically based on the columns
    present in the input DataFrame X. This handles cases where features
    have been dropped during feature selection.
    """

    # ==========================================================================
    # A. DEFINE ORIGINAL GROUPS
    # ==========================================================================
    NUM_LOG_COLS = [
        'competition_broad', 'competition_niche', 'num_primary_endpoints', 'number_of_arms'
    ]

    NUM_STD_COLS = [
        'design_rigor_score', 'eligibility_strictness_score', 'criteria_len_log'
    ]

    CAT_ONEHOT_COLS = [
        'agent_category', 'phase', 'sponsor_tier', 'agency_class',
        'has_dmc', 'is_fda_regulated_drug', 'includes_us', 'gender',
        'healthy_volunteers', 'masking', 'allocation', 'intervention_model',
        'primary_purpose', 'is_sick_only', 'is_gender_restricted',
        'child', 'adult', 'older_adult'
    ]

    CAT_TARGET_COLS = [
        'therapeutic_area', 'therapeutic_subgroup_name', 'best_pathology', 'sponsor_clean'
    ]

    EMB_COLS = [f"emb_{i}" for i in range(100)]

    # ==========================================================================
    # B. FILTER GROUPS (The Fix)
    # ==========================================================================
    # Only keep columns that are actually in X (survived the purge)
    valid_log = [c for c in NUM_LOG_COLS if c in X.columns]
    valid_std = [c for c in NUM_STD_COLS if c in X.columns]
    valid_onehot = [c for c in CAT_ONEHOT_COLS if c in X.columns]
    valid_target = [c for c in CAT_TARGET_COLS if c in X.columns]
    valid_emb = [c for c in EMB_COLS if c in X.columns]

    print(f"   > Pipeline Config (Active Features):")
    print(f"     - Log + Scale:    {len(valid_log)} / {len(NUM_LOG_COLS)}")
    print(f"     - Standard Scale: {len(valid_std)} / {len(NUM_STD_COLS)}")
    print(f"     - One-Hot:        {len(valid_onehot)} / {len(CAT_ONEHOT_COLS)}")
    print(f"     - Target Enc:     {len(valid_target)} / {len(CAT_TARGET_COLS)}")
    print(f"     - Embeddings:     {len(valid_emb)} / {len(EMB_COLS)}")

    # ==========================================================================
    # C. DEFINE SUB-PIPELINES
    # ==========================================================================

    # Pipeline A: Log + Scale
    pipe_log = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log1p', FunctionTransformer(np.log1p, validate=False, feature_names_out="one-to-one")),
        ('scaler', StandardScaler())
    ])

    # Pipeline B: Standard Scale
    pipe_std = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline C: One Hot Encoding
    pipe_onehot = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.int32))
    ])

    # Pipeline D: Target Encoding
    pipe_target = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ('encoder', TargetEncoder(target_type='binary', smooth=10.0, random_state=42))
    ])

    # Pipeline E: Embeddings
    pipe_emb = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)), # 1. Fix Holes
        ('scaler', StandardScaler())                                  # 2. Balance Numbers
    ])

    # ==========================================================================
    # D. ASSEMBLE TRANSFORMER
    # ==========================================================================
    transformers = []
    if valid_log: transformers.append(('num_log', pipe_log, valid_log))
    if valid_std: transformers.append(('num_std', pipe_std, valid_std))
    if valid_onehot: transformers.append(('cat_onehot', pipe_onehot, valid_onehot))
    if valid_target: transformers.append(('cat_target', pipe_target, valid_target))
    if valid_emb: transformers.append(('embeddings', pipe_emb, valid_emb))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        verbose_feature_names_out=False
    )

    return preprocessor
