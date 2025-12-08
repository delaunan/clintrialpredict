import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, TargetEncoder
from sklearn.decomposition import PCA  # <--- NEW IMPORT

def preprocessor():
    """
    Returns a static ColumnTransformer.
    Does NOT depend on input data X.
    Expects specific columns to exist in the dataframe.
    """

    # ==========================================================================
    # A. DEFINE STRICT COLUMN LISTS
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
        'masking', 'allocation', 'intervention_model',
        'primary_purpose', 'is_sick_only', 'is_gender_restricted',
        'child', 'adult', 'older_adult',
        #'healthy_volunteers'  # <--- REMOVED TO AVOID DATA LEAKAGE
    ]

    CAT_TARGET_COLS = [
        'therapeutic_area', 'therapeutic_subgroup_name', 'best_pathology', 'sponsor_clean'
    ]

    EMB_COLS = [f"emb_{i}" for i in range(100)]


    # ==========================================================================
    # B. DEFINE SUB-PIPELINES
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

    # Pipeline D: Target Encoding (CRITICAL MODIFICATION)
    pipe_target = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ('encoder', TargetEncoder(target_type='binary', smooth=200.0, random_state=42)) # <--- SMOOTHING INCREASED
    ])

    # Pipeline E: Embeddings (CRITICAL MODIFICATION)
    pipe_emb = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)), # 1. Fix Holes
        ('scaler', StandardScaler()),                                  # 2. Balance Numbers
        ('pca', PCA(n_components=20, random_state=42))                 # 3. NEW: Reduce to 20 dimensions
    ])

    # ==========================================================================
    # C. ASSEMBLE FINAL TRANSFORMER
    # ==========================================================================

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_log',    pipe_log,    NUM_LOG_COLS),
            ('num_std',    pipe_std,    NUM_STD_COLS),
            ('cat_onehot', pipe_onehot, CAT_ONEHOT_COLS),
            ('cat_target', pipe_target, CAT_TARGET_COLS),
            ('embeddings', pipe_emb,    EMB_COLS)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    return preprocessor
