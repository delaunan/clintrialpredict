import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, TargetEncoder
from sklearn.decomposition import PCA

def identity_transform(x):
    """
    Returns the input unchanged. Used for pickling stability in FunctionTransformer.
    """
    return x

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
        'number_of_arms',
        'duration_months'
    ]

    NUM_STD_COLS = [
        'design_rigor_score',
        'criteria_len_log'
    ]


    # Logic: If missing, assume 0 (False).
    BIN_COLS = [
        # Patient State
        'is_acute', 'is_refractory', 'is_severe',
        'is_critical_setting',
        # Existing Flags
        'has_dmc', 'is_sick_only'
    ]

    CAT_ONEHOT_COLS = [
        'therapeutic_area',
        'phase_group',
        'phase',
        'sponsor_tier',
        'masking',
        'primary_purpose'
    ]

    CAT_TARGET_COLS = [
        'therapeutic_subgroup_name'
    ]

    # --- RAW EMBEDDING COLUMNS (Inputs for PCA) ---
    CRIT_RAW = [f"crit_{i}" for i in range(768)]
    SCI_RAW  = [f"sci_{i}" for i in range(768)]
    ENDP_RAW = [f"endp_{i}" for i in range(768)]



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

    # Pipeline C: Binary Passthrough ---
    # Imputes NaN with 0, then passes the 0/1 integer through unchanged.
    pipe_binary = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('passthrough', FunctionTransformer(identity_transform, feature_names_out="one-to-one"))
    ])

    # Pipeline D: One Hot Encoding
    pipe_onehot = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.int32))
    ])

    # Pipeline E: Target Encoding (smooth 100 to reduce overfitting)
    pipe_target = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ('encoder', TargetEncoder(target_type='binary', smooth=100, random_state=42))
    ])


    # Channel 1: Criteria (Reduces 768 -> 160)
    pipe_crit_pca = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=160, random_state=42))
    ])

    # Channel 2: Scientific/Title (Reduces 768 -> 160)
    pipe_sci_pca = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=160, random_state=42))
    ])

    # Channel 3: Endpoints (Reduces 768 -> 170)
    pipe_endp_pca = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=170, random_state=42))
    ])



    # ==========================================================================
    # C. ASSEMBLE FINAL TRANSFORMER
    # ==========================================================================

    preprocessor_inst = ColumnTransformer(
        transformers=[
            ('num_log',    pipe_log,    NUM_LOG_COLS),
            ('num_std',    pipe_std,    NUM_STD_COLS),
            ('bin_flags',  pipe_binary, BIN_COLS),
            ('cat_onehot', pipe_onehot, CAT_ONEHOT_COLS),
            ('cat_target', pipe_target, CAT_TARGET_COLS),
            # The 3 PCA Channels
            ('pca_crit',   pipe_crit_pca, CRIT_RAW),
            ('pca_sci',    pipe_sci_pca,  SCI_RAW),
            ('pca_endp',   pipe_endp_pca, ENDP_RAW)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    return preprocessor_inst
