import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, TargetEncoder
from sklearn.decomposition import PCA

def identity_transform(x):
    return x

def preprocessor():
    """
    Returns a static ColumnTransformer aligned with v05 High-Alpha Consolidated Schema.
    """

    # ==========================================================================
    # A. DEFINE STRICT COLUMN LISTS
    # ==========================================================================

    NUM_LOG_COLS = [
        'number_of_arms',
        'duration_months'  # Consolidated High-Fidelity Signal (LLM Priority)
    ]

    NUM_STD_COLS = [
        'design_rigor_score',
        'criteria_len_log',
        'competition_broad',
        'competition_niche',
        'competition_agent'
    ]

    BIN_COLS = [
        'has_dmc', 
        'is_sick_only',
        'biomarker_stratification', 
        'pivotal_intent',
        'is_fda_regulated_drug',
        'includes_us'
    ]

    CAT_ONEHOT_COLS = [
        'therapeutic_area',          # Consolidated (GBD Priority)
        'therapeutic_modality',      # v05 LLM
        'clinical_line_of_therapy',  # v05 LLM
        'innovation_tier',           # v05 LLM
        'patient_severity',          # v05 LLM
        'comparator_type',           # v05 LLM
        'endpoint_rigor_tier',       # v05 LLM
        'rarity_tier',               # v05 LLM
        'sponsor_class',             # v05 LLM
        'orphan_status',             # v05 LLM
        'target_pathway_class',      # v05 LLM
        'target_novelty',            # v05 LLM
        'comparator_hurdle_tier',    # v05 LLM
        'protocol_design_sophistication', # v05 LLM
        'phase_group',
        'phase',
        'masking',
        'primary_purpose'
    ]

    CAT_TARGET_COLS = [
        'therapeutic_subgroup_name',
        'consolidated_sponsor'       # Union of LLM and Raw data
    ]

    CRIT_RAW = [f"crit_{i}" for i in range(768)]
    SCI_RAW  = [f"sci_{i}" for i in range(768)]
    ENDP_RAW = [f"endp_{i}" for i in range(768)]

    # ==========================================================================
    # B. DEFINE SUB-PIPELINES
    # ==========================================================================

    pipe_log = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log1p', FunctionTransformer(np.log1p, validate=False, feature_names_out="one-to-one")),
        ('scaler', StandardScaler())
    ])

    pipe_std = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    pipe_binary = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('passthrough', FunctionTransformer(identity_transform, feature_names_out="one-to-one"))
    ])

    pipe_onehot = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.int32))
    ])

    pipe_target = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ('encoder', TargetEncoder(target_type='binary', smooth=100, random_state=42))
    ])

    pipe_crit_pca = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=160, random_state=42))
    ])

    pipe_sci_pca = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=160, random_state=42))
    ])

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
            ('pca_crit',   pipe_crit_pca, CRIT_RAW),
            ('pca_sci',    pipe_sci_pca,  SCI_RAW),
            ('pca_endp',   pipe_endp_pca, ENDP_RAW)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    return preprocessor_inst