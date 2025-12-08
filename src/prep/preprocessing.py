import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, TargetEncoder

# --- OPTIONAL: TEXT PROCESSING (COMMENTED OUT AS REQUESTED) ---
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# import re

def get_preprocessor():
    """
    Returns a Scikit-Learn ColumnTransformer ready for XGBoost/LightGBM.

    Logic:
    1. Skewed Numerics -> Log1p + StandardScaler
    2. Normal Numerics -> StandardScaler
    3. Low Card Categoricals -> OneHot
    4. High Card Categoricals -> Target Encoding
    5. Embeddings -> Passthrough (or Standardize)
    6. Text (Raw) -> Disabled (using Embeddings instead)
    """

    # ==========================================================================
    # 1. FEATURE GROUPS (Based on Audit)
    # ==========================================================================

    # A. SKEWED NUMERICAL (Apply Log1p to compress outliers)
    # 'competition_broad' goes up to 1000+, 'criteria_len_log' is already logged but safe to scale
    NUM_LOG_COLS = [
        'competition_broad',
        'competition_niche',
        'num_primary_endpoints',
        'number_of_arms'
    ]

    # B. NORMAL / BOUNDED NUMERICAL (Standard Scale)
    # 'start_year' is roughly normal. 'design_rigor_score' is ordinal (0-5).
    NUM_STD_COLS = [
        'design_rigor_score',
        'eligibility_strictness_score',
        'criteria_len_log' # Already log-transformed in loader, just scale here
    ]

    # C. BINARY / LOW CARDINALITY CATEGORICAL (OneHot)
    # We include Binary flags here to ensure they are handled if they have missing values
    CAT_ONEHOT_COLS = [
        # Key Risk Drivers
        'agent_category',       # The Crown Jewel
        'phase',                # Critical
        'sponsor_tier',         # Tier 1 vs 2
        'agency_class',         # Industry vs Other

        # Protocol Flags
        'has_dmc',
        'is_fda_regulated_drug',
        'includes_us',
        'gender',
        'healthy_volunteers',
        'masking',
        'allocation',
        'intervention_model',
        'primary_purpose',

        # Calculated Flags
        'is_sick_only',
        'is_gender_restricted',
        'child',
        'adult',
        'older_adult'
    ]

    # D. HIGH CARDINALITY CATEGORICAL (Target Encoding)
    # These have too many unique values for OneHot (would create 1000+ columns)
    CAT_TARGET_COLS = [
        'therapeutic_area',          # ~23 categories (Borderline, but TE is safer)
        'therapeutic_subgroup_name', # ~1000 categories
        'best_pathology',            # ~1000 categories
        'sponsor_clean'              # ~Thousands
    ]

    # E. EMBEDDINGS (BioBERT)
    # 100 Dimensions pre-calculated
    EMB_COLS = [f"emb_{i}" for i in range(100)]

    # F. TEXT (Raw) - DISABLED
    # TEXT_COLS = ['txt_tags']

    # ==========================================================================
    # 2. SUB-PIPELINES
    # ==========================================================================

    # --- Pipeline A: Log + Scale ---
    # Use median imputation for skewed data
    pipe_log = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log1p', FunctionTransformer(np.log1p, validate=False, feature_names_out="one-to-one")),
        ('scaler', StandardScaler())
    ])

    # --- Pipeline B: Standard Scale ---
    pipe_std = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # --- Pipeline C: One Hot Encoding ---
    # handle_unknown='ignore' is crucial for production (new categories won't crash model)
    pipe_onehot = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.int32))
    ])

    # --- Pipeline D: Target Encoding ---
    # smooth=10.0 prevents overfitting on rare categories
    pipe_target = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ('encoder', TargetEncoder(target_type='binary', smooth=10.0, random_state=42))
    ])

    # --- Pipeline E: Embeddings ---
    # Just impute 0 if missing (rare) and pass through.
    # Scaling embeddings is generally good practice even for XGBoost to keep gradients stable.
    pipe_emb = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])

    # --- Pipeline F: Text (TF-IDF) - DISABLED ---
    # pipe_text = Pipeline([
    #     ('cleaner', ClinicalTextCleaner()), # Assuming class exists if needed
    #     ('tfidf', TfidfVectorizer(max_features=1000)),
    #     ('svd', TruncatedSVD(n_components=50))
    # ])

    # ==========================================================================
    # 3. COMPOSITE TRANSFORMER
    # ==========================================================================

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_log', pipe_log, NUM_LOG_COLS),
            ('num_std', pipe_std, NUM_STD_COLS),
            ('cat_onehot', pipe_onehot, CAT_ONEHOT_COLS),
            ('cat_target', pipe_target, CAT_TARGET_COLS),
            ('embeddings', pipe_emb, EMB_COLS),
            # ('text_tfidf', pipe_text, TEXT_COLS) # Disabled
        ],
        remainder='drop', # CRITICAL: Drops nct_id, target, p_values (Leakage prevention)
        verbose_feature_names_out=False
    )

    return preprocessor







# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================
if __name__ == "__main__":
    # Mock Data Test
    print(">>> Testing Preprocessor...")

    # Create dummy dataframe matching schema
    cols = ['competition_broad', 'competition_niche', 'num_primary_endpoints', 'number_of_arms',
            'start_year', 'design_rigor_score', 'eligibility_strictness_score', 'criteria_len_log',
            'agent_category', 'phase', 'sponsor_tier', 'agency_class', 'has_dmc', 'is_fda_regulated_drug',
            'includes_us', 'gender', 'healthy_volunteers', 'masking', 'allocation', 'intervention_model',
            'primary_purpose', 'is_sick_only', 'is_gender_restricted', 'child', 'adult', 'older_adult',
            'therapeutic_area', 'therapeutic_subgroup_name', 'best_pathology', 'sponsor_clean']

    # Add embeddings
    cols += [f"emb_{i}" for i in range(100)]

    df_mock = pd.DataFrame(np.random.randn(10, len(cols)), columns=cols)

    # Fix categorical columns with strings
    cat_cols = ['agent_category', 'phase', 'sponsor_tier', 'agency_class', 'gender',
                'healthy_volunteers', 'masking', 'allocation', 'intervention_model',
                'primary_purpose', 'therapeutic_area', 'therapeutic_subgroup_name',
                'best_pathology', 'sponsor_clean']

    for c in cat_cols:
        df_mock[c] = 'A'

    # Mock Target
    y_mock = np.random.randint(0, 2, 10)

    # Initialize
    pp = get_preprocessor()

    # Fit Transform
    X_out = pp.fit_transform(df_mock, y_mock)

    print(f"Input Shape: {df_mock.shape}")
    print(f"Output Shape: {X_out.shape}")
    print(">>> Preprocessor is ready.")
