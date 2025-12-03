import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer, TargetEncoder

def get_pipeline():
    # -------------------------------------------------------------------------
    # 1. DEFINE FEATURE GROUPS
    # -------------------------------------------------------------------------

    # Skewed numerical features -> Log Transform
    log_trans_cols = [
        'competition_niche', 'competition_broad',
        'num_primary_endpoints', 'number_of_arms'
    ]

    # Normal numerical features -> Standard Scaler
    stand_scal_cols = ["start_year"]

    # Bounded numerical features -> MinMax Scaler
    min_max_cols = ["phase_ordinal"]

    # Binary Categories -> OneHot (drop one)
    cat_binary_cols = [
        'is_international', 'covid_exposure', 'healthy_volunteers',
        'adult', 'child', 'older_adult', 'includes_us'
    ]

    # Nominal Categories (Low Cardinality) -> OneHot (keep all)
    cat_nominal_cols = [
        'gender', 'agency_class', 'masking', 'intervention_model',
        'primary_purpose', 'allocation', 'therapeutic_area' # Added back
    ]

    # High Cardinality Categories -> Target Encoding
    cat_high_card_cols = [
        'therapeutic_subgroup_name', 'best_pathology'
    ]

    # -------------------------------------------------------------------------
    # 2. DEFINE SUB-PIPELINES
    # -------------------------------------------------------------------------

    pipe_bin = Pipeline([
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("encoder", OneHotEncoder(drop='if_binary', dtype=int, handle_unknown='ignore'))
    ])

    pipe_nom = Pipeline([
        ("imputer", SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ("hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int))
    ])

    # Target Encoder for High Cardinality (Requires y during fit)
    pipe_high = Pipeline([
        ("imputer", SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ("target", TargetEncoder(target_type='binary', smooth=10.0, random_state=42))
    ])

    log_std_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", StandardScaler())
    ])

    std_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    minmax_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", MinMaxScaler())
    ])

    # -------------------------------------------------------------------------
    # 3. ASSEMBLE PREPROCESSOR
    # -------------------------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("log_std", log_std_pipeline, log_trans_cols),
            ("std", std_pipeline, stand_scal_cols),
            ("minmax", minmax_pipeline, min_max_cols),
            ("cat_binary", pipe_bin, cat_binary_cols),
            ("nominal", pipe_nom, cat_nominal_cols),
            ("high_card", pipe_high, cat_high_card_cols)
        ],
        remainder="drop", # Drop text columns (processed separately) or IDs
        verbose_feature_names_out=False
    )

    return preprocessor
