import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, StandardScaler, MinMaxScaler, FunctionTransformer

def get_pipeline():
    """
    Returns the Scikit-Learn ColumnTransformer for the Clinical Trial Risk Engine.
    """

    # 1. Feature Groups (As defined in your analysis)
    # ---------------------------------------------------------
    log_trans_cols = ['num_primary_endpoints', 'number_of_arms'] # Added competition later if needed
    stand_scal_cols = ["start_year"]
    min_max_cols = ["phase_ordinal"]

    cat_binary_cols = [
        'is_international', 'covid_exposure', 'healthy_volunteers',
        'adult', 'child', 'older_adult', 'includes_us'
    ]

    cat_nominal_cols = [
        'gender', 'agency_class', 'masking', 'intervention_model',
        'primary_purpose', 'allocation' # Removed therapeutic_area for now if high cardinality
    ]

    cat_high_card_cols = ['therapeutic_area'] # Moved Area here due to high cardinality potential

    # 2. Sub-Pipelines
    # ---------------------------------------------------------

    # A. Binary: Impute -> OneHot (Drop if binary to keep 1 column)
    pipe_bin = Pipeline([
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("encoder", OneHotEncoder(drop='if_binary', dtype=int, handle_unknown='ignore'))
    ])

    # B. Nominal: Impute -> OneHot (Full)
    pipe_nom = Pipeline([
        ("imputer", SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ("hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int))
    ])

    # C. High Cardinality: Impute -> Target Encoder
    # Why? To handle columns with 50+ categories without creating 50+ columns.
    pipe_high = Pipeline([
        ("imputer", SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ("target", TargetEncoder(target_type='binary', smooth=10.0, random_state=42))
    ])

    # D. Skewed Numeric: Impute -> Log1p -> StandardScale
    # Why? 'Arms' and 'Endpoints' often have long tails (outliers).
    log_std_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", StandardScaler())
    ])

    # E. Standard Numeric
    std_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # F. MinMax Numeric
    minmax_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", MinMaxScaler())
    ])

    # 3. Assembly
    # ---------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("log_std", log_std_pipeline, log_trans_cols),
            ("std", std_pipeline, stand_scal_cols),
            ("minmax", minmax_pipeline, min_max_cols),
            ("cat_binary", pipe_bin, cat_binary_cols),
            ("nominal", pipe_nom, cat_nominal_cols),
            ("high_card", pipe_high, cat_high_card_cols)
        ],
        remainder="drop", # Drop unused columns (like nct_id)
        verbose_feature_names_out=False
    )

    return preprocessor
