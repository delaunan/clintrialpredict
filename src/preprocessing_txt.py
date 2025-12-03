import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer, TargetEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD

# -------------------------------------------------------------------------
# CUSTOM CLASS: Clinical Text Cleaner
# -------------------------------------------------------------------------
class ClinicalTextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.clinical_stop_words = list(ENGLISH_STOP_WORDS) + [
            "study", "clinical", "trial", "randomized", "randomised", "phase",
            "double", "blind", "label", "placebo", "controlled", "safety",
            "efficacy", "group", "subject", "patient", "participants",
            "year", "month", "week", "day", "dose", "mg", "kg", "daily",
            "treatment", "comparison", "evaluation", "assessment", "vs", "versus"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            text_series = X.iloc[:, 0]
        else:
            text_series = pd.Series(X.flatten())
        return text_series.apply(self._clean_text).values

    def _clean_text(self, text):
        if pd.isna(text) or text == '':
            return ""
        text = str(text).lower()
        text = ' '.join([w for w in text.split() if not w.isdigit()])
        text = re.sub(r'[^a-z0-9-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # --- FIX 2: Add this method so Pipeline knows how to handle feature names ---
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(["cleaned_text"])
        return np.array(input_features)

def get_pipeline():
    # -------------------------------------------------------------------------
    # 1. DEFINE FEATURE GROUPS
    # -------------------------------------------------------------------------
    log_trans_cols = [
        'competition_niche', 'competition_broad',
        'num_primary_endpoints', 'number_of_arms'
    ]
    stand_scal_cols = ["start_year"]
    min_max_cols = ["phase_ordinal"]
    cat_binary_cols = [
        'is_international', 'covid_exposure', 'healthy_volunteers',
        'adult', 'child', 'older_adult', 'includes_us'
    ]
    cat_nominal_cols = [
        'gender', 'agency_class', 'masking', 'intervention_model',
        'primary_purpose', 'allocation', 'therapeutic_area'
    ]
    cat_high_card_cols = [
        'therapeutic_subgroup_name', 'best_pathology'
    ]
    text_tags_col = ['txt_tags']

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

    pipe_high = Pipeline([
        ("imputer", SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ("target", TargetEncoder(target_type='binary', smooth=10.0, random_state=42))
    ])

    # --- FIX 1: Add feature_names_out='one-to-one' to FunctionTransformer ---
    log_std_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, validate=False, feature_names_out="one-to-one")),
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

    # Text Pipeline
    cleaner = ClinicalTextCleaner()
    tags_pipeline = Pipeline([
        ("cleaner", cleaner),
        ("tfidf", TfidfVectorizer(
            stop_words=cleaner.clinical_stop_words,
            ngram_range=(1, 2),
            min_df=10,
            max_df=0.6,
            max_features=5000,
            sublinear_tf=True
        )),
        ("svd", TruncatedSVD(n_components=50, random_state=42))
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
            ("high_card", pipe_high, cat_high_card_cols),
            ("txt_tags_svd", tags_pipeline, text_tags_col)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor
