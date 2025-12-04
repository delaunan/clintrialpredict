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
# CUSTOM CLASS: Clinical Text Cleaner (The "No Mercy" Version)
# -------------------------------------------------------------------------
class ClinicalTextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        # 1. GENERIC NOISE (The "Effect" Killer)
        self.generic_stop_words = [
            "effect", "effects", "affect", "impact", "influence", "efficacy",
            "drug", "drugs", "agent", "agents", "compound", "compounds",
            "pilot", "exploratory", "feasibility", "preliminary",
            "function", "activity", "action", "mechanism",
            "hydrochloride", "sodium", "potassium", "calcium", "phosphate", "sulfate", # Chemical salts
            "plus", "versus", "vs", "via", "using", "use", "utilizing"
        ]

        # 2. UNITS & DOSAGE
        self.units_stop_words = [
            "mg", "kg", "ml", "l", "dl", "g", "mcg", "ug", "iu", "mu", "unit", "units",
            "dose", "doses", "dosage", "daily", "day", "days", "week", "weeks", "month", "months", "year", "years",
            "hour", "hours", "hr", "min", "sec",
            "qd", "bid", "tid", "qid", "od", "po", "iv", "im", "sc", "sl",
            "oral", "intravenous", "subcutaneous", "intramuscular", "topical", "inhalation",
            "administered", "administration", "receiving", "received", "taking", "take"
        ]

        # 3. BIOEQUIVALENCE & FORMULATION
        self.formulation_stop_words = [
            "bioavailability", "bioequivalence", "pharmacokinetic", "pharmacokinetics",
            "pk", "pd", "pharmacodynamics", "crossover", "single-dose", "food", "fasting", "fed",
            "tablet", "tablets", "capsule", "capsules", "solution", "suspension",
            "injection", "infusion", "cream", "gel", "patch", "formulation", "relative",
            "investigate", "investigation", "evaluate", "evaluation", "assessment", "assess"
        ]

        # 4. TRIAL DESIGN
        self.design_stop_words = [
            "study", "clinical", "trial", "randomized", "randomised", "phase",
            "group", "arm", "cohort", "subject", "subjects", "patient", "patients", "participant", "participants",
            "comparison", "treatment", "treat", "treating", "therapy", "therapeutic",
            "safety", "tolerability", "finding", "extension", "expansion", "escalation",
            "double", "blind", "label", "placebo", "controlled", "parallel",
            "open", "single", "multiple", "ascending", "multicenter", "international",
            "standard", "care", "soc", "active", "control",
            "double-blind", "placebo-controlled", "open-label", "randomized-controlled",
            "parallel-group", "dose-escalation", "safety-efficacy", "first-in-human", "multi-center",
            "naive", "refractory", "relapsed",
            "combination", "monotherapy", "adjunct", "chemotherapy",
            "i", "ii", "iii", "iv", "v"
        ]

        # 5. DEMOGRAPHICS
        self.demo_stop_words = [
            "adult", "adults", "male", "female", "women", "men", "child", "children", "pediatric",
            "adolescent", "elderly", "geriatric", "old", "young", "age", "gender", "sex", "population",
            "human", "healthy", "volunteer", "volunteers",
            "japanese", "chinese", "asian", "caucasian", "white", "black", "hispanic", "indian"
        ]

        # 6. DISEASE NAMES
        self.disease_stop_words = [
            "cancer", "tumor", "tumors", "solid", "malignancy", "malignancies",
            "disease", "condition", "disorder", "syndrome", "infection",
            "lung", "breast", "prostate", "colorectal", "ovarian", "renal", "kidney", "liver", "hepatic",
            "pancreatic", "pancreas", "gastric", "stomach", "esophageal", "brain", "cns",
            "diabetes", "mellitus", "type", "t1dm", "t2dm",
            "leukemia", "lymphoma", "myeloma", "carcinoma", "melanoma", "sarcoma", "nsclc", "sclc", "crc", "hcc",
            "virus", "viral", "bacterial", "respiratory", "hiv", "hcv", "hbv", "covid", "sars-cov-2",
            "arthritis", "rheumatoid", "ra", "psoriasis", "hepatitis", "alzheimer", "sclerosis", "ms",
            "pain", "chronic", "acute", "severe", "moderate", "mild", "recurrent", "advanced", "metastatic", "stage",
            "cell", "cells", "stem", "non-small", "non-small-cell", "small-cell", "b-cell", "t-cell", "nk-cell",
            "antigen", "receptor", "factor", "primary"
        ]

        # Combine all
        self.final_stop_words = list(ENGLISH_STOP_WORDS) + \
                                self.generic_stop_words + \
                                self.units_stop_words + \
                                self.formulation_stop_words + \
                                self.design_stop_words + \
                                self.demo_stop_words + \
                                self.disease_stop_words

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Handle both DataFrame (from ColumnTransformer) and Series/Array
        if isinstance(X, pd.DataFrame):
            text_series = X.iloc[:, 0]
        else:
            text_series = pd.Series(X.flatten())

        # Apply cleaning
        return text_series.apply(self._clean_text).values

    def _clean_text(self, text):
        if pd.isna(text) or text == '': return ""
        text = str(text).lower()

        # Remove numbers entirely (dosage numbers often confuse TF-IDF without context)
        tokens = text.split()
        clean_tokens = []
        for t in tokens:
            if not t.isdigit():
                clean_tokens.append(t)
        text = ' '.join(clean_tokens)

        # Regex: Keep letters, numbers, hyphens
        text = re.sub(r'[^a-z0-9-/]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_feature_names_out(self, input_features=None):
        if input_features is None: return np.array(["cleaned_text"])
        return np.array(input_features)

def get_pipeline():

    # -------------------------------------------------------------------------
    # 1. DEFINE FEATURE GROUPS
    # -------------------------------------------------------------------------

    # Skewed numerical features -> Log Transform
    log_trans_cols = [
        'competition_niche', 'competition_broad',
        'num_primary_endpoints', 'number_of_arms',
        'criteria_len_log'
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
        'primary_purpose', 'allocation', 'therapeutic_area',
        'sponsor_tier'
    ]

    # High Cardinality Categories -> Target Encoding
    cat_high_card_cols = [
        'therapeutic_subgroup_name', 'best_pathology',
        'sponsor_clean'
    ]

    # Text Features -> TF-IDF + SVD
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

    # Target Encoder for High Cardinality (Requires y during fit)
    pipe_high = Pipeline([
        ("imputer", SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ("target", TargetEncoder(target_type='binary', smooth=10.0, random_state=42))
    ])

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

    # --- TEXT PIPELINE ---
    cleaner = ClinicalTextCleaner()
    tags_pipeline = Pipeline([
        ("cleaner", cleaner),
        ("tfidf", TfidfVectorizer(
            stop_words=cleaner.final_stop_words,
            ngram_range=(1, 2),
            min_df=15,
            max_df=0.20,
            max_features=5000,
            sublinear_tf=True,
            token_pattern=r"(?u)[a-zA-Z0-9-]{2,}"
        )),
        ("svd", TruncatedSVD(n_components=50, random_state=42))
    ])


    # --- EMBEDDING PIPELINE (BioBERT PCA) ---


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
