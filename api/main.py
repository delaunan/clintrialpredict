import joblib
import pandas as pd
import numpy as np
import json
import xgboost as xgb
# import shap  # Parked for v01
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
from src.scoring.decomposition import DEFAULT_DISABLED_COLS, build_completion_decomposition

app = FastAPI()

# --- CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD ARTIFACTS (Environment Agnostic) ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model_prod_01.joblib"
SHAP_PATH = BASE_DIR / "models" / "shap_values_01.joblib"
THRESHOLDS_PATH = BASE_DIR / "models" / "thresholds_01.json"
TAXONOMY_PATH = BASE_DIR / "models" / "taxonomy_01.json"


def _is_missing(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _canonical_option_key(field_meta, value):
    if _is_missing(value):
        return None

    value_text = str(value).strip()
    value_upper = value_text.upper()
    ui = field_meta.get("ui", {})

    for option_key, option_label in ui.get("options", []) or []:
        if value_upper == str(option_key).upper() or value_text.lower() == str(option_label).lower():
            return str(option_key)

    for option_key, mapped in field_meta.get("mapping", {}).items():
        mapped_value = mapped[0] if isinstance(mapped, list) and mapped else mapped
        mapped_label = mapped[1] if isinstance(mapped, list) and len(mapped) > 1 else option_key

        if (
            value_upper == str(option_key).upper()
            or value_text == str(mapped_value)
            or value_text.lower() == str(mapped_label).lower()
        ):
            return str(option_key)

    return value_text


def _option_encoded_value(field_meta, value):
    option_key = _canonical_option_key(field_meta, value)
    if option_key is None:
        return np.nan

    mapping = field_meta.get("mapping", {})
    if option_key in mapping:
        mapped = mapping[option_key]
        return mapped[0] if isinstance(mapped, list) and mapped else mapped

    numeric = pd.to_numeric(option_key, errors="coerce")
    return numeric if pd.notna(numeric) else option_key


def _option_label(field_meta, value):
    option_key = _canonical_option_key(field_meta, value)
    if option_key is None:
        return "N/A"

    mapping = field_meta.get("mapping", {})
    if option_key in mapping and isinstance(mapping[option_key], list) and len(mapping[option_key]) > 1:
        return str(mapping[option_key][1])

    for candidate_key, candidate_label in field_meta.get("ui", {}).get("options", []) or []:
        if str(candidate_key).upper() == str(option_key).upper():
            return str(candidate_label)

    return str(value)


def _canonical_therapeutic_area(data, registry, ta_threshold_keys):
    raw_ta = data.get("therapeutic_area")
    field_meta = registry.get("therapeutic_area_ml", {})
    for value in (data.get("therapeutic_area_ml"), data.get("therapeutic_area_ui"), raw_ta):
        option_key = _canonical_option_key(field_meta, value)
        if option_key and str(option_key).upper() in ta_threshold_keys:
            return str(option_key).upper()

        if not _is_missing(value) and str(value).upper() in ta_threshold_keys:
            return str(value).upper()

    return "UNCLASSIFIED"


def _normalize_simulation_payload(data, registry, model_input_columns, ta_threshold_keys):
    normalized = {}
    canonical_ta = _canonical_therapeutic_area(data, registry, ta_threshold_keys)

    for col in model_input_columns:
        if col == "therapeutic_area":
            normalized[col] = canonical_ta
            continue

        value = data.get(col)
        field_meta = registry.get(col, {})
        encoding = field_meta.get("encoding")

        if col == "therapeutic_area_ml":
            value = canonical_ta

        if encoding == "ordinal" or field_meta.get("mapping"):
            normalized[col] = _option_encoded_value(field_meta, value)
        elif encoding in {"numeric", "target"} or col.endswith("_ml"):
            numeric = pd.to_numeric(value, errors="coerce")
            normalized[col] = numeric if pd.notna(numeric) else np.nan
        else:
            normalized[col] = np.nan if _is_missing(value) else value

    display_data = dict(data)
    display_data.update(normalized)
    display_data["therapeutic_area"] = canonical_ta

    for field_name, field_meta in registry.items():
        ui_col = field_name.replace("_ml", "_ui")
        if field_name in normalized and (field_meta.get("ui", {}).get("options") or field_meta.get("mapping")):
            value = data.get(field_name, normalized.get(field_name))
            display_data[ui_col] = _option_label(field_meta, value)

    return normalized, display_data, canonical_ta

@app.on_event("startup")
def load_artifacts():
    print("Loading production artifacts...")
    app.state.model = joblib.load(MODEL_PATH)
    app.state.shap_dict = joblib.load(SHAP_PATH)
    
    with open(THRESHOLDS_PATH, 'r') as f:
        app.state.thresholds = json.load(f)
        
    with open(TAXONOMY_PATH, 'r') as f:
        taxonomy_payload = json.load(f)
        # Handle split (v1.0), integrated (v2.0), and named-integrated (v2.1) formats
        if "FEATURE_REGISTRY" in taxonomy_payload:
            app.state.registry = taxonomy_payload.get("FEATURE_REGISTRY", {})
            app.state.ui_schema = taxonomy_payload.get("UI_SCHEMA", {})
        elif "FIELDS" in taxonomy_payload:
            app.state.registry = taxonomy_payload["FIELDS"]
            app.state.ui_schema = taxonomy_payload["FIELDS"]
        else:
            # Fallback for flat format
            app.state.registry = taxonomy_payload
            app.state.ui_schema = taxonomy_payload
    
    # 1. Prepare Feature Metadata from Pipeline
    prep = app.state.model.named_steps['prep']
    app.state.feature_names = prep.get_feature_names_out()
    app.state.model_input_columns = list(getattr(prep, "feature_names_in_", []))
    
    # 2. Reconstruct Taxonomy Mapping
    app.state.feature_to_taxonomy = {}
    app.state.pillars = []
    app.state.subcategories = {} # pillar -> set of subcategories
    
    # SYNC WITH NOTEBOOK: Features that were passed directly without transformer prefix
    app.state.DISABLED_COLS = list(DEFAULT_DISABLED_COLS)

    for feat_name, feat_meta in app.state.registry.items():
        ui = feat_meta.get("ui", {})
        pillar = ui.get("pillar")
        subgroup = ui.get("subgroup")
        label = ui.get("label", feat_name)
        
        if not pillar or not subgroup:
            continue
            
        if pillar not in app.state.pillars:
            app.state.pillars.append(pillar)
        if pillar not in app.state.subcategories:
            app.state.subcategories[pillar] = set()
        app.state.subcategories[pillar].add(subgroup)

        prefix = ""
        if feat_name not in app.state.DISABLED_COLS:
            enc = feat_meta.get("encoding")
            if enc == "ordinal": prefix = "ordinal__"
            elif enc == "target": prefix = "target__"
            elif enc == "numeric":
                if "arms" in feat_name: prefix = "num_arms__"
                elif "duration" in feat_name: prefix = "num_duration__"
            
        prefixed_feat = f"{prefix}{feat_name}"
        
        if feat_name == "therapeutic_area_ml":
            app.state.calibration_target = {"pillar": pillar, "subcategory": subgroup}

    print(f"Taxonomy initialization complete. Pillars: {app.state.pillars}")

@app.get("/")
def root():
    return {"status": "Clinical Trial Predictor API v01 Online"}

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        nct_id = data.get("nct_id")
        simulation_mode = bool(data.get("simulation_mode", False))
        ta = data.get("therapeutic_area", "Unclassified")

        if simulation_mode:
            normalized_inputs, data, ta = _normalize_simulation_payload(
                data,
                app.state.registry,
                app.state.model_input_columns,
                set(app.state.thresholds.get("ta_threshold_logits", {}).keys()),
            )
        
        # 1. SHAP Lookup or Live TreeSHAP
        live_probability = None
        if simulation_mode:
            input_df = pd.DataFrame([normalized_inputs], columns=app.state.model_input_columns)
            live_probability = float(app.state.model.predict_proba(input_df)[0][1])
            transformed = app.state.model.named_steps["prep"].transform(input_df)
            booster = app.state.model.named_steps["clf"].get_booster()
            dmatrix = xgb.DMatrix(transformed, feature_names=list(app.state.feature_names))
            contribs = booster.predict(dmatrix, pred_contribs=True)
            shap_vals = contribs[0][:-1]
            mode = "simulation"
        elif nct_id in app.state.shap_dict:
            shap_vals = app.state.shap_dict[nct_id]
            mode = "audit"
        else:
            return {"error": f"Trial ID {nct_id} not found."}

        return build_completion_decomposition(
            data=data,
            shap_vals=shap_vals,
            registry=app.state.registry,
            thresholds=app.state.thresholds,
            feature_names=app.state.feature_names,
            mode=mode,
            therapeutic_area=ta,
            live_probability=live_probability,
            disabled_cols=app.state.DISABLED_COLS,
            pillar_order=app.state.pillars,
        )

    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}
