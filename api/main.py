import joblib
import pandas as pd
import numpy as np
import json
# import shap  # Parked for v01
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path

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
    
    # 2. Reconstruct Taxonomy Mapping
    app.state.feature_to_taxonomy = {}
    app.state.pillars = set()
    app.state.subcategories = {} # pillar -> set of subcategories
    
    # SYNC WITH NOTEBOOK: Features that were passed directly without transformer prefix
    app.state.DISABLED_COLS = [
        'includes_us_ml', 'is_fda_regulated_drug_ml', 'gbd_cause_id_ml',
        'gbd_cause_id_2_ml', 'gbd_cause_id_4_ml', 'gbd_hierarchy_level_ml',
        'is_duration_unknown_ml', 'target',  'masking_ml',
        'therapeutic_area_ml', 'strategic_ambition_ml', 'intervention_model_ml'
    ]

    for feat_name, feat_meta in app.state.registry.items():
        ui = feat_meta.get("ui", {})
        pillar = ui.get("pillar")
        subgroup = ui.get("subgroup")
        label = ui.get("label", feat_name)
        
        if not pillar or not subgroup:
            continue
            
        app.state.pillars.add(pillar)
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
    return {"status": "Clinical Trial Predictor API v01 Online (Audit Only)"}

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        nct_id = data.get("nct_id")
        ta = data.get("therapeutic_area", "Unclassified")
        
        # 1. Calibration Constants
        ta_threshold_logits = app.state.thresholds.get("ta_threshold_logits", {})
        threshold_logit = ta_threshold_logits.get(ta, app.state.thresholds.get("global_threshold_logit", 0.0))
        gain_factor = app.state.thresholds.get("gain_factor", 25.0)
        intercept = app.state.thresholds.get("base_value", 0.0)
        
        # 2. SHAP Lookup
        if nct_id in app.state.shap_dict:
            shap_vals = app.state.shap_dict[nct_id]
            mode = "audit"
        else:
            return {"error": f"Trial ID {nct_id} not found."}
        
        # 3. IMPACT AGGREGATION (High Precision)
        sub_sums_raw = {} # (pillar, subcategory) -> float
        sub_features = {} # (pillar, subcategory) -> list
        
        feat_to_idx = {name: i for i, name in enumerate(app.state.feature_names)}
        mapped_indices = set()

        for feat_name, feat_meta in app.state.registry.items():
            ui = feat_meta.get("ui", {})
            p = ui.get("pillar")
            s = ui.get("subgroup")
            label = ui.get("label", feat_name)
            
            if not p or not s or p == "Metadata":
                continue
                
            impact = 0.0
            prefix = ""
            if feat_name not in app.state.DISABLED_COLS:
                enc = feat_meta.get("encoding")
                if enc == "ordinal": prefix = "ordinal__"
                elif enc == "target": prefix = "target__"
                elif enc == "numeric":
                    if "arms" in feat_name: prefix = "num_arms__"
                    elif "duration" in feat_name: prefix = "num_duration__"
            
            prefixed_feat = f"{prefix}{feat_name}"
            
            for full_name, idx in feat_to_idx.items():
                if full_name == prefixed_feat or full_name.startswith(f"{prefixed_feat}_"):
                    impact += -float(shap_vals[idx]) * gain_factor
                    mapped_indices.add(idx)
            
            ui_col = feat_name.replace("_ml", "_ui")
            if "gbd_cause_id_3" in feat_name: ui_col = "gbd_indication_name_3"
            val_to_show = data.get(ui_col, data.get(feat_name, "N/A"))
            if isinstance(val_to_show, (float, int)): val_to_show = f"{float(val_to_show):.1f}"
            elif not val_to_show: val_to_show = "N/A"
            
            feat_str = f"{label}: <b>{val_to_show}</b>"
            
            key = (p, s)
            sub_sums_raw[key] = sub_sums_raw.get(key, 0.0) + impact
            if key not in sub_features: sub_features[key] = []
            sub_features[key].append((ui.get("priority", 99), feat_str))

        # DIAGNOSTIC: Check for unmapped SHAP signal
        all_indices = set(range(len(shap_vals)))
        unmapped_indices = all_indices - mapped_indices
        if unmapped_indices:
            unmapped_impact = 0.0
            unmapped_details = []
            for idx in unmapped_indices:
                name = app.state.feature_names[idx]
                imp = -float(shap_vals[idx]) * gain_factor
                unmapped_impact += imp
                unmapped_details.append(f"{name}: {imp:+.2f}")
            
            # Put into a generic bucket to preserve mathematical parity
            key = ("Therapeutic Context", "Other Model Signals")
            sub_sums_raw[key] = sub_sums_raw.get(key, 0.0) + unmapped_impact
            if key not in sub_features: sub_features[key] = []
            sub_features[key].append((999, f"Unmapped internal factors: <b>{len(unmapped_indices)}</b>"))
            print(f"DEBUG: Unmapped Signal Detected: {unmapped_impact:+.2f} pts across {len(unmapped_indices)} features.")
            # print(f"DEBUG: Unmapped Details: {unmapped_details}")

        # 4. Calibration & Systematic Rounding for Absolute Parity
        calibration_offset_pts = (threshold_logit - intercept) * gain_factor
        cp = app.state.calibration_target["pillar"]
        cs = app.state.calibration_target["subcategory"]
        sub_sums_raw[(cp, cs)] = sub_sums_raw.get((cp, cs), 0.0) + calibration_offset_pts

        # STEP A: Round subcategories to 1 decimal point (UI standard)
        final_subcats = []
        pillar_totals = {p: 0.0 for p in app.state.pillars}
        
        for (p, s), raw_imp in sub_sums_raw.items():
            rounded_imp = round(raw_imp, 1)
            # Ensure -0.0 becomes 0.0 for clean UI
            if rounded_imp == -0.0: rounded_imp = 0.0
            
            pillar_totals[p] = round(pillar_totals[p] + rounded_imp, 1)
            
            # Narrative lookup
            narrative = ""
            for fn, fm in app.state.registry.items():
                u = fm.get("ui", {})
                if u.get("pillar") == p and u.get("subgroup") == s:
                    narrative = u.get("pos_impact" if rounded_imp >= 0 else "neg_impact", "")
                    break

            final_subcats.append({
                "Pillar": p,
                "Subcategory": s,
                "Impact": rounded_imp,
                "Narrative": narrative,
                "FeatureDetails": [x[1] for x in sorted(sub_features.get((p, s), []), key=lambda x: x[0])]
            })

        # STEP B: Robust Parity Alignment (Residual Absorption)
        # 1. Calculate the raw sum and clipped score
        total_impact_points = round(sum(v for p, v in pillar_totals.items() if p != "Metadata"), 1)
        final_score = round(np.clip(50.0 + total_impact_points, 1.0, 99.0), 1)

        # 2. Calculate the residual (the difference created by clipping)
        # Residual = (Clipped_Score - 50) - Raw_Sum
        residual = round((final_score - 50.0) - total_impact_points, 1)

        # 3. Absorb residual into the anchor pillar ("Therapeutic Context") and subcategory
        anchor_pillar = "Therapeutic Context"
        anchor_subcat = "Therapeutic Area Profile"
        
        if residual != 0:
            if anchor_pillar in pillar_totals:
                pillar_totals[anchor_pillar] = round(pillar_totals[anchor_pillar] + residual, 1)
            
            # Also update the leaf node in final_subcats for Treemap parity
            for sub in final_subcats:
                if sub["Pillar"] == anchor_pillar and sub["Subcategory"] == anchor_subcat:
                    sub["Impact"] = round(sub["Impact"] + residual, 1)
                    if sub["Impact"] == -0.0: sub["Impact"] = 0.0
                    break

        # Final cleanup for -0.0 across all pillars
        for p in pillar_totals:
            if pillar_totals[p] == -0.0: pillar_totals[p] = 0.0
        
        return {
            "score": final_score,
            "threshold": 50.0,
            "pillar_impacts": [
                {"Pillar": p, "Impact": v} 
                for p, v in pillar_totals.items() if p != "Metadata"
            ],
            "subcat_impacts": final_subcats,
            "mode": mode
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}
