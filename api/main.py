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
    
    # 2. Reconstruct Taxonomy Mapping (Logic-Locked to Registry)
    # This maps feature indices (i) to {pillar, subcategory, narrative_info}
    app.state.feature_to_taxonomy = {}
    app.state.pillars = set()
    app.state.subcategories = {} # pillar -> set of subcategories
    
    # Also identify where the calibration offset should be applied
    app.state.calibration_target = {"pillar": "Therapeutic Context", "subcategory": "Indication Risk Profile"}

    for feat_name, meta in app.state.registry.items():
        pillar = meta.get("pillar")
        subgroup = meta.get("subgroup")
        if not pillar or not subgroup:
            continue
            
        app.state.pillars.add(pillar)
        if pillar not in app.state.subcategories:
            app.state.subcategories[pillar] = set()
        app.state.subcategories[pillar].add(subgroup)

        # Determine prefix based on encoding (replicating notebook logic)
        prefix = ""
        enc = meta.get("encoding")
        if enc == "ordinal": prefix = "ordinal__"
        elif enc == "target": prefix = "target__"
        elif enc == "numeric":
            if "arms" in feat_name: prefix = "num_arms__"
            elif "duration" in feat_name: prefix = "num_duration__"
            
        prefixed_feat = f"{prefix}{feat_name}"
        
        # Check if this is the calibration anchor
        if feat_name == "therapeutic_area_ml":
            app.state.calibration_target = {"pillar": pillar, "subcategory": subgroup}

        # Map to actual indices in model output
        for i, full_name in enumerate(app.state.feature_names):
            if full_name == prefixed_feat or full_name.startswith(f"{prefixed_feat}_"):
                app.state.feature_to_taxonomy[i] = {
                    "pillar": pillar,
                    "subcategory": subgroup,
                    "pos_impact": meta.get("pos_impact", ""),
                    "neg_impact": meta.get("neg_impact", "")
                }

    print(f"Mapped {len(app.state.feature_to_taxonomy)} features to taxonomy.")

@app.get("/")
def root():
    return {"status": "Clinical Trial Predictor API v01 Online (Audit Only)"}

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        nct_id = data.get("nct_id")
        ta = data.get("therapeutic_area", "Unclassified")
        
        # 1. Calibration Constants (Synced with Notebook)
        ta_threshold_logits = app.state.thresholds.get("ta_threshold_logits", {})
        threshold_logit = ta_threshold_logits.get(ta, app.state.thresholds.get("global_threshold_logit", 0.0))
        gain_factor = app.state.thresholds.get("gain_factor", 25.0)
        intercept = app.state.thresholds.get("base_value", 0.0)
        
        # 2. AUDIT-ONLY Logic
        if nct_id in app.state.shap_dict:
            shap_vals = app.state.shap_dict[nct_id]
            pred_logit = np.sum(shap_vals) + intercept
            mode = "audit"
        else:
            return {
                "error": f"Trial ID {nct_id} not found in the production registry.",
                "status": "simulation_parked",
                "message": "Real-time simulation is disabled in the current production release (v01)."
            }
        
        # 3. Success Scoring (0-100)
        # score = 50 + (threshold_logit - pred_logit) * gain_factor
        raw_score = 50 + (threshold_logit - pred_logit) * gain_factor
        final_score = float(np.clip(raw_score, 1, 99))
        
        # 4. Aggregation by Taxonomy
        pillar_impacts = {p: 0.0 for p in app.state.pillars}
        sub_sums = {} # (pillar, subcategory) -> impact
        
        for i, val in enumerate(shap_vals):
            mapping = app.state.feature_to_taxonomy.get(i)
            if mapping:
                p, s = mapping['pillar'], mapping['subcategory']
                score_impact = -float(val) * gain_factor # Logit-to-Score unit conversion
                
                pillar_impacts[p] += score_impact
                sub_sums[(p, s)] = sub_sums.get((p, s), 0.0) + score_impact

        # 5. Inject Calibration Offset into Target Pillar/Subcategory
        calibration_offset_pts = (threshold_logit - intercept) * gain_factor
        cp = app.state.calibration_target["pillar"]
        cs = app.state.calibration_target["subcategory"]
        
        print(f"DEBUG: nct_id={nct_id}, ta={ta}")
        print(f"DEBUG: calibration_offset_pts={calibration_offset_pts}")
        print(f"DEBUG: target_pillar={cp}, target_subcategory={cs}")
        
        pillar_impacts[cp] = pillar_impacts.get(cp, 0.0) + calibration_offset_pts
        sub_sums[(cp, cs)] = sub_sums.get((cp, cs), 0.0) + calibration_offset_pts
        
        print(f"DEBUG: pillar_impacts after offset={pillar_impacts}")
        
        # 6. Format Final Response
        subcat_impacts = []
        for (p, s), imp in sub_sums.items():
            # Find metadata from registry via first feature found for this subcat
            # (Narrative is usually consistent across subcat features)
            narrative = ""
            for mapping in app.state.feature_to_taxonomy.values():
                if mapping['pillar'] == p and mapping['subcategory'] == s:
                    narrative = mapping['pos_impact' if imp >= 0 else 'neg_impact']
                    break
                    
            subcat_impacts.append({
                "Pillar": p,
                "Subcategory": s,
                "Impact": round(imp, 2),
                "Narrative": narrative
            })
            
        return {
            "score": round(final_score, 1),
            "threshold": 50.0,
            "pillar_impacts": [
                {"Pillar": p, "Impact": round(v, 2)} 
                for p, v in pillar_impacts.items() if p != "Metadata"
            ],
            "subcat_impacts": [item for item in subcat_impacts if item['Pillar'] != "Metadata"],
            "mode": mode,
            "calibration_offset": round(calibration_offset_pts, 2)
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}
