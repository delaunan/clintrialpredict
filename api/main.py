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
        app.state.taxonomy = json.load(f)
    
    # 1. Prepare Feature Metadata
    prep = app.state.model.named_steps['prep']
    app.state.feature_names = prep.get_feature_names_out()
    
    # 2. Initialize Real-Time SHAP Explainer (PARKED FOR v01)
    # print("Initializing TreeExplainer for real-time simulation...")
    # clf = app.state.model.named_steps['clf']
    # app.state.explainer = shap.TreeExplainer(clf)
    
    # 3. Map features to Taxonomy
    app.state.feature_to_pillar = {}
    for pillar, subcats in app.state.taxonomy.items():
        for subcat, info in subcats.items():
            for feat_prefix in info['features']:
                if feat_prefix == "CALIBRATION_OFFSET":
                    continue
                for i, full_name in enumerate(app.state.feature_names):
                    if full_name.startswith(feat_prefix):
                        app.state.feature_to_pillar[i] = {
                            "pillar": pillar,
                            "subcategory": subcat
                        }
    print(f"Mapped {len(app.state.feature_to_pillar)} features to taxonomy.")

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
        threshold_logit = ta_threshold_logits.get(ta, app.state.thresholds.get("global_threshold_logit", 0.0511))
        gain_factor = app.state.thresholds.get("gain_factor", 25.0)
        intercept = app.state.thresholds.get("base_value", 0.0)
        
        # 2. AUDIT-ONLY Logic (Simulation parked for v01)
        if nct_id in app.state.shap_dict:
            # AUDIT: Instant lookup from pre-calculated matrix
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
        raw_score = 50 + (threshold_logit - pred_logit) * gain_factor
        final_score = float(np.clip(raw_score, 1, 99))
        
        # 4. Aggregation by Taxonomy
        pillar_impacts = {p: 0.0 for p in app.state.taxonomy.keys()}
        sub_sums = {} # (pillar, subcat) -> impact
        
        for i, val in enumerate(shap_vals):
            mapping = app.state.feature_to_pillar.get(i)
            if mapping:
                p, s = mapping['pillar'], mapping['subcategory']
                score_impact = -float(val) * gain_factor # Logit-to-Score unit conversion
                
                pillar_impacts[p] += score_impact
                sub_sums[(p, s)] = sub_sums.get((p, s), 0.0) + score_impact

        # 5. Inject Calibration Offset into Pillar 1 (Therapeutic Context)
        calibration_offset_pts = (threshold_logit - intercept) * gain_factor
        p1 = "1. Therapeutic Context"
        s1 = "Indication Risk Profile"
        
        pillar_impacts[p1] += calibration_offset_pts
        sub_sums[(p1, s1)] = sub_sums.get((p1, s1), 0.0) + calibration_offset_pts
        
        # 6. Format Final Response
        subcat_impacts = []
        for (p, s), imp in sub_sums.items():
            meta = app.state.taxonomy.get(p, {}).get(s, {})
            subcat_impacts.append({
                "Pillar": p,
                "Subcategory": s,
                "Impact": round(imp, 2),
                "Narrative": meta.get('pos_impact' if imp >= 0 else 'neg_impact', "")
            })
            
        return {
            "score": round(final_score, 1),
            "threshold": 50.0,
            "pillar_impacts": [{"Pillar": p, "Impact": round(v, 2)} for p, v in pillar_impacts.items()],
            "subcat_impacts": subcat_impacts,
            "mode": mode,
            "calibration_offset": round(calibration_offset_pts, 2)
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}
