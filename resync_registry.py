import joblib
import pandas as pd
import numpy as np
import json
import re
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path.cwd()
MODEL_PATH = BASE_DIR / "models" / "model_prod_01.joblib"
SHAP_PATH = BASE_DIR / "models" / "shap_values_01.joblib"
THRESHOLDS_PATH = BASE_DIR / "models" / "thresholds_01.json"
TAXONOMY_PATH = BASE_DIR / "models" / "taxonomy_01.json"
REGISTRY_PATH = BASE_DIR / "frontend" / "data" / "search_registry.csv"

def resync_registry_scores():
    print(">>> Resyncing Registry Scores with API Rounding Logic...")
    
    # 1. Load Artifacts
    model = joblib.load(MODEL_PATH)
    shap_dict = joblib.load(SHAP_PATH)
    with open(THRESHOLDS_PATH, 'r') as f: thresholds = json.load(f)
    with open(TAXONOMY_PATH, 'r') as f: taxonomy_payload = json.load(f)
    
    registry_meta = taxonomy_payload.get("FIELDS", taxonomy_payload.get("FEATURE_REGISTRY", taxonomy_payload))
    
    prep = model.named_steps['prep']
    feature_names = prep.get_feature_names_out()
    feat_to_idx = {name: i for i, name in enumerate(feature_names)}
    
    # Constants
    gain_factor = thresholds.get("gain_factor", 25.0)
    intercept = thresholds.get("base_value", 0.0)
    ta_threshold_logits = thresholds.get("ta_threshold_logits", {})
    global_threshold_logit = thresholds.get("global_threshold_logit", 0.0)
    
    DISABLED_COLS = [
        'includes_us_ml', 'is_fda_regulated_drug_ml', 'gbd_cause_id_ml',
        'gbd_cause_id_2_ml', 'gbd_cause_id_4_ml', 'gbd_hierarchy_level_ml',
        'is_duration_unknown_ml', 'target',  'masking_ml',
        'therapeutic_area_ml', 'strategic_ambition_ml', 'intervention_model_ml'
    ]
    
    # Identify pillars
    pillars = set()
    for f_name, f_meta in registry_meta.items():
        p = f_meta.get("ui", {}).get("pillar")
        if p and p != "Metadata": pillars.add(p)

    # 2. Load Registry
    df = pd.read_csv(REGISTRY_PATH)
    total_trials = len(df)
    
    new_scores = []
    
    for idx, row in df.iterrows():
        nct_id = row['nct_id']
        ta = row['therapeutic_area']
        
        if nct_id not in shap_dict:
            new_scores.append(row['Clinical_Score'])
            continue
            
        shap_vals = shap_dict[nct_id]
        threshold_logit = ta_threshold_logits.get(ta, global_threshold_logit)
        
        # AGGREGATION LOGIC (Mirroring api/main.py exactly)
        sub_sums_raw = {}
        mapped_indices = set()
        
        for feat_name, feat_meta in registry_meta.items():
            ui = feat_meta.get("ui", {})
            p = ui.get("pillar")
            s = ui.get("subgroup")
            if not p or not s or p == "Metadata": continue
            
            impact = 0.0
            prefix = ""
            if feat_name not in DISABLED_COLS:
                enc = feat_meta.get("encoding")
                if enc == "ordinal": prefix = "ordinal__"
                elif enc == "target": prefix = "target__"
                elif enc == "numeric":
                    if "arms" in feat_name: prefix = "num_arms__"
                    elif "duration" in feat_name: prefix = "num_duration__"
            
            prefixed_feat = f"{prefix}{feat_name}"
            for full_name in feature_names:
                if full_name == prefixed_feat or full_name.startswith(f"{prefixed_feat}_"):
                    i = feat_to_idx[full_name]
                    impact += -float(shap_vals[i]) * gain_factor
                    mapped_indices.add(i)
            
            key = (p, s)
            sub_sums_raw[key] = sub_sums_raw.get(key, 0.0) + impact

        # Unmapped signal
        unmapped_indices = set(range(len(shap_vals))) - mapped_indices
        unmapped_impact = sum(-float(shap_vals[i]) * gain_factor for i in unmapped_indices)
        sub_sums_raw[("Therapeutic Context", "Other Model Signals")] = sub_sums_raw.get(("Therapeutic Context", "Other Model Signals"), 0.0) + unmapped_impact
        
        # Calibration Offset
        calibration_offset_pts = (threshold_logit - intercept) * gain_factor
        # Target CP/CS for calibration (usually Therapeutic Area Profile)
        sub_sums_raw[("Therapeutic Context", "Therapeutic Area Profile")] = sub_sums_raw.get(("Therapeutic Context", "Therapeutic Area Profile"), 0.0) + calibration_offset_pts
        
        # BOTTOM-UP ROUNDING
        pillar_totals = {p: 0.0 for p in pillars}
        for (p, s), raw_imp in sub_sums_raw.items():
            pillar_totals[p] += round(raw_imp, 1)
            
        final_score = 50.0 + sum(pillar_totals.values())
        new_scores.append(round(final_score, 1))
        
        if idx % 500 == 0:
            print(f"   Processed {idx}/{total_trials}...")

    df['Clinical_Score'] = new_scores
    
    # Update Zone labels if necessary
    df['Zone'] = pd.cut(df['Clinical_Score'], [0, 25, 50, 75, 100], labels=["High Risk", "Watchlist", "Favorable", "Low Risk"])
    
    # 3. Save
    df.to_csv(REGISTRY_PATH, index=False)
    print(">>> Registry Resync Complete. Absolute parity restored.")

if __name__ == "__main__":
    resync_registry_scores()
