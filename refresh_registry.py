import joblib
import pandas as pd
import numpy as np
import json
import csv
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path.cwd()
DATA_CLINPRED_PATH = BASE_DIR / "data" / "data_clinpred.csv"
MODEL_PATH = BASE_DIR / "models" / "model_prod_01.joblib"
SHAP_PATH = BASE_DIR / "models" / "shap_values_01.joblib"
THRESHOLDS_PATH = BASE_DIR / "models" / "thresholds_01.json"
TAXONOMY_PATH = BASE_DIR / "models" / "taxonomy_01.json"
REGISTRY_PATH = BASE_DIR / "frontend" / "data" / "search_registry.csv"

# --- DYNAMIC PIPELINE IMPORT ---
from src.prep.pipeline import create_search_label, export_pipeline_taxonomy, PIPELINE_REGISTRY

BIG_PHARMA = [
    'ROCHE', 'GENENTECH (ROCHE)', 'CHUGAI (ROCHE)', 'SPARK (ROCHE)',
    'J&J', 'ACTELION (J&J)', 'CENTOCOR (J&J)', 'CRUCELL (J&J)', 'MOMENTA (J&J)',
    'PFIZER', 'SEAGEN', 'HOSPIRA (PFIZER)', 'WARNER-LAMBERT (PFIZER)',
    'AZN', 'ALEXION (AZN)', 'MEDIMMUNE (AZN)', 'PEARL THERAPEUTICS (AZN)', 'ALEXION',
    'NOVARTIS', 'ALCON (NOVARTIS)', 'SANDOZ (NOVARTIS)', 'MORPHOSYS (NOVARTIS)', 'MEDICINES COMPANY (NOVARTIS)',
    'MERCK (USA)', 'ORGANON (MERCK (USA))', 'CUBIST (MERCK (USA))', 'ACCELERON (MERCK (USA))',
    'GSK', 'TESARO (GSK)', 'SIERRA ONCOLOGY (GSK)', 'BELLUS HEALTH (GSK)',
    'SANOFI', 'GENZYME (SANOFI)', 'ABLYNX (SANOFI)', 'BIOVERATIV (SANOFI)', 'PROVENTION BIO (SANOFI)',
    'LILLY', 'AVID RADIOPHARMACEUTICALS (LILLY)', 'MORPHIC (LILLY)', 'DERMIRA (LILLY)',
    'BMS', 'CELGENE', 'CELGENE (BMS)', 'MYOKARDIA (BMS)', 'JUNO (BMS)', 'KARUNA (BMS)',
    'ABBVIE', 'ABBOTT (ABBVIE)', 'ALLERGAN', 'ALLERGAN (ABBVIE)', 'FOREST LABS (ABBVIE)', 'PHARMACYCLICS (ABBVIE)', 'IMMUNOGEN (ABBVIE)',
    'NOVO NORDISK', 'DICERNA (NOVO NORDISK)', 'AMGEN', 'KAI (AMGEN)', 'BIOVEX (AMGEN)',
    'TAKEDA', 'SHIRE (TAKEDA)', 'SHIRE', 'BAXALTA (TAKEDA)', 'MILLENNIUM (TAKEDA)',
    'GILEAD', 'KITE (GILEAD)', 'BAYER', 'BOEHRINGER INGELHEIM', 'TEVA', 'ACTAVIS (TEVA)', 'WATSON (TEVA)',
    'ASTELLAS', 'OSI PHARMA (ASTELLAS)', 'UCB', 'RA PHARMACEUTICALS (UCB)', 'ZOGENIX (UCB)',
    'BIOGEN', 'HI-BIO (BIOGEN)', 'DAIICHI SANKYO', 'VERTEX', 'ALPINE (VERTEX)', 'EISAI', 'OTSUKA', 'ASTEX (OTSUKA)',
    'SUMITOMO PHARMA', 'SUMITOMO', 'MERCK KGAA', 'KYOWA KIRIN', 'JAZZ PHARMACEUTICALS', 'IPSEN', 'ALBIREO (IPSEN)', 'EPIZYME (IPSEN)',
    'LUNDBECK', 'FERRING PHARMACEUTICALS', 'MITSUBISHI TANABE', 'CHIESI FARMACEUTICI', 'MODERNA', 'BIONTECH', 'REGENERON', 'GRIFOLS',
    'SUN PHARMACEUTICAL INDUSTRIES', 'SUN PHARMACEUTICAL', 'MALLINCKRODT', 'ALMIRALL', 'CELLTRION', 'INCYTE', 'NEUROCRINE BIOSCIENCES',
    'HENGRUI', 'BEIGENE', 'INNOVENT', 'AKESO', 'REMEGEN'
]
MIN_YEAR = 2017

def refresh_registry():
    print(">>> Refreshing Search Registry (Dynamic Registry-Based Generation)...")
    export_pipeline_taxonomy(TAXONOMY_PATH)
    
    df_full = pd.read_csv(DATA_CLINPRED_PATH, low_memory=False)
    if df_full.index.name == 'nct_id':
        df_full = df_full.reset_index()
    
    # 1. Prediction Probabilities
    model = joblib.load(MODEL_PATH)
    feature_names = model.named_steps['prep'].get_feature_names_out()
    
    cols_to_keep = [c for c in df_full.columns if c.endswith('_ml') or c == 'therapeutic_area']
    y_prob_full = model.predict_proba(df_full[cols_to_keep])[:, 1]
    df_full['Internal_Score_Raw'] = y_prob_full
    
    # 2. Hybrid Search Identity
    df_full['ui_search_label'] = df_full.apply(create_search_label, axis=1)
    
    # 3. Success Scores & Pillars
    shap_dict = joblib.load(SHAP_PATH)
    with open(THRESHOLDS_PATH, 'r') as f: thresholds = json.load(f)
    with open(TAXONOMY_PATH, 'r') as f: taxonomy_payload = json.load(f)
    
    registry_meta = taxonomy_payload.get("FIELDS", taxonomy_payload.get("FEATURE_REGISTRY", taxonomy_payload))
    gain_factor = float(thresholds.get("gain_factor", 25.0))
    intercept = float(thresholds.get("base_value", 0.0))
    ta_threshold_logits = thresholds.get("ta_threshold_logits", {})
    global_threshold_logit = float(thresholds.get("global_threshold_logit", 0.0))
    
    DISABLED_COLS = ['includes_us_ml', 'is_fda_regulated_drug_ml', 'gbd_cause_id_ml', 'gbd_cause_id_2_ml', 'gbd_cause_id_4_ml', 'gbd_hierarchy_level_ml', 'is_duration_unknown_ml', 'target', 'masking_ml', 'therapeutic_area_ml', 'strategic_ambition_ml', 'intervention_model_ml']
    
    pillars = ["Therapeutic Context", "Scientific Challenge", "Execution Framework", "Patient Profile"]

    # Pre-map features for speed
    feat_to_subcat = {}
    mapped_indices = set()
    for i, feat in enumerate(feature_names):
        for f_name, f_meta in registry_meta.items():
            ui = f_meta.get("ui", {})
            p, s = ui.get("pillar"), ui.get("subgroup")
            if not p or not s or p == "Metadata": continue
            
            prefix = ""
            if f_name not in DISABLED_COLS:
                enc = f_meta.get("encoding")
                if enc == "ordinal": prefix = "ordinal__"
                elif enc == "target": prefix = "target__"
                elif enc == "numeric":
                    if "arms" in f_name: prefix = "num_arms__"
                    elif "duration" in f_name: prefix = "num_duration__"
            
            prefixed_feat = f"{prefix}{f_name}"
            if feat == prefixed_feat or feat.startswith(f"{prefixed_feat}_"):
                feat_to_subcat[i] = (p, s)
                mapped_indices.add(i)

    print(f"    Calculating Success Scores for {len(df_full)} trials...")
    all_scores = []; all_zones = []
    pillar_scores = {p: [] for p in pillars}
    
    for idx, row in df_full.iterrows():
        nct_id = str(row['nct_id']); ta = row['therapeutic_area']
        if nct_id not in shap_dict:
            all_scores.append(30.0); all_zones.append("Watchlist")
            for p in pillars: pillar_scores[p].append(0.0)
            continue
            
        shap_vals = shap_dict[nct_id]
        threshold_logit = float(ta_threshold_logits.get(ta, global_threshold_logit))
        sub_sums_raw = {}
        
        # Aggregate raw impacts
        for i, (p, s) in feat_to_subcat.items():
            sub_sums_raw[(p, s)] = sub_sums_raw.get((p, s), 0.0) + (-float(shap_vals[i]) * gain_factor)

        # Unmapped Signals
        unmapped_indices = set(range(len(shap_vals))) - mapped_indices
        unmapped_impact = sum(-float(shap_vals[i]) * gain_factor for i in unmapped_indices)
        if unmapped_impact != 0:
            key = ("Therapeutic Context", "Other Model Signals")
            sub_sums_raw[key] = sub_sums_raw.get(key, 0.0) + unmapped_impact
        
        # Calibration Offset
        cal_key = ("Therapeutic Context", "Therapeutic Area Profile")
        sub_sums_raw[cal_key] = sub_sums_raw.get(cal_key, 0.0) + (threshold_logit - intercept) * gain_factor
        
        # STEP A: Round subcategories to 1 decimal point
        subcat_impacts_rounded = {k: round(v, 1) for k, v in sub_sums_raw.items()}
        for k, v in subcat_impacts_rounded.items():
            if v == -0.0: subcat_impacts_rounded[k] = 0.0
            
        # STEP B: Sum rounded subcategories to pillars
        p_totals = {p: 0.0 for p in pillars}
        for (pk, sk), val in subcat_impacts_rounded.items():
            if pk in p_totals:
                p_totals[pk] += val
        
        # ROBUST PARITY ALIGNMENT
        # 1. Calculate raw sum and clipped score
        total_impact_points = sum(p_totals.values())
        final_score = round(np.clip(50.0 + total_impact_points, 1.0, 99.0), 1)

        # 2. Calculate the residual
        residual = round((final_score - 50.0) - total_impact_points, 1)

        # 3. Absorb residual into anchor pillar
        anchor_pillar = "Therapeutic Context"
        if anchor_pillar in p_totals:
            p_totals[anchor_pillar] = round(p_totals[anchor_pillar] + residual, 1)
            if p_totals[anchor_pillar] == -0.0: p_totals[anchor_pillar] = 0.0
        
        # Final cleanup for all pillars
        for p in p_totals:
            p_totals[p] = round(p_totals[p], 1)
            if p_totals[p] == -0.0: p_totals[p] = 0.0

        all_scores.append(final_score)
        all_zones.append("High Risk" if final_score <= 25 else "Watchlist" if final_score <= 50 else "Favorable" if final_score <= 75 else "Low Risk")
        for p in pillars: pillar_scores[p].append(p_totals[p])

    df_full['Clinical_Score'] = all_scores
    df_full['Zone'] = all_zones
    for p in pillars: df_full[p] = pillar_scores[p]

    # 4. Backtesting Correctness
    def check_accuracy(row):
        if pd.isna(row.get('target')): return None
        target = row['target']
        score = row.get('Clinical_Score', 0)
        return (score >= 50.0 and target == 0.0) or (score < 50.0 and target == 1.0)
    
    df_full['is_correct'] = df_full.apply(check_accuracy, axis=1)

    # 6. DYNAMIC SELECTION
    registry_fields = list(PIPELINE_REGISTRY["FIELDS"].keys())
    calculated_artifacts = [
        'Clinical_Score', 'Zone', 'is_correct', 'ui_search_label', 'Internal_Score_Raw',
        'Therapeutic Context', 'Scientific Challenge', 'Execution Framework', 'Patient Profile',
        'therapeutic_context', 'therapeutic_area'
    ]
    ui_derived = [c for c in df_full.columns if c.endswith('_ui')]
    
    cols_to_export = list(dict.fromkeys(registry_fields + calculated_artifacts + ui_derived))
    existing_cols = [c for c in cols_to_export if c in df_full.columns]
    if REGISTRY_PATH.exists():
        current_header = list(pd.read_csv(REGISTRY_PATH, nrows=0).columns)
        if set(current_header) == set(existing_cols):
            existing_cols = current_header

    print(f">>> Applying Custom Filter for CSV: {len(BIG_PHARMA)} companies after {MIN_YEAR}...")
    df_filtered = df_full[
        (df_full['lead_sponsor_canonical'].isin(BIG_PHARMA)) &
        (df_full['start_year'] > MIN_YEAR)
    ][existing_cols].copy()

    if 'nct_id' in df_filtered.columns:
        df_filtered = df_filtered.set_index('nct_id')
    
    df_filtered.to_csv(REGISTRY_PATH, index=True, quoting=csv.QUOTE_ALL)
    print(f">>> Registry Refreshed: {len(df_filtered):,} trials saved to {REGISTRY_PATH}")

if __name__ == "__main__":
    refresh_registry()
