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
MIN_YEAR = 2019

def create_search_label(row):
    acro = str(row['acronym']).strip() if pd.notna(row['acronym']) else ""
    drug = str(row['alpha_drug_name']).strip() if pd.notna(row['alpha_drug_name']) else "Unknown Drug"
    sponsor = str(row['lead_sponsor_canonical']).strip() if pd.notna(row['lead_sponsor_canonical']) else "Unknown Sponsor"
    ta_val = row.get('therapeutic_area_ui', row.get('therapeutic_area', 'Unclassified'))
    ta = str(ta_val).strip()
    year = str(int(row['start_year'])) if pd.notna(row['start_year']) else "YYYY"
    prefix = f"[{acro}] " if acro and acro.lower() != 'nan' else ""
    return f"{prefix}{drug} ({sponsor}) | {ta} ({year})"

def refresh_registry():
    print(">>> Refreshing Search Registry (Top 50 + Parity Scores)...")
    
    df_full = pd.read_csv(DATA_CLINPRED_PATH, low_memory=False)
    df_filtered = df_full[(df_full['lead_sponsor_canonical'].isin(BIG_PHARMA)) & (df_full['start_year'] > MIN_YEAR)].copy()
    df_filtered['ui_search_label'] = df_filtered.apply(create_search_label, axis=1)
    
    model = joblib.load(MODEL_PATH)
    feature_names = model.named_steps['prep'].get_feature_names_out()
    shap_dict = joblib.load(SHAP_PATH)
    with open(THRESHOLDS_PATH, 'r') as f: thresholds = json.load(f)
    with open(TAXONOMY_PATH, 'r') as f: taxonomy_payload = json.load(f)
    
    registry_meta = taxonomy_payload.get("FIELDS", taxonomy_payload.get("FEATURE_REGISTRY", taxonomy_payload))
    gain_factor = float(thresholds.get("gain_factor", 25.0))
    intercept = float(thresholds.get("base_value", 0.0))
    ta_threshold_logits = thresholds.get("ta_threshold_logits", {})
    global_threshold_logit = float(thresholds.get("global_threshold_logit", 0.0))
    
    DISABLED_COLS = ['includes_us_ml', 'is_fda_regulated_drug_ml', 'gbd_cause_id_ml', 'gbd_cause_id_2_ml', 'gbd_cause_id_4_ml', 'gbd_hierarchy_level_ml', 'is_duration_unknown_ml', 'target', 'masking_ml', 'therapeutic_area_ml', 'strategic_ambition_ml', 'intervention_model_ml']
    
    pillars = set()
    for f_name, f_meta in registry_meta.items():
        p = f_meta.get("ui", {}).get("pillar")
        if p and p != "Metadata": pillars.add(p)

    new_scores = []
    print(f"    Calculating Parity Scores for {len(df_filtered)} trials...")
    
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

    for idx, row in df_filtered.iterrows():
        nct_id = str(row['nct_id']); ta = row['therapeutic_area']
        if nct_id not in shap_dict:
            new_scores.append(30.0); continue
            
        shap_vals = shap_dict[nct_id]
        threshold_logit = float(ta_threshold_logits.get(ta, global_threshold_logit))
        sub_sums_raw = {}
        
        for i, (p, s) in feat_to_subcat.items():
            sub_sums_raw[(p, s)] = sub_sums_raw.get((p, s), 0.0) + (-float(shap_vals[i]) * gain_factor)

        unmapped_indices = set(range(len(shap_vals))) - mapped_indices
        unmapped_impact = sum(-float(shap_vals[i]) * gain_factor for i in unmapped_indices)
        sub_sums_raw[("Therapeutic Context", "Other Model Signals")] = sub_sums_raw.get(("Therapeutic Context", "Other Model Signals"), 0.0) + unmapped_impact
        sub_sums_raw[("Therapeutic Context", "Therapeutic Area Profile")] = sub_sums_raw.get(("Therapeutic Context", "Therapeutic Area Profile"), 0.0) + (threshold_logit - intercept) * gain_factor
        
        p_totals = {p: 0.0 for p in pillars}
        for (pk, sk), val in sub_sums_raw.items():
            p_totals[pk] += round(val, 1)
            
        new_scores.append(round(50.0 + sum(p_totals.values()), 1))

    df_filtered['Clinical_Score'] = new_scores
    df_filtered['Zone'] = pd.cut(df_filtered['Clinical_Score'], [0, 25, 50, 75, 100], labels=["High Risk", "Watchlist", "Favorable", "Low Risk"])
    df_filtered.to_csv(REGISTRY_PATH, index=False, quoting=csv.QUOTE_ALL)
    print(f">>> Registry Refreshed: {len(df_filtered):,} trials saved to {REGISTRY_PATH}")

if __name__ == "__main__":
    refresh_registry()
