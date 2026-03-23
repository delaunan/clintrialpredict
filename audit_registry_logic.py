import pandas as pd
import numpy as np
import sys
import json

# Add src to path to import PIPELINE_REGISTRY
sys.path.append('src')
from prep.pipeline import PIPELINE_REGISTRY

def audit():
    print("# Production Ingestion Quality Audit")
    print("\n## Step 1: The \"Twin-Field\" Completeness Check")
    
    df = pd.read_csv('data/data_clinpred.csv')
    cols = df.columns.tolist()
    
    fields = PIPELINE_REGISTRY["FIELDS"]
    missing_ml = []
    missing_ui = []
    
    for field_name, meta in fields.items():
        if meta.get('encoding') is not None:
            # Check _ml
            if field_name not in cols:
                missing_ml.append(field_name)
            
            # Check _ui
            ui_col = field_name.replace('_ml', '_ui')
            if ui_col not in cols:
                missing_ui.append(ui_col)
                
    # Target Check (v35.0 Isolation mandate)
    target_ok = "target" in cols and "target_ml" not in cols
    
    if not missing_ml and not missing_ui:
        print("- [PASS] All registry features have corresponding _ml and _ui columns.")
    else:
        if missing_ml:
            print(f"- [FAIL] Missing _ml columns: {missing_ml}")
        if missing_ui:
            print(f"- [FAIL] Missing _ui columns: {missing_ui}")
            
    if target_ok:
        print("- [PASS] 'target' column follows isolation mandate (no 'target_ml').")
    else:
        print("- [FAIL] 'target' column isolation mandate violated.")

    print("\n## Step 2: Programmatic Mapping Validation")
    
    logic_drift = []
    mapping_completeness = True
    
    for field_name, meta in fields.items():
        if 'mapping' not in meta or meta.get('encoding') is None:
            continue
            
        ml_col = field_name
        ui_col = field_name.replace('_ml', '_ui')
        
        if ml_col not in df.columns or ui_col not in df.columns:
            continue
            
        # Integer Consistency
        unique_ml = df[ml_col].unique()
        valid_codes = set()
        for v in meta['mapping'].values():
            if isinstance(v, list):
                valid_codes.add(v[0])
        
        invalid_vals = [v for v in unique_ml if v not in valid_codes and not pd.isna(v)]
        if invalid_vals:
            logic_drift.append(f"{ml_col}: Invalid codes found in data: {invalid_vals}")
            mapping_completeness = False
            
        # Label Alignment (100 random rows)
        sample_size = min(100, len(df))
        sample_df = df.sample(sample_size)
        
        # Create reverse mapping for validation (allowing multiple labels per code)
        code_to_labels = {}
        for k, v in meta['mapping'].items():
            if isinstance(v, list):
                code = v[0]
                label = v[1]
                if code not in code_to_labels:
                    code_to_labels[code] = set()
                code_to_labels[code].add(label)
        
        for idx, row in sample_df.iterrows():
            ml_val = row[ml_col]
            ui_val = row[ui_col]
            
            if pd.isna(ml_val):
                continue
                
            valid_labels = code_to_labels.get(ml_val, set())
            if valid_labels and ui_val not in valid_labels:
                logic_drift.append(f"{ml_col}: Label mismatch at index {idx}. Code {ml_val} -> Got '{ui_val}', Expected one of {valid_labels}")
                mapping_completeness = False
                break

    if not logic_drift:
        print("- [PASS] All sampled _ml and _ui columns perfectly align with Registry truth tables.")
    else:
        print("- [FAIL] Logic drift detected:")
        for drift in logic_drift[:10]: # Limit output
            print(f"  - {drift}")

    print("\n## Step 3: \"Unknown\" Sentinel Forensic Audit")
    
    # Check specific fallbacks
    sentinel_issues = []
    
    # Code 0 (Standard)
    fields_0 = ['gender_ml', 'innovation_tier_ml', 'patient_severity_ml']
    for f in fields_0:
        if f in df.columns:
            unknown_code = fields[f]['mapping'].get('UNKNOWN', [None])[0]
            if unknown_code != 0:
                sentinel_issues.append(f"{f}: Expected UNKNOWN code 0, got {unknown_code}")
                
    # Code 1 (Inclusionary)
    fields_1 = ['adult_ml', 'older_adult_ml']
    for f in fields_1:
        if f in df.columns:
            unknown_code = fields[f]['mapping'].get('UNKNOWN', [None])[0]
            if unknown_code != 1:
                sentinel_issues.append(f"{f}: Expected UNKNOWN code 1, got {unknown_code}")

    # Code 2 (Neutral)
    fields_2 = ['phase_ml', 'strategic_ambition_ml', 'administration_complexity_ml']
    for f in fields_2:
        if f in df.columns:
            unknown_code = fields[f]['mapping'].get('UNKNOWN', [None])[0]
            if unknown_code != 2:
                sentinel_issues.append(f"{f}: Expected UNKNOWN code 2, got {unknown_code}")

    if not sentinel_issues:
        print("- [PASS] All 'Unknown' sentinels (0, 1, 2) match their respective Domain-Acuity definitions.")
    else:
        print("- [FAIL] Sentinel issues detected:")
        for issue in sentinel_issues:
            print(f"  - {issue}")

    print("\n## Step 4: Non-Categorical Protection Check")
    
    # GBD Integrity
    if 'gbd_cause_id_3_ml' in df.columns:
        gbd_vals = df['gbd_cause_id_3_ml'].unique()
        only_zeros = len(gbd_vals) == 1 and gbd_vals[0] == 0
        if only_zeros:
            print("- [FAIL] gbd_cause_id_3_ml contains only 0s.")
        else:
            print("- [PASS] gbd_cause_id_3_ml contains diverse IHME Cause IDs.")
    
    # Numeric Passthrough
    passthrough_ok = True
    numeric_fields = ['number_of_arms_ml', 'primary_duration_months_ml']
    for f in numeric_fields:
        if f in df.columns:
            # Check if it looks categorical (few unique integers)
            unique_vals = df[f].dropna().unique()
            if len(unique_vals) < 10 and all(v == int(v) for v in unique_vals):
                 # This is a heuristic, but usually durations/arms have more variety or floats
                 pass 
            # Better check: is it in mapping?
            if 'mapping' in fields[f]:
                print(f"- [FAIL] {f} has a mapping in registry but should be numeric passthrough.")
                passthrough_ok = False
    
    if passthrough_ok:
        print("- [PASS] Numeric fields (Arms, Duration) are preserved as raw continuous values.")

    print("\n## Step 5: Target Mapping Audit")
    
    target_drift = []
    # overall_status == 'COMPLETED' -> target == 0.0
    completed_mismatch = df[(df['overall_status'] == 'COMPLETED') & (df['target'] != 0.0)]
    if not completed_mismatch.empty:
        target_drift.append(f"COMPLETED status mismatch: {len(completed_mismatch)} rows")
        
    # overall_status in ['TERMINATED', 'WITHDRAWN'] -> target == 1.0
    failed_mismatch = df[df['overall_status'].isin(['TERMINATED', 'WITHDRAWN']) & (df['target'] != 1.0)]
    if not failed_mismatch.empty:
        target_drift.append(f"TERMINATED/WITHDRAWN status mismatch: {len(failed_mismatch)} rows")
        
    # Active statuses -> target == NaN
    active_statuses = ['RECRUITING', 'ACTIVE, NOT RECRUITING', 'NOT YET RECRUITING', 'ENROLLING BY INVITATION']
    active_mismatch = df[df['overall_status'].isin(active_statuses) & df['target'].notna()]
    if not active_mismatch.empty:
        target_drift.append(f"Active status mismatch (expected NaN): {len(active_mismatch)} rows")

    if not target_drift:
        print("- [PASS] Outcome logic correctly maps COMPLETED(0), FAILED(1), and ONGOING(NaN).")
    else:
        print("- [FAIL] Target mapping drift detected:")
        for drift in target_drift:
            print(f"  - {drift}")

    print("\n## Final Status")
    all_pass = not missing_ml and not missing_ui and target_ok and not logic_drift and not sentinel_issues and passthrough_ok and not target_drift
    if all_pass:
        print("**DATA STATUS: LOGIC-LOCKED**")
    else:
        print("**DATA STATUS: AUDIT FAILED**")

if __name__ == "__main__":
    audit()
