import pandas as pd
import numpy as np
from pathlib import Path

MASTER_PATH = Path("data/data_clinpred.csv")
REGISTRY_PATH = Path("frontend/data/search_registry.csv")

def prune_and_enhance():
    print(f"> Reading master: {MASTER_PATH}")
    df_master = pd.read_csv(MASTER_PATH, low_memory=False)
    
    # 1. Define final columns to keep
    cols_to_keep = [
        "nct_id", "ui_search_label", "alpha_drug_name", "lead_sponsor_canonical", 
        "therapeutic_area", "gbd_indication_name", "start_year", "phase", 
        "Clinical_Score", "overall_status", "trial_segment", "target",
        "enrollment", "number_of_arms", "primary_purpose", "lead_sponsor", 
        "sponsor_tier", "has_dmc", "number_of_facilities",
        # --- NEW: Prose UI Fields ---
        "ui_brief_title", "ui_summary", "ui_criteria"
    ]
    
    # Ensure is_correct is calculated based on master data
    def check_accuracy(row):
        if pd.isna(row.get('target')): return None
        target = row['target']
        score = row.get('Clinical_Score', 0)
        if score >= 50.0 and target == 0.0: return True
        if score < 50.0 and target == 1.0: return True
        return False

    # Filter columns and add accuracy logic
    existing_cols = [c for c in cols_to_keep if c in df_master.columns]
    df_lean = df_master[existing_cols].copy()
    df_lean['is_correct'] = df_lean.apply(check_accuracy, axis=1)
    
    # 2. Re-save
    df_lean.to_csv(REGISTRY_PATH, index=False)
    
    final_size = REGISTRY_PATH.stat().st_size / (1024 * 1024)
    print(f"✅ Registry Pruned with FULL UI Prose!")
    print(f"   - Final Size: {final_size:.2f} MB")
    print(f"   - Functionality: All Expanders (Summary/Criteria) are now populated.")

if __name__ == "__main__":
    prune_and_enhance()
