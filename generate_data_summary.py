import pandas as pd
import numpy as np
import sys
import os

# Add src to path to import PIPELINE_REGISTRY
sys.path.append('src')
from prep.pipeline import PIPELINE_REGISTRY

def update_summary():
    print(">>> Generating Exhaustive Data Summary (2,455 Fields)...")
    
    # Load data
    df = pd.read_csv('data/data_clinpred.csv', low_memory=False)
    all_cols = df.columns.tolist()
    
    fields_registry = PIPELINE_REGISTRY["FIELDS"]
    summary_rows = []
    
    # Define XGBoost active columns based on pipeline.py logic
    DISABLED_COLS = [
        'includes_us_ml', 'is_fda_regulated_drug_ml', 'gbd_cause_id_ml',
        'gbd_cause_id_2_ml', 'gbd_cause_id_4_ml', 'gbd_hierarchy_level_ml',
        'is_duration_unknown_ml', 'target_ml'
    ]
    
    print(f"    [Audit] Processing {len(all_cols)} columns...")

    for col in all_cols:
        # 1. Defaults
        pillar = "Metadata / Raw"
        subgroup = "Other"
        label = col
        encoding = "PASSTHROUGH"
        dist_info = ""
        in_xgboost = ""
        
        # 2. Registry Override (if exists)
        reg_key = col if col in fields_registry else (f"{col}_ml" if f"{col}_ml" in fields_registry else None)
        
        if reg_key:
            meta = fields_registry[reg_key]
            ui_meta = meta.get('ui', {})
            label = ui_meta.get('label', col)
            pillar = ui_meta.get('pillar', 'Model Feature')
            subgroup = ui_meta.get('subgroup', 'Registry Field')
            enc_val = meta.get('encoding')
            encoding = str(enc_val).upper()
            
            # XGBoost logic check
            if reg_key not in DISABLED_COLS:
                if enc_val in ['ordinal', 'target'] or reg_key in ['number_of_arms_ml', 'primary_duration_months_ml']:
                    in_xgboost = "x"
        
        # 3. Pattern-Based Categorization (for fields not in registry)
        elif col.startswith('crit_'):
            pillar = "NLP / Embeddings"
            subgroup = "Eligibility Criteria (BioBERT)"
            encoding = "NUMERIC (768d)"
        elif col.startswith('sci_'):
            pillar = "NLP / Embeddings"
            subgroup = "Scientific Essence (BioBERT)"
            encoding = "NUMERIC (768d)"
        elif col.startswith('endp_'):
            pillar = "NLP / Embeddings"
            subgroup = "Primary Endpoints (BioBERT)"
            encoding = "NUMERIC (768d)"
        elif any(x in col for x in ['daly_', 'yld_', 'yll_', 'market_skew', 'chronic_ratio']):
            pillar = "Epidemiology (IHME)"
            subgroup = "Disease Burden Metrics"
            encoding = "NUMERIC (Joined)"
        elif col.startswith('ui_') or col.startswith('txt_'):
            pillar = "UI Display / Text"
            subgroup = "Sanitized Content"
            encoding = "TEXT"
        elif col in ['nct_id', 'brief_title', 'official_title', 'lead_sponsor', 'start_date']:
            pillar = "Identity / Temporal"
            subgroup = "Core AACT Metadata"
            encoding = "ID/STRING"
        elif col in ['includes_us', 'raw_geographic_footprint', 'number_of_facilities']:
            pillar = "Execution Framework"
            subgroup = "Operations & Geography"
            encoding = "RAW/OPERATIONAL"
            
        # 4. Distribution Calculation
        if df[col].nunique() < 20 or (reg_key and fields_registry[reg_key].get('encoding') in ['ordinal', 'target']):
            counts = df[col].value_counts(dropna=False, normalize=True).head(5) * 100
            dist_info = " | ".join([f"{cat}: {pct:.1f}%" for cat, pct in counts.items()])
        elif pd.api.types.is_numeric_dtype(df[col]):
            vals = df[col].dropna()
            dist_info = f"Range: [{vals.min():.2f} to {vals.max():.2f}] | Mean: {vals.mean():.2f}" if not vals.empty else "EMPTY"
        else:
            dist_info = f"Text Strings ({df[col].nunique()} unique)"

        summary_rows.append({
            "In XGBoost": in_xgboost,
            "Pillar": pillar,
            "Subgroup": subgroup,
            "UI Label": label,
            "Column Name": col,
            "Encoding Strategy": encoding,
            "Distribution / Stats": dist_info,
            "Missing %": f"{(df[col].isna().sum() / len(df) * 100):.2f}%"
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.sort_values(["In XGBoost", "Pillar", "Subgroup", "Column Name"], ascending=[False, True, True, True])
    
    output_path = 'data_clinpred_summary.xlsx'
    df_summary.to_excel(output_path, index=False)
    print(f"> SUCCESS: Updated {output_path} with XGBoost markings.")

if __name__ == "__main__":
    update_summary()
