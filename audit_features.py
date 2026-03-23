import pandas as pd
import numpy as np
import json
import os

def run_audit():
    # Load data
    df = pd.read_csv('data/data_clinpred.csv', low_memory=False)
    
    # Load taxonomy
    with open('models/taxonomy_01.json', 'r') as f:
        taxonomy = json.load(f)
    
    fields = taxonomy.get("FIELDS", {})
    
    # --- Logic from preprocessing_clinpred.py ---
    DISABLED_COLS = [
        'includes_us_ml', 'is_fda_regulated_drug_ml', 'gbd_cause_id_ml', 
        'gbd_cause_id_2_ml', 'gbd_cause_id_4_ml', 'gbd_hierarchy_level_ml', 
        'is_duration_unknown_ml', 'target_ml'
    ]
    ACTIVE_NUMERIC = ['number_of_arms_ml', 'primary_duration_months_ml']
    
    def get_usage_role(name, encoding):
        if name in DISABLED_COLS:
            return "Encoded (Disabled)"
        if name in ACTIVE_NUMERIC:
            return "XGBoost Input"
        if encoding in ['ordinal', 'target']:
            return "XGBoost Input"
        if encoding == 'numeric' and name.endswith('_ml'):
            return "XGBoost Input"
        return "UI Metadata"

    # --- Filtering logic ---
    def should_include(name):
        if any(name.startswith(pre) for pre in ['sci_', 'endp_', 'crit_']): return False
        return True

    all_eligible = [f for f in fields.keys() if should_include(f)]
    
    # Priority: _ml fields first, then others. Sort each group by priority.
    ml_fields = [f for f in all_eligible if f.endswith('_ml')]
    other_fields = [f for f in all_eligible if not f.endswith('_ml')]
    
    ml_fields.sort(key=lambda x: fields[x].get('ui', {}).get('priority', 999))
    other_fields.sort(key=lambda x: fields[x].get('ui', {}).get('priority', 999))
    
    feature_names = ml_fields + other_fields

    audit_data = []

    fallback_labels = ["UNKNOWN", "Unclassified", "Line N/A", "Unknown Intent", "Unknown Precedent", "Unknown", "Other / Unclassified", "Not Specified", "Other / Unknown", "Other Modality / Unknown", "Unknown Tier", "Single Group / Not Specified", "Other Purpose / Unknown", "Static Design (Default)", "Not Specified / Unknown", "Single Goal (Default)", "Unknown Sponsor Tier", "Open Label / Not Specified", "Non-Randomized / Not Specified", "No Control / Not Specified", "Unknown Complexity", "Uncertain Severity", "Yes (Default)", "No (Default)", "Patients Only (Default)"]

    for feat in feature_names:
        meta = fields.get(feat, {})
        ui = meta.get('ui', {})
        mapping = meta.get('mapping', {})
        options = ui.get('options', [])
        encoding = meta.get('encoding', "passthrough" if not feat.endswith('_ml') else "N/A")
        
        role = get_usage_role(feat, encoding)
        pillar = ui.get('pillar', 'N/A')
        subgroup = ui.get('subgroup', 'N/A')
        ui_desc = ui.get('label', 'N/A')

        # 1. Metadata Fields (No options/mapping)
        if not feat.endswith('_ml') and not options and not mapping:
            count_vals = df[feat].dropna().nunique() if feat in df.columns else 0
            # Data Row
            audit_data.append({
                "Feature Name": feat,
                "UI Description": ui_desc,
                "Pillar": pillar,
                "Subgroup": subgroup,
                "Order": "",
                "Technical Value": "N/A",
                "Code": "",
                "Default/Fallback": "",
                "Count": df[feat].count() if feat in df.columns else 0,
                "Category": f"Metadata Field ({count_vals} unique values)",
                "Encoding Strategy": encoding,
                "Usage Role": role
            })
            # NaN Row
            audit_data.append({
                "Feature Name": feat,
                "UI Description": ui_desc,
                "Pillar": pillar,
                "Subgroup": subgroup,
                "Order": "",
                "Technical Value": "NaN",
                "Code": "",
                "Default/Fallback": "x",
                "Count": df[feat].isna().sum() if feat in df.columns else len(df),
                "Category": "Missing (NaN)",
                "Encoding Strategy": encoding,
                "Usage Role": role
            })
            continue

        # 2. Numeric Fields
        if encoding == 'numeric' or feat in ACTIVE_NUMERIC:
            vals = df[feat].dropna()
            dist_str = f"Numeric Distribution (Min: {vals.min()} / Max: {vals.max()} / Median: {vals.median()})" if not vals.empty else "Numeric Distribution (No Data)"
            audit_data.append({
                "Feature Name": feat,
                "UI Description": ui_desc,
                "Pillar": pillar,
                "Subgroup": subgroup,
                "Order": "",
                "Technical Value": "NUMERIC_RANGE",
                "Code": "",
                "Default/Fallback": "",
                "Count": len(vals),
                "Category": dist_str,
                "Encoding Strategy": "numeric",
                "Usage Role": role
            })
            audit_data.append({
                "Feature Name": feat,
                "UI Description": ui_desc,
                "Pillar": pillar,
                "Subgroup": subgroup,
                "Order": "",
                "Technical Value": "NaN",
                "Code": "",
                "Default/Fallback": "x",
                "Count": df[feat].isna().sum(),
                "Category": "Missing (NaN)",
                "Encoding Strategy": "numeric",
                "Usage Role": role
            })
            continue

        # 3. Target Encoding (GBD)
        if feat == 'gbd_cause_id_3_ml':
            counts_ml = df[feat].value_counts(dropna=False).sort_index()
            for val, count in counts_ml.items():
                is_nan = pd.isna(val)
                audit_data.append({
                    "Feature Name": feat,
                    "UI Description": ui_desc,
                    "Pillar": pillar,
                    "Subgroup": subgroup,
                    "Order": "",
                    "Technical Value": "NaN" if is_nan else str(val),
                    "Code": "" if is_nan else val,
                    "Default/Fallback": "x" if is_nan else "",
                    "Count": count,
                    "Category": "Missing (NaN)" if is_nan else f"ID: {int(val) if float(val).is_integer() else val}",
                    "Encoding Strategy": "target",
                    "Usage Role": role
                })
            continue

        # 4. Categorical Fields
        ui_feat = feat.replace('_ml', '_ui')
        counts_ser = df[ui_feat].value_counts() if ui_feat in df.columns else pd.Series()

        # Process Options
        for idx, (opt_val, opt_label) in enumerate(options):
            code = ""
            if opt_val in mapping: code = mapping[opt_val][0]
            elif str(opt_val) in mapping: code = mapping[str(opt_val)][0]
            
            count = counts_ser.get(opt_label, 0)
            is_fallback = "x" if (code == 0 or any(f.lower() in opt_label.lower() for f in fallback_labels)) else ""
            
            audit_data.append({
                "Feature Name": feat,
                "UI Description": ui_desc,
                "Pillar": pillar,
                "Subgroup": subgroup,
                "Order": idx,
                "Technical Value": opt_val,
                "Code": code,
                "Default/Fallback": is_fallback,
                "Count": count,
                "Category": opt_label,
                "Encoding Strategy": encoding,
                "Usage Role": role
            })

        # Mapping-only categories
        option_labels = {opt[1] for opt in options}
        for map_key, map_val in mapping.items():
            code, label = map_val[0], map_val[1]
            if label not in option_labels:
                count = counts_ser.get(label, 0)
                is_fallback = "x" if (code == 0 or any(f.lower() in label.lower() for f in fallback_labels)) else ""
                audit_data.append({
                    "Feature Name": feat,
                    "UI Description": ui_desc,
                    "Pillar": pillar,
                    "Subgroup": subgroup,
                    "Order": len(options),
                    "Technical Value": map_key,
                    "Code": code,
                    "Default/Fallback": is_fallback,
                    "Count": count,
                    "Category": label,
                    "Encoding Strategy": encoding,
                    "Usage Role": role
                })
                option_labels.add(label)

        # NaN Row
        accounted_count = sum(d['Count'] for d in audit_data if d['Feature Name'] == feat)
        nan_count = max(0, 29556 - accounted_count)
        audit_data.append({
            "Feature Name": feat,
            "UI Description": ui_desc,
            "Pillar": pillar,
            "Subgroup": subgroup,
            "Order": "",
            "Technical Value": "NaN",
            "Code": "",
            "Default/Fallback": "x",
            "Count": nan_count,
            "Category": "Missing (NaN)",
            "Encoding Strategy": encoding,
            "Usage Role": role
        })

    # Final Export with explicit Column Ordering
    column_order = ["Feature Name", "UI Description", "Pillar", "Subgroup", "Order", "Technical Value", "Code", "Default/Fallback", "Count", "Category", "Encoding Strategy", "Usage Role"]
    pd.DataFrame(audit_data)[column_order].to_excel('features_check.xlsx', index=False)
    print(">>> Forensic Audit Complete. Saved to features_check.xlsx with UI Description.")

if __name__ == "__main__":
    run_audit()
