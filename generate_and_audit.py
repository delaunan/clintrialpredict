import os
import pandas as pd
import numpy as np
import json
import sys

# Ensure project root is in path for imports
sys.path.append(os.getcwd())

from src.prep.data_loader_clinpred import ClinicalTrialLoader

def run():
    print(">>> Step 1: Generating data/data_clinpred.csv")
    os.environ['GLOBAL_START_YEAR'] = '2009'
    os.environ['GLOBAL_END_YEAR'] = '2025'

    loader = ClinicalTrialLoader(data_path='data')
    try:
        # Load and process data
        df = loader.load_base_data()
        df = loader.add_features(df)
        loader.save(df, 'data_clinpred.csv')
    except Exception as e:
        print(f"Error during data generation: {e}")
        raise e

    print(">>> Step 2: Generating features_check.xlsx")
    with open('models/taxonomy_01.json', 'r') as f:
        taxonomy = json.load(f)

    fields_meta = taxonomy.get('FIELDS', {})

    exclusions = ['sci_', 'endp_', 'crit_']
    disabled = [
        'includes_us_ml', 'is_fda_regulated_drug_ml', 'gbd_cause_id_ml', 
        'gbd_cause_id_2_ml', 'gbd_cause_id_4_ml', 'gbd_hierarchy_level_ml', 
        'is_duration_unknown_ml', 'target_ml'
    ]

    # Priority: _ml fields first
    ml_fields = [f for f in fields_meta.keys() if f.endswith('_ml') and f not in disabled and not any(f.startswith(ex) for ex in exclusions)]
    other_fields = [f for f in fields_meta.keys() if f not in ml_fields and f not in disabled and not any(f.startswith(ex) for ex in exclusions)]
    
    all_to_process = ml_fields + other_fields

    audit_data = []

    for field_name in all_to_process:
        meta = fields_meta[field_name]
        ui = meta.get('ui', {})
        options = ui.get('options', [])
        mapping = meta.get('mapping', {})

        # 1. Numeric Fields Summary
        if field_name in ['number_of_arms_ml', 'primary_duration_months_ml']:
            if field_name in df.columns:
                series = pd.to_numeric(df[field_name], errors='coerce')
                audit_data.append({'Feature Name': field_name, 'Category': 'Summary: Min', 'Count': series.min(), 'Default/Fallback': '', 'Code': '', 'Order': ''})
                audit_data.append({'Feature Name': field_name, 'Category': 'Summary: Max', 'Count': series.max(), 'Default/Fallback': '', 'Code': '', 'Order': ''})
                audit_data.append({'Feature Name': field_name, 'Category': 'Summary: Median', 'Count': series.median(), 'Default/Fallback': '', 'Code': '', 'Order': ''})
                audit_data.append({'Feature Name': field_name, 'Category': 'NaN (Missing)', 'Count': int(series.isna().sum()), 'Default/Fallback': 'x', 'Code': 'NaN', 'Order': ''})
            continue

        # 2. gbd_cause_id_3_ml (Target Encoding)
        if field_name == 'gbd_cause_id_3_ml':
            if field_name in df.columns:
                counts = df[field_name].value_counts(dropna=False)
                # Try to get names from gbd_indication_name_3 or gbd_indication_name
                id_to_name = {}
                name_col = 'gbd_indication_name_3' if 'gbd_indication_name_3' in df.columns else 'gbd_indication_name'
                if name_col in df.columns:
                    temp_map = df[[field_name, name_col]].dropna().drop_duplicates()
                    id_to_name = dict(zip(temp_map[field_name], temp_map[name_col]))

                unique_ids = [uid for uid in df[field_name].dropna().unique()]
                for uid in sorted(unique_ids):
                    audit_data.append({
                        'Feature Name': field_name,
                        'Category': id_to_name.get(uid, f"ID {uid}"),
                        'Count': int(counts.get(uid, 0)),
                        'Default/Fallback': '',
                        'Code': uid,
                        'Order': ''
                    })
                audit_data.append({
                    'Feature Name': field_name,
                    'Category': 'NaN (Missing)',
                    'Count': int(counts.get(np.nan, 0)),
                    'Default/Fallback': 'x',
                    'Code': 'NaN',
                    'Order': ''
                })
            continue

        # 3. Categorical Fields (options/mapping)
        if options or mapping:
            # Combine all keys from mapping and options
            all_keys = set()
            if mapping:
                all_keys.update(mapping.keys())
            if options:
                all_keys.update([opt[0] for opt in options])
            
            # Map of val_key to its order in options
            key_to_order = {opt[0]: i for i, opt in enumerate(options)}
            
            # Categorize by the value used in the dataframe (Code)
            # If it's an _ml field, the dataframe contains the codes from mapping
            processed_codes = set()
            
            if field_name in df.columns:
                counts = df[field_name].value_counts(dropna=False)
            else:
                counts = pd.Series()

            # We want to list all categories from the JSON
            # Categories are defined by the mapping values or options labels
            
            # First, process based on options to maintain order
            seen_categories = set()
            
            for opt in options:
                val_key, label = opt
                code = None
                if mapping and val_key in mapping:
                    code = mapping[val_key][0]
                else:
                    code = val_key # Fallback
                
                count = counts.get(code, 0) if field_name.endswith('_ml') else counts.get(val_key, 0)
                
                # Default logic: code is 0 or label contains keywords
                is_default = ''
                if str(code) == '0' or code == 0:
                    is_default = 'x'
                else:
                    lbl_up = label.upper()
                    if any(kw in lbl_up for kw in ["UNKNOWN", "UNCLASSIFIED", "LINE N/A", "NOT SPECIFIED"]):
                        is_default = 'x'

                audit_data.append({
                    'Feature Name': field_name,
                    'Category': label,
                    'Count': int(count),
                    'Default/Fallback': is_default,
                    'Code': code,
                    'Order': key_to_order.get(val_key, '')
                })
                seen_categories.add(str(code))

            # Add categories from mapping that were not in options
            if mapping:
                for val_key, map_val in mapping.items():
                    code, label = map_val
                    if str(code) not in seen_categories:
                        count = counts.get(code, 0) if field_name.endswith('_ml') else counts.get(val_key, 0)
                        
                        is_default = ''
                        if str(code) == '0' or code == 0:
                            is_default = 'x'
                        else:
                            lbl_up = label.upper()
                            if any(kw in lbl_up for kw in ["UNKNOWN", "UNCLASSIFIED", "LINE N/A", "NOT SPECIFIED"]):
                                is_default = 'x'

                        audit_data.append({
                            'Feature Name': field_name,
                            'Category': label,
                            'Count': int(count),
                            'Default/Fallback': is_default,
                            'Code': code,
                            'Order': key_to_order.get(val_key, '')
                        })
                        seen_categories.add(str(code))

            # NaN row
            audit_data.append({
                'Feature Name': field_name,
                'Category': 'NaN (Missing)',
                'Count': int(counts.get(np.nan, 0)),
                'Default/Fallback': 'x',
                'Code': 'NaN',
                'Order': ''
            })

    audit_df = pd.DataFrame(audit_data)
    audit_df.to_excel('features_check.xlsx', index=False)
    print(f">>> Successfully generated features_check.xlsx with {len(audit_df)} rows.")

if __name__ == "__main__":
    run()
