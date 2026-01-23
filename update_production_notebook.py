import nbformat as nbf

def update_artifact_names():
    file_path = 'notebooks/production_01.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    target_id = "788b52ba"
    
    for cell in nb.cells:
        if cell.id == target_id:
            # Update the export logic with new names and add SHAP export
            new_source = r"""# 1. Save Model, Thresholds, and Taxonomy
joblib.dump(model, MODELS_PATH / 'model_prod_01.joblib')

cal_export = {
    'global_threshold': float(global_thresh),
    'ta_thresholds': {ta: float(t) for ta, t in final_thresholds.items()},
    'ta_threshold_logits': {ta: float(l) for ta, l in final_threshold_logits.items()},
    'base_value': float(model_base_value),
    'gain_factor': float(GAIN_FACTOR)
}
with open(MODELS_PATH / 'thresholds_01.json', 'w') as f:
    json.dump(cal_export, f, indent=4)

with open(MODELS_PATH / 'taxonomy_01.json', 'w') as f:
    json.dump(RISK_TAXONOMY, f, indent=4)

# 2. Save SHAP Values (as requested: shap_values_01.json)
# Note: Converting to list for JSON serialization
with open(MODELS_PATH / 'shap_values_01.json', 'w') as f:
    json.dump(shap_values.tolist(), f)

# 3. Create Search Database
cols_to_keep = ['nct_id', 'brief_title', 'official_title', 'therapeutic_area', 'therapeutic_subgroup_name', 'lead_sponsor', 'phase', 'start_year', 'target', 'why_stopped']
app_df = df[[c for c in cols_to_keep if c in df.columns]].copy()
app_df['Clinical_Score'] = df_scores['Clinical_Score']
app_df['Zone'] = df_scores['Zone']

row_thresholds = app_df['therapeutic_area'].map(final_thresholds).fillna(global_thresh)
y_pred = (y_prob_train >= row_thresholds).astype(int)
app_df['is_demo_quality'] = (y_pred == app_df['target']).astype(int)

app_df.to_csv(MODELS_PATH / 'app_search_data_01.csv', index=False)
print(">>> ALL ARTIFACTS EXPORTED.")"""
            cell.source = new_source
            print("Successfully updated artifact export cell.")
            break

    with open(file_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    update_artifact_names()