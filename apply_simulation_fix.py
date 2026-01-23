import nbformat as nbf
import os

def update_notebook_simulation():
    nb_path = 'notebooks/production_01.ipynb'
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)
    
    # 1. Clean up all cells from Step 14 onwards to ensure fresh generation
    # We look for the Predict Universe start marker
    markers = [
        "#### <REF:PROD_PREDICT_GEN>", 
        "#### <REF:PROD_SERIALIZE_LOCKED>", 
        "#### <REF:PROD_FORENSIC_AUDIT>", 
        "#### <REF:PROD_FEATURE_AUDIT>",
        "#### <REF:PROD_SIGNAL_AUDIT>"
    ]
    nb.cells = [c for c in nb.cells if not any(m in c.source for m in markers)]
    
    # 2. Add Step 14 (Ingestion)
    step14_md = "#### <REF:PROD_PREDICT_GEN>\n> #### **14. Predict Universe Generation: Full Portfolio Engineering**\n\nThis phase utilizes the specialized `ClinicalTrialLoaderPredict` to ingest the expanded clinical universe (2005–2026). Unlike the training phase, this loader preserves ongoing trials (Recruiting, Active) and enriches the metadata with enrollment and facility counts to provide a comprehensive searchable registry."
    step14_code = "# --- STEP 14: LOAD FULL PREDICT UNIVERSE (2005-2026) ---\nfrom src.prep.data_loader_predict import ClinicalTrialLoaderPredict

print(\">>> Initializing Predict Loader for expanded horizon (2005-2026)...\")
predict_loader = ClinicalTrialLoaderPredict(str(DATA_PATH))

# Load and Engineer Features for the full universe
df_universe = predict_loader.load_and_clean()
df_universe = predict_loader.add_features(df_universe)

# Reset index to ensure integer alignment during SHAP calculation
df_universe = df_universe.reset_index(drop=True)

n_hist = len(df_universe[df_universe.trial_segment == 'HISTORICAL'])
n_ongo = len(df_universe[df_universe.trial_segment == 'ONGOING'])

print(f\"\
>>> Full Predict Universe: {len(df_universe):,} trials engineered.\")
print(f\"    - Historical (Labeled): {n_hist:,}\")
print(f\"    - Ongoing (Predictive):  {n_ongo:,}\")"

    # 3. Add Step 16 (Math Audit - adding it here for sequence)
    step16_md = "#### <REF:PROD_FORENSIC_AUDIT>\n> #### **16. Forensic Data Audit: Mathematical Recoupling**\n\nThe final validation check ensures the exported artifacts are mathematically consistent. We perform a recoupling test to verify that the generated scores in the registry match the sum of the explainability drivers."
    step16_code = "# --- STEP 16: FORENSIC MATH AUDIT ---\nprint(\">>> Running Mathematical Recoupling Check...\")

sample_check = app_data.sample(5, random_state=42)
base_v = float(model_base_value)
gain = float(GAIN_FACTOR)

print(\"\
--- [MATH] Scoring Integrity Check ---\")
for _, row in sample_check.iterrows():
    nid = row['nct_id']
    target_score = row['Clinical_Score']
    ta = row['therapeutic_area']
    
    ta_t_logit = final_threshold_logits.get(ta, global_logit)
    s_sum = np.sum(shap_lookup[nid])
    
    p_logit = base_v + s_sum
    calc_s = np.clip(50 + (ta_t_logit - p_logit) * gain, 1, 99)
    
    diff = abs(target_score - calc_s)
    status = "PASSED" if diff < 0.01 else "FAILED"
    print(f\[\"{status}\\] {nid:<12} | CSV: {target_score:.2f} | Calc: {calc_s:.2f}")"

    # 4. Load Step 15 and 17 from the temp file
    with open('step17_simulation.tmp', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content into markdown text and code blocks
    # Sections start with #### <REF
    import re
    sections = re.split(r'(#### <REF:.*?>)', content)
    
    processed_steps = []
    current_md = ""
    for item in sections:
        if not item.strip(): continue
        if item.startswith('#### <REF:'): current_md = item
        else:
            parts = item.split("```python")
            md_text = current_md + "\n" + parts[0].strip()
            code_text = parts[1].replace("```", "").strip()
            processed_steps.append((md_text, code_text))

    # 5. Assemble final order: 14, 15, 16, 17
    nb.cells.append(nbf.v4.new_markdown_cell(step14_md))
    nb.cells.append(nbf.v4.new_code_cell(step14_code))
    
    # Add Step 15 (from processed_steps[0])
    nb.cells.append(nbf.v4.new_markdown_cell(processed_steps[0][0]))
    nb.cells.append(nbf.v4.new_code_cell(processed_steps[0][1]))
    
    # Add Step 16
    nb.cells.append(nbf.v4.new_markdown_cell(step16_md))
    nb.cells.append(nbf.v4.new_code_cell(step16_code))
    
    # Add Step 17 (from processed_steps[1])
    nb.cells.append(nbf.v4.new_markdown_cell(processed_steps[1][0]))
    nb.cells.append(nbf.v4.new_code_cell(processed_steps[1][1]))

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Successfully finalized {nb_path} with Steps 14-17 (Simulation fidelity).")

if __name__ == "__main__":
    update_notebook_simulation()
