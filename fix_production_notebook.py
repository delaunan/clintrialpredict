
import nbformat as nbf

def fix_ui_and_scoring():
    file_path = 'notebooks/production_01.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # 1. Update Scoring Engine to include Pillars in df_scores
    for cell in nb.cells:
        if cell.get('id') == '3e8ef21d':
            # We preserve the RISK_TAXONOMY definition but fix the function
            cell.source = r"""# --- PRODUCTION RISK TAXONOMY (FULL FIDELITY) ---
RISK_TAXONOMY = {
    "1. Therapeutic Context": {
        "Indication Risk Profile": {
            "features": ["cat_onehot__therapeutic_area", "cat_target__therapeutic_subgroup_name", "CALIBRATION_OFFSET"],
            "feature_labels": ["Net Disease Risk"],
            "logic": "The combined risk profile of the therapeutic area and specific indication, adjusted for sector difficulty.",
            "pos_impact": "Therapeutic area and indication show a rather favorable risk profile based on historical evidence.",
            "neg_impact": "Therapeutic area and indication may correspond to elevated challenge levels given past records."
        },
        "Development Phase": {
            "features": ["cat_onehot__phase", "cat_onehot__phase_group"],
            "feature_labels": ["Trial Phase", "Phase Tier"],
            "logic": "The statistical probability of success (POS) based on development stage.",
            "pos_impact": "The current study stage is statistically associated with higher completion rates.",
            "neg_impact": "The current study stage is associated with higher attrition risks."
        }
    },
    "2. Scientific Design": {
        "Scientific Rationale": {
            "features": ["pca_sci"],
            "feature_labels": ["Scientific Text Analysis"],
            "logic": "Text analysis of the Title and Brief Summary for novelty and clarity.",
            "pos_impact": "Text analysis suggests the scientific rationale is clear, focused, and aligns with successful precedents.",
            "neg_impact": "Text analysis suggests the scientific description may be ambiguous, with lesser focus or highly novel, which tends to carry higher uncertainty."
        },
        "Endpoint Strategy": {
            "features": ["pca_endp"],
            "feature_labels": ["Endpoint Text Analysis"],
            "logic": "Text analysis of the Primary Endpoints for measurability.",
            "pos_impact": "Primary endpoints appear well-defined and measurable, consistent with previous successful trials.",
            "neg_impact": "Endpoints can be improved as they may appear complex or subjective, which tends to introduce measurement variability."
        },
        "Trial Complexity": {
            "features": ["num_log__number_of_arms", "cat_onehot__primary_purpose"],
            "feature_labels": ["Arm Count", "Primary Purpose"],
            "logic": "Structural design of the study arms.",
            "pos_impact": "The structural design (number of arms and primary purpose) suggests a comprehensive comparative strategy.",
            "neg_impact": "The structural design (number of arms and primary purpose) suggests either a simplified exploratory focus or a level of complexity that may increase operational burden,"
        }
    },
    "3. Execution Framework": {
        "Quality & Bias Control": {
            "features": ["bin_flags__has_dmc", "num_std__design_rigor_score", "cat_onehot__masking"],
            "feature_labels": ["Data Monitoring Comm.", "Rigor Score", "Blinding"],
            "logic": "Operational mechanisms implemented to minimize bias and ensure safety oversight.",
            "pos_impact": "Operational controls (e.g., DMC, masking) are in place to safeguard data integrity and reduce bias.",
            "neg_impact": "The protocol suggests limited bias mitigation or oversight mechanisms, increasing operational vulnerability."
        },
        "Time Horizon": {
            "features": ["num_log__duration_months"],
            "feature_labels": ["Estimated Duration"],
            "logic": "The operational exposure time of the study.",
            "pos_impact": "Shorter duration tends to minimize exposure to long-term operational risks.",
            "neg_impact": "Extended timeline tends to increase the risk of participant dropout and resource exhaustion."
        },
        "Sponsor Profile": {
            "features": ["cat_onehot__sponsor_tier"],
            "feature_labels": ["Sponsor Tier"],
            "logic": "The track record and financial depth of the sponsor.",
            "pos_impact": "Sponsor profile suggests established resources, experience or operational resilience.",
            "neg_impact": "Sponsor profile may be sensitive to funding constraints or prone to frequent strategic portfolio shifts."
        }
    },
    "4. Patient Profile": {
        "Inclusion Criteria": {
            "features": ["pca_crit", "num_std__criteria_len_log"],
            "feature_labels": ["Criteria Text Analysis", "Criteria Length"],
            "logic": "Text analysis of the eligibility rules.",
            "pos_impact": "Eligibility criteria appear concise and balanced, facilitating smoother enrollment.",
            "neg_impact": "Lengthy or highly restrictive criteria tend to create enrollment bottlenecks."
        },
        "Patient Acuity": {
            "features": ["bin_flags__is_sick_only", "bin_flags__is_severe", "bin_flags__is_acute", "bin_flags__is_refractory"],
            "feature_labels": ["Healthy vs Patient", "Severe Disease", "Acute Setting", "Refractory Status"],
            "logic": "The health status of the target population.",
            "pos_impact": "Target population appears stable, potentially reducing unexpected medical complications.",
            "neg_impact": "Populations with high disease severity, acuity, or refractory status tend to be associated with elevated operational risks."
        }
    }
}

GAIN_FACTOR = 25.0

def generate_production_scorecard(X, shap_values, feature_names, taxonomy, 
                                base_value, thresholds_logit_map, global_thresh_logit, gain=GAIN_FACTOR):
    # --- 1. Pillar Aggregation ---
    feat_to_pillar = {}
    for i, feat in enumerate(feature_names):
        for pillar, topics in taxonomy.items():
            for topic, details in topics.items():
                if any(feat.startswith(prefix) for prefix in details['features']):
                    feat_to_pillar[i] = pillar

    pillar_shap_sums = pd.DataFrame(0.0, index=X.index, columns=taxonomy.keys())
    for i in range(shap_values.shape[1]):
        p = feat_to_pillar.get(i)
        if p:
            pillar_shap_sums[p] += shap_values[:, i]

    # --- 2. Normalized Success Scoring ---
    row_threshold_logits = X['therapeutic_area'].map(thresholds_logit_map).fillna(global_thresh_logit)
    model_pred_logit = base_value + np.sum(shap_values, axis=1)
    
    delta_logit = row_threshold_logits - model_pred_logit
    scorecard = pd.DataFrame(index=X.index)
    scorecard['Clinical_Score'] = (50 + (delta_logit * gain)).clip(1, 99)

    # --- 3. Pillar Decomposition (Success-Oriented) ---
    calibration_offset = row_threshold_logits - base_value
    for pillar in taxonomy.keys():
        # IMPORTANT: We add the pillars directly to scorecard so validation UI logic works
        scorecard[pillar] = pillar_shap_sums[pillar] * gain * -1 # Invert risk to success

    scorecard['CALIBRATION_OFFSET'] = calibration_offset * gain
    scorecard['Therapeutic_Area'] = X['therapeutic_area']
    scorecard['Zone'] = pd.cut(scorecard['Clinical_Score'], [0, 25, 50, 75, 100], 
                          labels=["High Risk", "Watchlist", "Good", "Robust"])

    return scorecard

df_scores = generate_production_scorecard(
    X_train, shap_values, feature_names, RISK_TAXONOMY,
    base_value=model_base_value,
    thresholds_logit_map=final_threshold_logits,
    global_thresh_logit=global_logit
)
print(f">>> Mean Production Success Score: {df_scores['Clinical_Score'].mean():.1f}")"""
            break

    # 2. Update UI Sync (Cell Step 11 / REF:UI_VIS_CODE)
    for cell in nb.cells:
        if cell.get('cell_type') == 'code' and '# <REF:UI_VIS_CODE>' in cell.source:
            # We ensure the execution part is clean and synced with the updated df_scores
            lines = cell.source.split('\n')
            new_lines = []
            for line in lines:
                if 'target_idx = df_scores.index[Trial_Example]' in line:
                    new_lines.append("# ==============================================================================")
                    new_lines.append("# 4. EXECUTION (Synced with Production Dataset)")
                    new_lines.append("# ==============================================================================")
                    new_lines.append("target_idx = df_scores.index[Trial_Example]")
                    new_lines.append("score_val = df_scores.loc[target_idx, 'Clinical_Score']")
                    new_lines.append("")
                    new_lines.append("# Display Production Metadata")
                    new_lines.append('print(f"NCT ID:    {df.loc[target_idx, \'nct_id\']}")')
                    new_lines.append('print(f"Title:     {df.loc[target_idx, \'brief_title\']}")')
                    new_lines.append('print(f"Area:      {df.loc[target_idx, \'therapeutic_area\']}")')
                    new_lines.append('outcome_str = "Failure (1)" if df.loc[target_idx, "target"]==1 else "Success (0)"')
                    new_lines.append('print(f"Outcome:   {outcome_str}")')
                    new_lines.append("")
                    continue
                
                # Keep the plotting logic as requested
                new_lines.append(line)
            
            cell.source = '\n'.join(new_lines)

    with open(file_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Successfully fixed Scoring and synced UI in {file_path}")

if __name__ == "__main__":
    fix_ui_and_scoring()
