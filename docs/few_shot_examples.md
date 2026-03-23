# **Clinical Trial Enrichment: Few-Shot Examples (v14 Lean)**

**MANDATE**: These examples demonstrate the new "Reasoning-Ready" format. Python post-processing will handle IDs and Math.

--- 

### **Example 1: Oncology (Combination)**
**[CONTEXT]**
[TRIAL_START]
[NCT_ID]: NCT03734029
[START_YEAR]: 2018
[OFFICIAL_TITLE]: T-DXd Versus T-DM1 in HER2-positive Metastatic Breast Cancer
[PRIMARY_ENDPOINTS_DETAIL]: TITLE: Progression-Free Survival (PFS) | TIMEFRAME: 24 months
[TRIAL_END]

**[MAPPING_OUTPUT]**
```json
{
  "rationale": "Mapping: Oncology TA; Breast cancer selected from menu. Temporal: In 2018, T-DM1 was Standard of Care, hurdle is STANDARD. HER2 is a VALIDATED target. Duration: 24 months. Modality: BIOLOGIC_MAB (ADC). Biomarker: HER2-positive status required for enrollment.",
  "nct_id": "NCT03734029",
  "therapeutic_area_llm": "Oncology",
  "gbd_indication_name": "Breast cancer",
  "therapeutic_modality": "BIOLOGIC_MAB",
  "clinical_line_of_therapy": "LATER_LINE",
  "innovation_tier": "NEXT_GEN",
  "patient_severity": "ADVANCED_METASTATIC",
  "comparator_type": "ACTIVE_COMPARATOR",
  "raw_duration_value": 24.0,
  "raw_duration_unit": "months",
  "endpoint_rigor_tier": "HARD_CLINICAL",
  "rarity_tier": "COMMON",
  "sponsor_class": "BIG_PHARMA",
  "orphan_status": "UNLIKELY",
  "parent_sponsor_llm": "Daiichi Sankyo",
  "extraction_confidence": "HIGH",
  "target_pathway_class": "KINASE_INHIBITOR",
  "target_novelty": "VALIDATED",
  "molecular_target_name": "HER2",
  "biomarker_stratification": true,
  "biomarker_description": "HER2-positive",
  "pivotal_intent": true,
  "comparator_hurdle_tier": "STANDARD",
  "protocol_design_sophistication": "FIXED"
}
```

--- 

### **Example 2: Immunology (IBD Specificity)**
**[CONTEXT]**
[TRIAL_START]
[NCT_ID]: NCT03484299
[START_YEAR]: 2018
[OFFICIAL_TITLE]: A Study of Risankizumab in Participants With Moderate to Severe Active Crohn's Disease
[PRIMARY_ENDPOINTS_DETAIL]: TITLE: Percentage of participants achieving clinical remission | TIMEFRAME: 52 weeks
[TRIAL_END]

**[MAPPING_OUTPUT]**
```json
{
  "rationale": "Mapping: Immunology TA as per rule for IBD. Indication: Deepest level L4 'Crohn's disease' selected. Temporal: In 2018, IL-23 inhibitors were NEXT_GEN. Duration: 52 weeks. Modality: BIOLOGIC_MAB.",
  "nct_id": "NCT03484299",
  "therapeutic_area_llm": "Immunology",
  "gbd_indication_name": "Crohn's disease",
  "therapeutic_modality": "BIOLOGIC_MAB",
  "clinical_line_of_therapy": "LATER_LINE",
  "innovation_tier": "NEXT_GEN",
  "patient_severity": "CHRONIC_PROGRESSIVE",
  "comparator_type": "PLACEBO_CONTROL",
  "raw_duration_value": 52.0,
  "raw_duration_unit": "weeks",
  "endpoint_rigor_tier": "HARD_CLINICAL",
  "rarity_tier": "COMMON",
  "sponsor_class": "BIG_PHARMA",
  "orphan_status": "UNLIKELY",
  "parent_sponsor_llm": "AbbVie",
  "extraction_confidence": "HIGH",
  "target_pathway_class": "INTERLEUKIN_CYTOKINE",
  "target_novelty": "VALIDATED",
  "molecular_target_name": "IL23",
  "biomarker_stratification": false,
  "biomarker_description": "N/A",
  "pivotal_intent": true,
  "comparator_hurdle_tier": "PLACEBO",
  "protocol_design_sophistication": "FIXED"
}
```

--- 

### **Example 3: Metabolic (Novel)**
**[CONTEXT]**
[TRIAL_START]
[NCT_ID]: NCT03987919
[START_YEAR]: 2019
[OFFICIAL_TITLE]: Tirzepatide Monotherapy in Patients With Type 2 Diabetes
[PRIMARY_ENDPOINTS_DETAIL]: TITLE: Mean change in HbA1c | TIMEFRAME: 40 weeks
[TRIAL_END]

**[MAPPING_OUTPUT]**
```json
{
  "rationale": "Mapping: Metabolic TA; Diabetes mellitus type 2 selected. Temporal: In 2019, dual GIP/GLP-1 was NOVEL. Comparison is PLACEBO. Duration: 40 weeks. Modality: PEPTIDE_HORMONES.",
  "nct_id": "NCT03987919",
  "therapeutic_area_llm": "Metabolic",
  "gbd_indication_name": "Diabetes mellitus type 2",
  "therapeutic_modality": "PEPTIDE_HORMONES",
  "clinical_line_of_therapy": "FIRST_LINE",
  "innovation_tier": "FIRST_IN_CLASS",
  "patient_severity": "CHRONIC_STABLE",
  "comparator_type": "PLACEBO_CONTROL",
  "raw_duration_value": 40.0,
  "raw_duration_unit": "weeks",
  "endpoint_rigor_tier": "SURROGATE",
  "rarity_tier": "COMMON",
  "sponsor_class": "BIG_PHARMA",
  "orphan_status": "UNLIKELY",
  "parent_sponsor_llm": "Lilly",
  "extraction_confidence": "HIGH",
  "target_pathway_class": "GPCR_TARGET",
  "target_novelty": "NOVEL",
  "molecular_target_name": "GIP | GLP-1",
  "biomarker_stratification": false,
  "biomarker_description": "N/A",
  "pivotal_intent": true,
  "comparator_hurdle_tier": "PLACEBO",
  "protocol_design_sophistication": "FIXED"
}
```