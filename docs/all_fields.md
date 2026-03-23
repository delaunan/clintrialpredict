# ClinTrialPredict: Master Field & Lineage Registry

This document provides a detailed mapping of every field used in the system, its origin in the AACT database, and the transformation logic applied.

---

### 1. Primary Model Features (Training Input)
These features are processed through `preprocessing.py` and directly impact the prediction score.

| Technical Name | Source Table | Raw Source Column(s) | Transformation Logic | Role |
| :--- | :--- | :--- | :--- | :--- |
| `therapeutic_area` | `browse_conditions.txt` | `mesh_term` | **Dictionary Match**: Maps MeSH terms to 15+ broad areas (e.g., Oncology). | Context |
| `therapeutic_subgroup_name` | `browse_conditions.txt` | `mesh_term` | **Direct**: Used for Target Encoding (disease-specific baseline). | Context |
| `phase` | `studies.txt` | `phase` | **Direct**: One-hot encoded (Phase 2, 2/3, 3). | Context |
| `phase_group` | `studies.txt` | `phase` | **Bucketing**: Groups phases into 'Early_Efficacy' vs 'Confirmatory'. | Context |
| `sponsor_tier` | `sponsors.txt` | `name` | **Regex Filter**: Checks for Big Pharma keywords (Pfizer, Roche, etc.) to assign Tier 1/2. | Context |
| `number_of_arms` | `studies.txt` | `number_of_arms` | **Log Scaled**: `log1p` to normalize the distribution of arm counts. | Design |
| `masking` | `designs.txt` | `masking` | **Direct**: One-hot encoded level of blinding. | Design |
| `primary_purpose` | `designs.txt` | `primary_purpose` | **Direct**: One-hot encoded (Treatment, Diagnostic, etc.). | Design |
| `design_rigor_score` | `designs.txt` | `masking`, `allocation`, `intervention_model` | **Composite Sum**: Assigns points for Double Blinding (+2), Randomization (+1), and Crossover (+1). | Design |
| `pca_sci` | `studies.txt`, `brief_summaries.txt` | `official_title`, `description` | **NLP**: BioBERT embedding (768d) -> PCA reduction (160d). | Design (NLP) |
| `pca_endp` | `design_outcomes.txt` | `measure` | **NLP**: BioBERT embedding (768d) -> PCA reduction (170d). | Design (NLP) |
| `duration_months` | `outcomes.txt`, `studies.txt` | `time_frame`, `official_title` | **Waterfall Regex**: Extracts months from outcome timeframes or title mentions. | Execution |
| `has_dmc` | `studies.txt` | `has_dmc` | **Boolean**: Presence of a Data Monitoring Committee. | Execution |
| `is_acute` | `brief_summaries.txt`, `studies.txt` | `description`, `official_title` | **Regex**: Detects keywords like "Sepsis", "Emergency", "ICU", "Trauma". | Patient |
| `is_refractory` | `brief_summaries.txt`, `studies.txt` | `description`, `official_title` | **Regex**: Detects "Resistance", "Relapsed", "Second-line". | Patient |
| `is_severe` | `brief_summaries.txt`, `studies.txt` | `description`, `official_title` | **Regex**: Detects "Metastatic", "Stage IV", "Advanced". | Patient |
| `is_critical_setting` | - | Derived from above | **Union**: `MAX(is_acute, is_refractory, is_severe)`. | Patient |
| `is_sick_only` | `eligibilities.txt` | `healthy_volunteers` | **Inverse**: 1 if `healthy_volunteers` is False/No. | Patient |
| `criteria_len_log` | `eligibilities.txt` | `criteria` | **Log Count**: `log1p` of character count of inclusion/exclusion text. | Patient |
| `pca_crit` | `eligibilities.txt` | `criteria` | **NLP**: BioBERT embedding (768d) -> PCA reduction (160d). | Patient (NLP) |

---

### 2. Market Intelligence & Analysis Features
Engineered for diagnostic depth and future model versions.

| Technical Name | Source Table | Raw Source Column(s) | Transformation Logic | Role |
| :--- | :--- | :--- | :--- | :--- |
| `agent_category` | `interventions.txt` | `name`, `intervention_type` | **Regex Priority**: Classifies molecule type (e.g., MAB, CAR-T, JAK Inhibitor). | Market |
| `competition_broad` | `studies.txt` | `start_date` | **Rolling Density**: Count of trials in the same TA over the prior 24 months. | Market |
| `competition_niche` | `studies.txt` | `start_date` | **Rolling Density**: Count of trials in the same indication over prior 24 months. | Market |
| `eligibility_strictness`| `eligibilities.txt` | `gender`, `minimum_age`, `maximum_age` | **Scoring**: Points for gender restrictions and narrow age ranges. | Patient |
| `scientific_success` | `outcome_analyses.txt` | `p_value`, `p_value_modifier` | **Threshold**: Binary 1 if `p_value` <= 0.05 on primary endpoints. | Analysis |
| `is_fda_regulated_drug` | `studies.txt` | `is_fda_regulated_drug` | **Gated**: Verified FDA status for trials starting 2017 or later. | Regulatory |

---

### 3. UI Metadata & Display Fields
Fields that support the interface and user navigation.

| Technical Name | Source Table | Raw Source Column(s) | Transformation Logic | Role |
| :--- | :--- | :--- | :--- | :--- |
| `ui_title` | `studies.txt` | `official_title` | **Sanitization**: Removes HTML, NCT IDs, and future-dated edits. | UI Display |
| `ui_summary` | `brief_summaries.txt`| `description` | **Sanitization**: Removes dates and admin metadata. | UI Display |
| `ui_criteria` | `eligibilities.txt` | `criteria` | **Sanitization**: Removes struck-through content and admin markers. | UI Display |
| `enrollment` | `studies.txt` | `enrollment` | **Direct**: Used for dashboard scale indicator. | UI Info |
| `number_of_facilities` | `calculated_values.txt`| `number_of_facilities` | **Direct**: Used to show geographic breadth. | UI Info |
| `start_year` | `studies.txt` | `start_date` | **Date Extract**: Extracted year for filtering. | Filtering |
| `trial_segment` | `studies.txt` | `overall_status` | **Mapping**: Historical (Completed/Terminated) vs Ongoing. | Filtering |