# ClinTrialPredict: Master UI Field Registry (v01)

This document defines the interface architecture for the ClinTrialPredict platform. It distinguishes between **Source Inputs** (controllable for simulation) and **Internal Intelligence** (derived for audit).

---

### I. User-Controllable Simulation Inputs (Source Fields)
*These are the "Levers." In Audit Mode, they display database facts. In Simulation Mode, these are the interactive widgets (sliders/dropdowns).*

| Field | Variable | Role | Type | Logic / Simulation Value |
| :--- | : :--- | :--- | :--- | :--- |
| **Development Phase** | `phase` | Source | Select | Phase 2, 2/3, or 3. High impact on decision boundary. |
| **Enrollment Target** | `enrollment` | Source | Slider | Defines the "Scale Risk." |
| **Study Scale** | `number_of_arms` | Source | Slider | Complexity marker. More arms = higher failure risk. |
| **Planned Duration** | `duration_months` | Source | Slider | The operational horizon. |
| **Site Breadth** | `number_of_facilities`| Source | Slider | Reflects logistical and geographic reach. |
| **Healthy Volunteers**| `healthy_volunteers`| Source | Toggle | **Direct Input**. If 'No', the model flags `is_sick_only`. |
| **Protocol Blinding** | `masking` | Source | Select | Double, Single, None. Source for Design Rigor. |
| **Randomization** | `allocation` | Source | Select | Randomized vs. Non-randomized. Source for Design Rigor. |
| **Structural Model** | `intervention_model`| Source | Select | Parallel, Crossover, etc. Source for Design Rigor. |
| **Primary Purpose** | `primary_purpose` | Source | Select | Treatment, Prevention, Diagnostic, etc. |
| **Therapeutic Area** | `therapeutic_area` | Source | Search | The "Context" anchor (Oncology, Neurology, etc.). |
| **Medical Indication**| `therapeutic_subgroup_name`| Source | Search | The specific disease niche (Target Encoded). |
| **Oversight (DMC)** | `has_dmc` | Source | Toggle | Presence of a Data Monitoring Committee. |
| **FDA Regulated** | `is_fda_regulated_drug`| Source | Toggle | Regulatory oversight status. |

---

### II. Internal Model Intelligence (Calculated Features)
*These are the "Diagnostic Badges." They are derived from the source data or narrative text. In the UI, these explain the "Discovery" made by the model's brain.*

| Feature | Variable | Source | UI Display | Discovery Logic |
| :--- | :--- | :--- | :--- | :--- |
| **Patient Acuity** | `is_acute` | Narrative | Badge | System detected "ICU," "Sepsis," or "Trauma" in text. |
| **Drug Resistance** | `is_refractory` | Narrative | Badge | System detected "Failed Prior Lines" or "Resistant." |
| **Disease Burden** | `is_severe` | Narrative | Badge | System detected "Metastatic" or "Stage IV" text. |
| **Acute Setting** | `is_critical_setting`| Logic | Alert | Aggregate of Acute/Refractory/Severe flags. |
| **Patient Only** | `is_sick_only` | Logic | Badge | Set automatically if Healthy Volunteers = "No". |
| **Design Rigor** | `design_rigor_score`| Logic | Meter | Inferred quality index (Masking + Allocation + Model). |
| **Enrollment Friction**| `eligibility_strictness_score`| Logic | Meter | Complexity of Age/Gender/Sick constraints. |
| **Criteria Density** | `criteria_len_log` | Logic | Signal | How many "rules" are in the inclusion/exclusion text. |
| **Agent Profile** | `agent_category` | Narrative | Label | Inferred from drug name (e.g., CAR-T, JAK Inhibitor). |
| **Sponsor Power** | `sponsor_tier` | Context | Label | Tier 1 (Giant) vs Tier 2 (Other). |
| **Market Density** | `competition_niche` | Context | Signal | Count of similar trials in the current market. |

---

### III. Audit Metadata & Narrative (Information)
*The "Glass Box" surroundings. These fields provide trust and transparency without directly driving simulation math.*

| Information | Variable | Purpose | Logic |
| :--- | :--- | :--- | :--- |
| **NCT Identity** | `nct_id` | Unique ID | Primary key for database lookups. |
| **Display Title** | `ui_brief_title` | Metadata | Cleaned name for the dashboard selection. |
| **Lead Sponsor** | `lead_sponsor` | Metadata | The standardized organization name. |
| **Data Fidelity** | `prediction_quality`| Result | **High, Med, Low**. Based on missing database fields. |
| **Imputation Count** | `imputed_features_count`| Audit | Shows how many fields the model had to "guess". |
| **Start Year** | `start_year` | Metadata | Temporal context for recency weighting. |
| **Brief Summary** | `ui_summary` | Narrative | Cleaned, date-free narrative text. |
| **Eligibility Criteria**| `ui_criteria` | Narrative | Cleaned list of patient requirements. |

---

### IV. Predictive Results & Performance
*The final outputs of the production engine.*

| Result | Variable | UI Component | Meaning |
| :--- | :--- | :--- | :--- |
| **Clinical Score** | `Clinical_Score` | Gauge | **0-100 Score**. 50.0 is the TA-specific boundary. |
| **Confidence Zone** | `Zone` | Label | Robust, Good, Watchlist, High Risk. |
| **Impact Drivers** | `shap_values` | Treemap | Decomposition of the score into the 4 Pillars. |
