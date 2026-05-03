# **Clinical Trial Prediction: Project Status & Architecture (v55.0)**

## **1. The Enrichment Engine Manifest (Production v23.0)**
| Run | Stage Name | Instruction (Ground Truth) | Input Context | Output Data | Runner Script |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Run 1** | **Clinical Epidemiologist** | `docs/prompts/llm_prompt_in_01.md` | `data/llm_in_01.csv` | `data/processed/llm_out_00.csv` | `src/prep/llm_in_01_run.py` |
| **Run 1.2** | **Forensic Refiner** | `docs/prompts/llm_prompt_in_01_2.md` | `data/processed/llm_out_00.csv` | `data/processed/llm_out_00.csv` | `src/prep/llm_in_01_2_run.py` |
| **Run 2** | **Molecular Blueprint** | `docs/prompts/llm_prompt_in_02.md` | `data/llm_in_02.csv` | `data/processed/llm_out_02.csv` | `src/prep/llm_in_02_run.py` |
| **Run 3** | **Forensic Strategist** | `docs/prompts/llm_prompt_in_03.md` | `data/llm_in_03.csv` | `data/processed/llm_out_03.csv` | `src/prep/llm_in_03_run.py` |
| **Run 4** | **Structural Anchor** | `docs/prompts/llm_prompt_in_04.md` | `data/llm_in_04.csv` | `data/processed/llm_out_04.csv` | `src/prep/llm_in_04_run.py` |
| **Merge** | **Master Enriched** | N/A | Multiple | `data/processed/llm_out_01.csv` | `src/prep/llm_in_01_3_run.py` |

---

## **2. Production Engine Approach (notebooks/production_01.ipynb)**
- **Objective**: Construct the finalized predictive model for deployment using the full historical signal.
- **Analytical Universe**: Expanded cohort of **34,066 trials (2009–2026)**.
- **Recency-Weighted Calibration**: 
    - **Logic**: Applies temporal weights to training records *only during threshold optimization*. 
    - **Formula**: Linear weight gradient from **0.1 (2009)** to **1.0 (2026)**.
    - **Impact**: Prioritizes modern failure patterns for the decision boundary while XGBoost retains scientific depth from the full 17-year depth.
- **Decision Policy ($\beta = 1.10$)**:
    - **Recall Focus**: Optimized to intercept 70-75% of portfolio failures.
    - **TA-Specific Boundaries**: Generates granular logit thresholds for 13+ Therapeutic Areas.
- **Surgical Lean Toggle**: 
    - **Toggle**: A single-line drop command in Cell 4 allows the engine to strip ~2,300 embedding columns.
    - **Efficiency**: Reduces final registry size by ~90% (from ~500MB to ~50MB) while maintaining 100% predictive power for metadata features.

---

## **3. Deployment Artifacts & Interaction Architecture**

### **A. Artifact Inventory**
| Service Layer | Artifact Name | Location | Content / Purpose |
| :--- | :--- | :--- | :--- |
| **API (FastAPI)** | `model_prod_01.joblib` | `models/` | Full Pipeline (Preprocessor + XGBoost). Used for feature validation. |
| **API (FastAPI)** | `thresholds_01.json` | `models/` | Logit boundaries per TA and the `CALIBRATION_OFFSET` constants. |
| **API (FastAPI)** | `shap_values_01.joblib` | `models/` | Dictionary mapping `nct_id` to pre-calculated SHAP attribution arrays. |
| **API (FastAPI)** | `taxonomy_01.json` | `models/` | Logic-locked map of Feature-to-Pillar display hierarchies. |
| **UI (Streamlit)** | `search_registry.csv` | `frontend/data/` | Enriched trial database with titles, scores, and sanitized narratives. |

---

## **4. Data Integrity & Unified Feature Engine**
- **Single Source of Truth**: `src/prep/pipeline.py` consolidate all 40+ feature mappings, encoding strategies, and UI hierarchies into one self-contained module.
- **Registry-Aware Imputation**: The `RegistryImputer` bakes fallback codes (0, 1, 2) into the serialized model, removing run-time dependency on external JSON files.
- **Zero-Anchor Standard**: Guaranteed that all "Baseline/Unknown" states map to **Integer 0** with standardized **"Not Specified"** UI labels.
- **Hierarchy Recovery**: `ClinicalTrialLoader` uses recursive lookup to reconstruct full ancestral lineage (ID_4 -> ID_3 -> ID_2) for 100% of the universe.
- **Clean UI & Logic Refinement (v20.0 Standard)**:
    - **Global Renaming**: "Scientific Challenge" has been renamed to **"Scientific Challenge"**.
    - **Subgroup Consolidation**: "Execution Framework" is now strictly organized into two subgroups: **Methodological Setup** and **Trial Complexity Footprint**.
    - **Sponsor Integration**: "Sponsor Type" is now classified under **Trial Complexity Footprint** (Priority 20.9).
    - **Metadata Pruning**: Features with high cardinality or low immediate signal (Age, FDA Reg., Specific Indications) are reclassified to the **Metadata** pillar to focus the Treemap on high-impact drivers.
    - **Global Renaming (Phase)**: "Early Phase / Dose Finding" has been globally renamed to **"Dose Characterization"** to better reflect scientific intent.
    - **Identity Purge**: Targeted removal of redundant `_ui` fields for Identity and System columns (e.g., `acronym_ui`, `nct_id_ui`, `target_ui`) to maintain a clean, single-entry project manifest.

---

## **5. Simulation-Ready CSV Architecture (v46.0 Upgrade)**
- **Unified Source of Truth**: The project has transitioned from `.txt` dependencies to a **Self-Documenting CSV** (`data/data_clinpred.csv`).
- **Semantic Tagging**: Scientific text fields now include parsable semantic tags for effortless LLM simulation:
    - **Interventions**: `• NAME: [Drug] || DESC: [Mechanism]` (newline-separated).
    - **Outcomes**: `• TITLE: [Measure] || TIMEFRAME: [Time]` (newline-separated).
- **Linguistic Diet Synchronization**: CSV character caps are now bit-perfectly matched to LLM context windows:
    - **`summary_ui`**: 5,000 chars.
    - **`criteria_ui`**: 10,000 chars.
    - **`title`**: 1,000 chars.
- **User-Friendly Interaction**: Switched from technical pipes (` || `) to standard newlines (`\n`) in all multiline fields. This enables users to edit trial protocol directly in Streamlit via "Enter" while maintaining perfect programmatic extractability.
- **Forensic Purge**: Redundant `raw_` scientific fields (Conditions, Interventions, Outcomes, Geography) are surgically dropped after UI engineering to keep the production dataset lean and secure.
- **Registry Alignment**: `PIPELINE_REGISTRY` now enforces role-specific documentation:
    - **ML Fields**: Documented as `Source -> Internal Code`.
    - **UI Fields**: Documented as `Display Labels` (ordered by dictionary priority).

---

## **7. Streamlit Application Architecture (v54.0)**

### **A. View Hierarchy & Navigation**
The application implements a multi-view state machine driven by `st.session_state`:
1.  **View 1: Intelligence Discovery (Landing)**: Interactive search and mission statement. Sidebar is **completely hidden** (via CSS) to ensure a clean professional entry.
2.  **View 2: Clinical Registry (Results)**: High-density `AgGrid` display. Sidebar becomes **visible**, containing filters and secret analysis tools.
3.  **View 3: Protocol Forensic (Trial Detail)**: Refined tab-based layout. Sidebar is **surgically removed** (via CSS) to focus on the trial protocol and prediction results.
4.  **View 4: Signal Analysis (Strategic Audit)**: Integrates vibrant visualizations. Clicking **"Back to Results"** restores the sidebar and its previous filter states.

---

## **8. Latest Session Achievements (v55.0)**
- **'Steel Shield' Data Refresh**: Successfully refreshed the entire trial universe to **34,066 trials** (including new 2026 data).
- **Pipeline Robustness Upgrade**:
    - **Chemical Wash**: Implemented `wash_input_text` to normalize Greek characters (α, β, γ) preventing JSON corruption.
    - **Auto-Wrap Parser**: Upgraded `safe_json_loads` to automatically wrap single-object LLM responses into arrays when `BATCH_SIZE=1`.
    - **Context Isolation**: Deployed `### DATA_START ###` tags for perfect NCT_ID resolution and loop prevention.
- **Molecular Precision (Run 2)**: Performed the **"Ultimate Rescue"** using Batch Size 1, reducing "Unknown" molecular targets by 40% and reclaiming ~1,900 trials.
- **Structural Integrity (Run 4)**: Implemented the **"Long-Horizon Rule"** to differentiate between immediate dosing and primary study duration, significantly improving temporal data fidelity.
- **Model Standard**: Migrated all enrichment runners to **Gemini 2.5 Flash-Lite** for an optimal balance of 4,000 RPM quota and legacy 2.0 pricing.
- **Surgical Cleanup**: Reconciled the cumulative output databases with the target universe, removing 101 legacy trials to ensure 100% compliance with strict production filters.
- **Simulation Mode & Protocol Editing**: Finalized a comprehensive `global_edit_mode` architecture for interactive "What-If" analysis.
