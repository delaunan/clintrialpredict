# **Clinical Trial Prediction: Project Status & Architecture (v44.0)**

## **1. The Enrichment Engine Manifest (Production v22.0)**
| Run | Stage Name | Instruction (Ground Truth) | Input Context | Output Data | Runner Script |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Run 1** | **Clinical Epidemiologist** | `docs/prompts/llm_prompt_in_01.md` | `data/llm_in_01.csv` | `data/processed/llm_out_00.csv` | `src/prep/llm_in_01_run.py` |
| **Run 1.2** | **Forensic Refiner** | `docs/prompts/llm_prompt_in_01_2.md` | `data/processed/llm_out_00.csv` | `data/processed/llm_out_01_2.csv` | `src/prep/llm_in_01_2_run.py` |
| **Run 2** | **Molecular Blueprint** | `docs/prompts/llm_prompt_in_02.md` | `data/llm_in_02.csv` | `data/processed/llm_out_02.csv" | `src/prep/llm_in_02_run.py` |
| **Run 3** | **Forensic Strategist** | `docs/prompts/llm_prompt_in_03.md" | `data/llm_in_03.csv` | `data/processed/llm_out_03.csv" | `src/prep/llm_in_03_run.py` |
| **Run 4** | **Structural Anchor** | `docs/prompts/llm_prompt_in_04.md` | `data/llm_in_04.csv` | `data/processed/llm_out_04.csv` | `src/prep/llm_in_04_run.py` |
| **Merge** | **Master Enriched** | N/A | Multiple | `data/processed/llm_out_01.csv` | `src/prep/gbd_master_merge.py` |

---

## **2. Production Engine Approach (notebooks/production_06.ipynb)**
- **Objective**: Construct the finalized predictive model for deployment using the full historical signal.
- **Analytical Universe**: Full historical cohort of **~24,191 trials (2009–2022)**.
- **Recency-Weighted Calibration**: 
    - **Logic**: Applies temporal weights to training records *only during threshold optimization*. 
    - **Formula**: Linear weight gradient from **0.1 (2009)** to **1.0 (2022)**.
    - **Impact**: Prioritizes modern failure patterns for the decision boundary while XGBoost retains scientific depth from the full 13-year depth.
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

### **B. The Service "Handshake" (Communication Flow)**
1.  **Local Discovery (Streamlit)**: Users filter/search trials using the local `search_registry.csv` database. All searching is in-memory and instant.
2.  **The Handshake Request**: Upon selecting a trial, Streamlit sends `nct_id` and `therapeutic_area` to the API via an HTTP POST.
3.  **The API Lookup Hub**: 
    - FastAPI fetches the specific **SHAP vector** from the dictionary.
    - It retrieves the **TA-specific logit boundary**.
    - It reconstructs the **Success Score (0-100)** where 50.0 is the exact boundary.
    - It applies the **Calibration Offset** to the "Therapeutic Context" pillar to ensure mathematical summing consistency.
4.  **The JSON Response**: API returns a structured package including the score, pillar impacts, and impact-aware narratives.
5.  **Vibrant Visualization**: Streamlit renders the Gauge, Treemap, and Impact Bar charts using the API's mathematical truth.

---

### **C. Strategic Rationale ("The Why")**
- **Audit Mode Primacy**: v01 focuses on auditing the existing universe. Pre-calculating SHAP values eliminates BioBERT environment drift and reduces per-click latency from seconds to milliseconds.
- **Mathematical Parity**: Centralizing scoring logic in FastAPI ensures that all UI elements (Gauge vs Treemap) are bit-perfectly aligned using the same offsets validated in research.
- **Vertical Scalability**: Isolating the search database in the frontend container allows Streamlit to scale horizontally without putting load on the inference API.

---

## **4. Data Integrity & Unified Feature Engine**
- **Single Source of Truth**: `src/prep/pipeline.py` consolidate all 40+ feature mappings, encoding strategies, and UI hierarchies into one self-contained module.
- **Registry-Aware Imputation**: The `RegistryImputer` bakes fallback codes (0, 1, 2) into the serialized model, removing run-time dependency on external JSON files.
- **Zero-Anchor Standard**: Guaranteed that all "Baseline/Unknown" states map to **Integer 0** with standardized **"Not Specified"** UI labels.
- **Hierarchy Recovery**: `ClinicalTrialLoader` uses recursive lookup to reconstruct full ancestral lineage (ID_4 -> ID_3 -> ID_2) for 100% of the universe.
