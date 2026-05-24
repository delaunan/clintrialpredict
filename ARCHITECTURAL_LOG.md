# **Clinical Trial Prediction: Project Status & Architecture (v56.0)**

## **1. Production Deployment (v1.0 - "Steel Shield")**
- **Production URL**: [https://clintrial-ui-835962039082.europe-west1.run.app/](https://clintrial-ui-835962039082.europe-west1.run.app/)
- **Infrastructure**: Google Cloud Run (Serverless).
- **Architecture**: Decoupled FastAPI (Scoring Engine) + Streamlit (UI).
- **Analytical Universe**: **34,066 trials** (2009–2026).

---

## **2. As-Is Technical Architecture (The "Steel Shield" Engine)**

### **A. End-to-End ETL Pipeline**
The system uses a 4-stage transformation process to turn raw trial data into a strategic forensic registry:
1.  **Extraction (`src/prep/data_loader_clinpred.py`)**: Pulls raw clinical data from AACT (Postgres exports: `studies.txt`, `sponsors.txt`, `countries.txt`, `designs.txt`, `eligibilities.txt`, `interventions.txt`, `design_outcomes.txt`).
2.  **LLM Enrichment**: Consolidates 4 specialized LLM runs (`llm_out_01.csv` to `llm_out_04.csv`) covering Clinical Evidence, Pharmacology, Strategic Intent, and Structural Forensic Monologues.
3.  **Preprocessing (`src/prep/pipeline.py`)**: Uses a Scikit-learn Pipeline with a custom preprocessor anchored by the **Central Registry**.
4.  **Registry Generation (`refresh_registry.py`)**: Pre-calculates model probabilities, SHAP-based impacts, and parity-aligned scores for the entire universe, saving to `frontend/data/search_registry.csv`.

### **B. Central Registry (The Single Source of Truth)**
The `PIPELINE_REGISTRY` in `src/prep/pipeline.py` is the **master configuration** for the entire project:
- **Feature Metadata**: Defines every model feature, its encoding (Ordinal, Numeric, Target), and its allowed options.
- **UI Taxonomy**: Maps technical features to the 4 UI Pillars (**Therapeutic Context, Scientific Challenge, Execution Framework, Patient Profile**) and their respective Subgroups.
- **Consistency**: All agents must update this registry when adding new features to ensure the Model, API, and UI remain in sync.

### **C. Model Specifications (XGBoost v1.0)**
- **Algorithm**: `XGBClassifier` trained on the 17-year historical cohort (2009–2022).
- **Calibration**: Recency-Weighted (0.1 to 1.0 gradient) to prioritize modern failure patterns.
- **Artifacts**:
    - `models/model_prod_01.joblib`: Full Preprocessor + XGBoost Pipeline.
    - `models/shap_values_01.joblib`: Pre-calculated TreeExplainer SHAP vectors.
    - `models/thresholds_01.json`: Logit boundaries (Global and TA-specific).

### **D. Frontend Engine (`frontend/app.py`)**
The UI is a professional-grade Streamlit application designed for trial discovery and forensic simulation:
- **State Management**: Uses a robust `session_state` system to manage transitions between the **Landing View** (Search/Discovery) and the **Forensic View** (Trial Detail & Simulation).
- **Visual Shell**: Implements a high-precision CSS injection system for responsive typography, layout contracts (equal-height cards), and brand-specific aesthetics.
- **Shadow Audit System**: Implements server-side JSON logging (`audit_log`) for Cloud Run. It tracks visitor IDs (hashed IPs) and session transitions without storing PII or using third-party cookies.
- **Simulation Mode**: Enables manual overrides of model features. When "Simulation Mode" is toggled, the UI switches from reading `search_registry.csv` to calling the live FastAPI `/predict` endpoint.

### **E. The Parity Engine (Mathematical Identity)**
To ensure the UI and API match the Notebook precisely, the system implements **Residual Absorption**:
- **Baseline**: 50.0 points (Neutral).
- **Transformation**: Model logit impact is scaled by a `gain_factor` (default: 25.0).
- **Calibration Offset**: Adjusted per Therapeutic Area based on historical logit offsets.
- **Residual Absorption**: Any rounding errors or clipping residuals are absorbed into the "Therapeutic Context" pillar and "Therapeutic Area Profile" subcategory to preserve a perfect mathematical sum.

---

## **3. Data Universe & Feature Schema (The 27 Drivers)**

### **A. Primary Pillars & Features**
| Pillar | Features | Origin |
| :--- | :--- | :--- |
| **Therapeutic Context** | Therapeutic Area, Indication (GBD L3), Rare Disease Status, Clinical Phase, Regulatory Intent | AACT + GBD Mapping |
| **Scientific Challenge** | Target Precedent, Pathway Class, Modality, Innovation Tier, Intervention Model, Purpose, Design Flexibility, Endpoint Rigor, Endpoint Structure, Biomarker Stratification | LLM Enrichment + Designs |
| **Execution Framework** | Sponsor Tier, Bias Control (Masking), Allocation Method, DMC Involvement, Placebo Control, Benchmark Comparator, Endpoint Duration, Number of Arms, Delivery Complexity | AACT + LLM + Designs |
| **Patient Profile** | Clinical Severity, Line of Therapy, Gender Eligibility, Population Type (Healthy/Patient), Age Eligibility (Adult/Child/Older Adult) | AACT + LLM + Eligibilities |

### **B. Key Technical Constraints**
- **Ordinal Encoding**: Most features are mapped to 0-4 range enums (e.g., `innovation_tier_ml`: 0=Established, 1=Next-Gen, 2=First-in-Class).
- **Target Encoding**: `gbd_cause_id_3_ml` uses smoothed binary target encoding (smooth=200.0).
- **Numeric Scaling**: `number_of_arms_ml` and `primary_duration_months_ml` use `StandardScaler`.

---

## **4. Simulation Roadmap (v2.0 - "Strategic Forecaster")**
*Phase 2.1 (Simulation Workspace) and Phase 2.2 (Intelligent Protocol Mapping) are currently in development.*

---

## **5. Deployment Artifacts & Audit**
| Service Layer | Artifact Name | Purpose |
| :--- | :--- | :--- |
| **API** | [model_prod_01.joblib](file:///home/delaunan/code/delaunan/clintrialpredict/models/model_prod_01.joblib) | Production Scoring Engine. |
| **API** | [taxonomy_01.json](file:///home/delaunan/code/delaunan/clintrialpredict/models/taxonomy_01.json) | UI Hierarchy Mapping. |
| **UI** | [search_registry.csv](file:///home/delaunan/code/delaunan/clintrialpredict/frontend/data/search_registry.csv) | Pre-calculated Forensic Database. |

---

## **6. Reference & Audit Tools**
- **Database Refresh**: `python refresh_registry.py` (Re-calculates `search_registry.csv`).
- **Parity Audit**: `python audit_parity.py` (Checks for mathematical drift).
- **Deployment**: `scripts/deploy.sh` (Standard Cloud Run push).
