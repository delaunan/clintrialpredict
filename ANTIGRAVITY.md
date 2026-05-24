# **Clinical Trial Prediction: Project Status & Architecture (v56.0)**

## **1. Production Deployment (v1.0 - "Steel Shield")**
- **Production URL**: [https://clintrial-ui-835962039082.europe-west1.run.app/](https://clintrial-ui-835962039082.europe-west1.run.app/)
- **Infrastructure**: Google Cloud Run (Fully Managed Serverless).
- **Architecture**: Decoupled FastAPI (Scoring Engine) + Streamlit (Professional UI).
- **Analytical Universe**: **34,066 trials** (2009–2026).

---

## **2. Simulation Roadmap (v2.0 - "Strategic Forecaster")**
The v2.0 cycle transforms the platform from a historical discovery engine into an interactive strategic forecasting tool.

### **Phase 1: Manual Feature Override (v2.1)**
- **Feature Console**: A new **Tab 4 (Simulation Workspace)** will be added to the Trial Forensic view.
- **Direct Adjustment**: Users can manually modify the **27 model features** currently visualized in the Treemap.
- **Instant Feedback**: The backend will recalculate the Success Score in real-time as features are adjusted.

### **Phase 2: Intelligent Protocol Mapping (v2.2)**
- **Text-to-Logic**: Users can edit raw trial information (Population, Summary, Criteria).
- **Antigravity One-Shot Strategy**: A specialized agent will parse the edited text and automatically map it to the 27 model features.
- **Verification Loop**: Automated feature updates will be displayed in the Tab 4 console for final user review and manual refinement.

---

## **3. Production Engine (notebooks/production_01.ipynb)**
- **Algorithm**: XGBoost (v1.0) trained on the 17-year historical cohort.
- **Calibration**: Recency-Weighted (0.1 to 1.0 gradient) to focus on modern failure patterns.
- **Performance**: Validated **Recall of 70%** on the "Modern Era" (2021-2022) portfolio.
- **Explainability**: Global SHAP TreeExplainer pre-calculates the "Why" for all 34k trials, stored in `shap_values_01.joblib`.

---

## **4. Deployment Artifacts & Parity**
| Service Layer | Artifact Name | Purpose |
| :--- | :--- | :--- |
| **API** | [model_prod_01.joblib](file:///home/delaunan/code/delaunan/clintrialpredict/models/model_prod_01.joblib) | Full Preprocessor + XGBoost Pipeline. |
| **API** | [thresholds_01.json](file:///home/delaunan/code/delaunan/clintrialpredict/models/thresholds_01.json) | TA-specific logit boundaries & Calibration Offsets. |
| **API** | [shap_values_01.joblib](file:///home/delaunan/code/delaunan/clintrialpredict/models/shap_values_01.joblib) | Pre-calculated SHAP vectors for instant UI response. |
| **API** | [taxonomy_01.json](file:///home/delaunan/code/delaunan/clintrialpredict/models/taxonomy_01.json) | Feature-to-Pillar mapping and UI hierarchies. |
| **UI** | [search_registry.csv](file:///home/delaunan/code/delaunan/clintrialpredict/frontend/data/search_registry.csv) | Enriched database for the AgGrid discovery engine. |

- **Scoring Parity Standard**: 100% mathematical consistency between Notebook and API achieved via **Bottom-Up Rounding** and **Residual Absorption** into the "Therapeutic Context" pillar.
- **Pipeline Generation**: Generate [search_registry.csv](file:///home/delaunan/code/delaunan/clintrialpredict/frontend/data/search_registry.csv) dynamically by running [refresh_registry.py](file:///home/delaunan/code/delaunan/clintrialpredict/refresh_registry.py).
- **Mathematical Audit**: Run [audit_parity.py](file:///home/delaunan/code/delaunan/clintrialpredict/audit_parity.py) to check for score calculation or UI aggregation discrepancies.

---

## **5. Development Workflow (v2-dev)**
To maintain 100% production uptime, all improvements follow the Staging-to-Production pipeline:
1. **Branching**: All v2 features developed on the `v2-dev` branch.
2. **Local Sync**: UI connects to local `uvicorn` (`localhost:8000`) using `.env`.
3. **Staging Services**:
   - `clintrial-api-staging`: Private API endpoint for v2 testing.
   - `clintrial-ui-staging`: Private UI for internal verification.
4. **Deployment**:
   ```bash
   # Deploy Staging API
   gcloud run deploy clintrial-api-staging --source . --command "uvicorn api.main:app --host 0.0.0.0 --port 8080"
   # Deploy Staging UI (pointing to staging API)
   gcloud run deploy clintrial-ui-staging --source . --set-env-vars API_URL=[STAGING_API_URL]
   ```

---

## **6. Agent Operational Guidelines (Antigravity Rules)**

### **Default Behavior**
- Treat repositories as read-only unless the user explicitly approves a specific code edit.
- Do not create, modify, delete, format, rename, or move files unless the user explicitly asks for that exact change.
- Before making any code change, first inspect the relevant files and propose a minimal plan.
- Do not run destructive commands unless explicitly approved.

### **Advisory & Planning Protocols**
- Default to advisory mode: explain, inspect, compare, review risks, and propose options before editing.
- Adhere strictly to the Planning Mode workflow: create/update `implementation_plan.md` and obtain approval before execution.

### **Parity & Regression Safeguards**
- **Before staging or committing any code edits**: If model coefficients, categories, preprocess pipeline scripts (`src/prep/`), or scoring functions are modified:
  1. Re-generate the local Streamlit database: `python refresh_registry.py`
  2. Run the mathematical audit verification: `python audit_parity.py`
  3. The audit must show **100% Perfect Parity** before changes are eligible for Staging deployment.
