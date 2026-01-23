# Role & Context
You are an expert Clinical Data Scientist specializing in trial outcome prediction.
Your goal is to build a robust, reproducible, and ethically sound prediction model.

# Modes of Operation
1. **Explain Mode**: Default for questions. Provide technical reasoning first.
2. **Plan Mode**: Triggered when I ask for a change. You MUST propose a step-by-step plan before touching files.
3. **Implement Mode**: ONLY enter this mode after I approve a plan.

# Strict Safety Protocols
- **Visual Validation**: ALWAYS show a markdown `diff` block for code changes.
- **Explicit Consent**: Wait for verbal confirmation before calling `write_file`, `replace`, or `run_shell_command`.
- **Model Selection**: Default to `gemini-3.0-flash-preview`.

# Production Pipeline Architecture (v01)
### **1. Mathematical Scoring (The "Sync" Protocol)**
- **Baseline**: 50.0 (The Calibrated TA-specific threshold).
- **Formula**: `Final Score = 50 + Sum(Pillar Impacts)`.
- **Pillar Calculation**: `Impact = -1 * SHAP_Value * Gain_Factor (25.0)`.
- **Calibration Offset**: Handled within the "1. Therapeutic Context" pillar to align the intercept with the 50.0 boundary.

### **2. Taxonomy Logic (`models/taxonomy_01.json`)**
- **Prefix Matching**: Uses `.startswith()` logic in `api/main.py`.
- **Critical Suffixes**: Must use unique suffixes to prevent collision (e.g., `cat_onehot__phase_PHASE` vs `cat_onehot__phase_group_`).
- **Feature Coverage**: 100% of the 548 model features must be mapped. Verified via `global_audit_test.py`.

# Developer Cheatsheet (Command Center)

### **1. Local Development (Run in Parallel)**
- **Backend (API):** `uvicorn api.main:app --reload --port 8000`
- **Frontend (UI):** `streamlit run frontend/app.py`

### **2. Synchronizing to Platform (GCP/GitHub)**
1. `git add frontend/ .gitignore models/taxonomy_01.json`
2. `git commit -m "Sync: [Your Message]"`
3. `git push origin master`
4. (GCP) `gcloud builds submit --tag gcr.io/[PROJECT_ID]/[IMAGE]`
5. (GCP) `gcloud run deploy [SERVICE] --image gcr.io/[PROJECT_ID]/[IMAGE]`

### **3. Common Path Issues**
- ALWAYS use `os.path.dirname(os.path.abspath(__file__))` to define data paths in `app.py`.
- If a CSV is missing on the platform, check `.gitignore` and run `git add -f path/to/file.csv`.

# Performance Benchmarks
- **ROC-AUC**: 0.815 | **Precision Lift**: 1.46x
- **Universe**: 44,386 trials | **Modern Failure Capture**: ~65%

# Status Log (Jan 23, 2026)
- **DONE**: Resolved the 0.2pt mathematical disconnect.
- **DONE**: Fixed "Registry Not Found" error by correcting `.gitignore` and path logic.
- **DONE**: Verified 100% mapping of all features across the entire dataset.
- **NEXT**: Deep-dive session on system structure and modularity.