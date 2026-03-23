# Clinical Trial Risk Engine: Project Architecture & Logic Map

## 1. Project Vision & Objective
The **ClinTrialPredict** system is a specialized machine learning platform designed to predict the risk of premature clinical trial termination (Terminated/Withdrawn vs. Completed). 

### The "Core Why"
Standard predictive models in this domain often suffer from **Context Blindness**. They might flag an Oncology trial as "High Risk" simply because Oncology trials fail more often than others. This model introduces **Context-Aware Calibration** to distinguish between the *Intrinsic Risk* of a disease and the *Operational/Scientific Risk* of the specific trial protocol.

---

## 2. Data Strategy & Integrity

### A. The Golden Split (Temporal Validation)
To simulate real-world forecasting, the project uses a strict chronological split:
*   **Training Window:** 2009 – 2021.
*   **Production/Test Window:** 2022.
*   **Rationale:** Avoids "Data Leakage" from future success patterns and accounts for temporal drift in clinical trial standards.

### B. "Day-Zero" Sanitization (Leakage Prevention)
Located in `src/prep/text_cleaning.py`, this logic ensures the model only sees information available *at the moment the trial was registered*.
*   **Administrative Purge:** Removes edit trails like "Protocol v2", "Amendment 1", and "Last updated on...".
*   **NCT-ID Removal:** Prevents the model from memorizing specific trial outcomes via their unique identifiers.
*   **Struck-through Content:** Uses regex to delete `<s>` and `<strike>` HTML blocks, which typically contain "future" corrections to the original protocol.
*   **Scientific Preservation:** Protects critical nomenclature (e.g., RECIST, ECOG, HbA1c) to ensure the BioBERT model maintains scientific nuance.

---

## 3. Feature Engineering & Taxonomy

The model organizes its inputs into a hierarchical **Risk Taxonomy** across 5 business pillars:

### I. Intrinsic Risk (The Baseline)
*   **Features:** Therapeutic Area, Indication (Target Encoding), Phase (1, 2, 3), and Agent Category (Small Molecule vs. Biologic/Cell Therapy).
*   **Role:** Defines the "Handicap" or starting risk level for a specific sector.

### II. Scientific Intent (The Hypothesis)
*   **Features:** NLP embeddings (BioBERT) of the Official Title and Brief Summary.
*   **Role:** Analyzes the clarity, novelty, and scientific focus of the rationale.

### III. Trial Design (The Methodology)
*   **Features:** NLP embeddings of Primary Endpoints, number of arms, masking (blinding), and randomized allocation.
*   **Role:** Measures the rigor and complexity of the protocol.

### IV. Patient Profile (The Population)
*   **Features:** NLP embeddings of Inclusion/Exclusion criteria, eligibility strictness scores, and patient acuity flags (Acute, Refractory, Severe).
*   **Role:** Gauges recruitment friction and patient vulnerability.

### V. Operational Context (The Resources)
*   **Features:** Sponsor Tier (Big Pharma vs. Other), market competition (broad and niche), and FDA oversight status.
*   **Role:** Evaluates the experience and environment in which the trial is executed.

---

## 4. The NLP Architecture
The system uses a state-of-the-art NLP pipeline to process unstructured clinical text:
1.  **BioBERT Embeddings:** Generates 768-dimensional vectors for three channels: Scientific, Criteria, and Endpoints.
2.  **PCA Reduction:** To manage the "Curse of Dimensionality," each 768-dim vector is reduced to ~160 principal components, capturing ~90% of the variance while ensuring XGBoost stability.

---

## 5. Advanced Calibration Mechanics

### The "Partial Handicap" Strategy
The project uses a **Dynamic Thresholding** system to ensure fairness across diseases.
*   **Math:** `Threshold_TA = Global_Base_Logit + (Intrinsic_Risk_SHAP * k_factor)`
*   **Logic:** If Oncology trials are inherently twice as likely to fail, we raise the "Failure" threshold for Oncology. This forces the model to only flag Oncology trials that are *even riskier* than the Oncology average.
*   **Safety Latch:** A threshold is never allowed to drop below the global baseline (preventing leniency).

### Score Normalization (The Universal 50)
To provide a consistent UI experience, all TA-specific thresholds are warped into a **Clinical Success Score (0-100)**:
*   **50.0** is the absolute decision boundary for every trial.
*   **> 50 (Blue/Green):** The trial is "Safer" than its peers in that specific therapeutic area.
*   **< 50 (Orange/Red):** The trial carries higher-than-average risk for its context.

---

## 6. Validation & Robustness
The model undergoes a three-part audit (`notebooks/model.ipynb`):
1.  **Executive Scorecard:** Splits "Pure Model Power" (Raw AUC) from "Strategic Value" (Normalized Precision).
2.  **Generalization Audit:** Compares performance on 2009-2021 data vs. 2022 data to detect overfitting.
3.  **Leakage Audit:** A permutation test that shuffles labels to ensure the model isn't relying on hidden structural artifacts.

---

## 7. System Architecture
*   **ETL & Preprocessing:** Python-based pipeline (`data_loader.py` + `preprocessing.py`).
*   **Model:** XGBoost Classifier wrapped in a stateless Scikit-Learn Pipeline.
*   **API:** FastAPI (`api/main.py`) serving predictions and SHAP-based explanations.
*   **Frontend:** Streamlit (`frontend/app.py`) providing an interactive dashboard with Gauges, Treemaps, and Pillar Impact charts.

---
**Status as of Jan 11, 2026:** ETL, preprocessing, model training, and calibration are complete. The project is currently in the "Scoring Robustness" and "UI Integration" phase.
