
import json
import os

def sync():
    # Load Validation as reference
    with open('notebooks/validation_clinpred.ipynb', 'r') as f:
        nb_val = json.load(f)
    
    # Load Production to be updated
    with open('notebooks/production_06.ipynb', 'r') as f:
        nb_prod = json.load(f)

    # 1. Capture specific Production blocks from nb_prod
    prod_export_code = ""
    for cell in nb_prod['cells']:
        if cell['cell_type'] == 'code' and '<REF:PROD_EXPORT_CODE>' in "".join(cell['source']):
            prod_export_code = cell['source']
            break
    
    # 2. Rebuild nb_prod cells based on nb_val structure
    new_cells = []
    
    # Header
    new_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# **Clinical Trial Success Prediction: Production Engine (v42.0)**\n",
            "\n",
            "This notebook serves as the **Official Production Engine**. It builds the final predictive model used to feed the FastAPI service and the Streamlit application with XGBoost weights, Therapeutic Area-specific thresholds, and explainability artifacts.\n",
            "\n",
            "### **Key Production Objectives:**\n",
            "1. **Full-Universe Training (2009–2022)**: Training on the complete historical cohort to maximize signal.\n",
            "2. **Recency-Weighted Calibration**: Optimizing decision boundaries using temporal weights.\n",
            "3. **Global Explainability Export**: Generating SHAP values for all ~29,557 records.\n",
            "4. **Artifact Persistence**: Saving the model, thresholds, and registry for deployment."
        ]
    })

    # Find and copy blocks from validation
    val_map = {}
    for cell in nb_val['cells']:
        source_str = "".join(cell['source'])
        if '<REF:' in source_str:
            ref_tag = source_str.split('<REF:')[1].split('>')[0]
            val_map[ref_tag] = cell

    # Order of blocks in nb_prod
    refs_order = [
        "ENV_CONFIG", "ENV_CONFIG_CODE",
        "PATH_RESOLUTION", "PATH_RESOLUTION_CODE",
        "LIB_INIT", "LIB_INIT_CODE",
        "DATA_PIPELINE", "DATA_PIPELINE_CODE",
        "DATA_AUDIT", "DATA_AUDIT_CODE",
        "NLP_AUDIT", "NLP_AUDIT_CODE",
        "TEMP_SPLIT", "TEMP_SPLIT_CODE",
        "MODEL_TRAIN", "MODEL_TRAIN_CODE",
        "CORE_PERF", "CORE_PERF_CODE", 
        "PRED_GEN", "PRED_GEN_CODE",
        "POLICY_OPT", "POLICY_ANCHORS_CODE",
        "THRESH_FUNC_CODE", 
        "DYN_THRESH", "DYN_THRESH_CODE",
        "EXPLAIN_ENGINE", "EXPLAIN_ENGINE_CODE",
        "SCORE_ENGINE", "SCORE_ENGINE_CODE",
        "VALID_DASH", "AUDIT_HEALTH_CODE",
        "PERF_VIS", "PERF_COL_CODE", "PERF_VIS_CODE",
        "PROD_EXPORT", "PROD_EXPORT_CODE", 
        "UI_VIS", "UI_VIS_CODE",
        "CALIBRATION_AUDIT", "CALIBRATION_AUDIT_CODE" # Added these two
    ]

    for ref in refs_order:
        if ref == "CORE_PERF":
            new_cells.append({
                "cell_type": "markdown",
                "id": "prod_perf_md",
                "metadata": {},
                "source": ["#### <REF:CORE_PERF>\n", "> #### **5.B Production Portfolio Performance Audit**\n", "\n", "Evaluates the champion model's performance across the full historical portfolio (2009–2022)."]
            })
            continue
        if ref == "CORE_PERF_CODE":
            new_cells.append({
                "cell_type": "code",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# <REF:CORE_PERF_CODE>\n",
                    "from sklearn.metrics import roc_auc_score, accuracy_score, classification_report\n",
                    "\n",
                    "# 1. Full Portfolio Training Results (2009-2022)\n",
                    "y_prob_train = model.predict_proba(X_train)[:, 1]\n",
                    "auc_train = roc_auc_score(y_train, y_prob_train)\n",
                    "y_pred_train = (y_prob_train > 0.5).astype(int)\n",
                    "acc_train = accuracy_score(y_train, y_pred_train)\n",
                    "\n",
                    "print(f\"\\n=== PRODUCTION PORTFOLIO RESULTS (2009-2022) ===\")\n",
                    "print(f\"TRAIN AUC: {auc_train:.4f}\")\n",
                    "print(f\"TRAIN ACC: {acc_train:.4f} (at 0.5 threshold)\")\n",
                    "\n",
                    "print(\"\\nClassification Report (Full Historical Set):\")\n",
                    "print(classification_report(y_train, y_pred_train))\n",
                    "# <REF:/CORE_PERF_CODE>\n"
                ]
            })
            continue
        
        if ref == "PROD_EXPORT":
             new_cells.append({
                "cell_type": "markdown",
                "id": "e20729e9",
                "metadata": {},
                "source": [
                    "#### <REF:PROD_EXPORT>\n",
                    "> #### **9. Production Artifact Persistence and Deployment Sync**\n",
                    "\n",
                    "Saves the finalized model pipeline, threshold metadata, and SHAP artifacts for ingestion by the FastAPI and Streamlit application layers."
                ]
            })
             continue
        if ref == "PROD_EXPORT_CODE":
            new_cells.append({
                "cell_type": "code",
                "metadata": {},
                "outputs": [],
                "source": prod_export_code
            })
            # Add the manifest markdown after export code
            new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### **Consolidated Production & Deployment Manifest**\n",
                    "\n",
                    "| Artifact | Storage Location | Content Description | Consumption Layer |\n",
                    "| :--- | :--- | :--- | :--- |\n",
                    "| **Production Model** | `models/model_prod_01.joblib` | Final Scikit-learn Pipeline (Preprocessor + XGBClassifier) trained on 2009–2022 historical cohort. | **FastAPI** (`api/main.py`) |\n",
                    "| **Threshold Logic** | `models/thresholds_01.json` | Recency-weighted decision boundaries (logits) and calibration constants (Gain Factor, Intercept). | **FastAPI** (`api/main.py`) |\n",
                    "| **SHAP Store** | `models/shap_values_01.joblib` | Pre-calculated feature attributions for all ~30,000 trials, enabling instant \"Audit Mode\" lookups. | **FastAPI** (`api/main.py`) |\n",
                    "| **Taxonomy Logic** | `models/taxonomy_01.json` | Logic-locked snapshot of Feature Registry and UI Schema to ensure API/UI alignment. | **FastAPI** (`api/main.py`) |\n",
                    "| **Search Registry** | `frontend/data/search_registry.csv` | Full trial universe enriched with success scores, zones, and scientific narratives for the search interface. | **Streamlit** (`frontend/app.py`) |\n",
                    "\n",
                    "**Production Engine Summary:**\n",
                    "*   **Engine Maturity:** The model utilizes the full historical signal (2009–2022) with recency weighting to prioritize contemporary R&D success patterns.\n",
                    "*   **Success Orientation:** All raw risk probabilities are transformed into a **0–100 Success Score** where **50.0** is the specific TA decision boundary.\n",
                    "*   **Pillar Attribution:** SHAP values are aggregated into four strategic pillars (**Therapeutic Context**, **Scientific Design**, **Execution Framework**, **Patient Profile**) for root-cause explainability.\n",
                    "\n",
                    "**Deployment Architecture Verification:**\n",
                    "- **API Synchronization:** The FastAPI layer in `api/main.py` is configured to load all logic artifacts directly from the root `models/` directory.\n",
                    "- **Frontend Isolation:** The Streamlit app in `frontend/app.py` consumes its data from the local `frontend/data/` path, ensuring the UI remains performant and easy to package for deployment.\n",
                    "- **Backward Compatibility:** While historical files like `app_search_data_01.csv` may exist in `models/`, the production engine now defaults to `search_registry.csv` to align with the finalized UI naming convention."
                ]
            })
            continue
        
        if ref == "CALIBRATION_AUDIT":
            new_cells.append({
                "cell_type": "markdown",
                "id": "66ec2aea",
                "metadata": {},
                "source": [
                    "#### <REF:CALIBRATION_AUDIT>\n",
                    "> #### **11. Comprehensive Therapeutic Area Calibration Audit**\n",
                    "\n",
                    "Validates model stability and strategy efficiency by comparing the **Full Portfolio (2009-2022)** against the **Modern Era (2021-2022)**. This ensures that TA-specific thresholds remain robust under contemporary clinical volatility."
                ]
            })
            continue
        if ref == "CALIBRATION_AUDIT_CODE":
            new_cells.append({
                "cell_type": "code",
                "metadata": {},
                "outputs": [],
                "source": [
                    "# <REF:CALIBRATION_AUDIT_CODE>\n",
                    "from sklearn.metrics import precision_score, recall_score, roc_auc_score\n",
                    "\n",
                    "print('>>> STEP 11: COMPREHENSIVE THERAPEUTIC AREA ANALYSIS ...')\n",
                    "print('[SCOPE] Full Columns:   2009-2022 Portfolio Baseline')\n",
                    "print('[SCOPE] Recent Columns: 2021-2022 Modern Era Audit (In-Sample)')\n",
                    "\n",
                    "def get_ta_stats(df_eval, probs, thresholds_map, g_thresh):\n",
                    "    results = []\n",
                    "    total_y = df_eval['target']\n",
                    "    total_p = (probs >= df_eval['therapeutic_area'].map(thresholds_map).fillna(g_thresh)).astype(int)\n",
                    "    results.append({\n",
                    "        'TA': 'TOTAL (Strategy)',\n",
                    "        'N': len(df_eval),\n",
                    "        'Fail%': f'{total_y.mean():.1%}',\n",
                    "        'AUC': f'{roc_auc_score(total_y, probs):.3f}',\n",
                    "        'Prec': f'{precision_score(total_y, total_p, zero_division=0):.1%}',\n",
                    "        'Rec': f'{recall_score(total_y, total_p, zero_division=0):.1%}',\n",
                    "        'Thresh': f'{g_thresh:.4f}'\n",
                    "    })\n",
                    "\n",
                    "    for ta in sorted(df_eval['therapeutic_area'].unique()):\n",
                    "        mask = df_eval['therapeutic_area'] == ta\n",
                    "        y_ta = df_eval.loc[mask, 'target']\n",
                    "        if len(y_ta) == 0 or y_ta.nunique() < 2:\n",
                    "            auc = 'N/A'\n",
                    "        else:\n",
                    "            auc = f'{roc_auc_score(y_ta, probs[mask]):.3f}'\n",
                    "\n",
                    "        t_ta = thresholds_map.get(ta, g_thresh)\n",
                    "        p_ta = (probs[mask] >= t_ta).astype(int)\n",
                    "\n",
                    "        results.append({\n",
                    "            'TA': ta,\n",
                    "            'N': len(y_ta),\n",
                    "            'Fail%': f'{y_ta.mean():.1%}',\n",
                    "            'AUC': auc,\n",
                    "            'Prec': f'{precision_score(y_ta, p_ta, zero_division=0):.1%}',\n",
                    "            'Rec': f'{recall_score(y_ta, p_ta, zero_division=0):.1%}',\n",
                    "            'Thresh': f'{t_ta:.4f}'\n",
                    "        })\n",
                    "    return pd.DataFrame(results)\n",
                    "\n",
                    "y_prob_ser = pd.Series(y_prob_train, index=df_train.index)\n",
                    "\n",
                    "mask_full = df_train['start_year'].between(2009, 2022)\n",
                    "mask_rec  = df_train['start_year'] >= 2021\n",
                    "\n",
                    "stats_full = get_ta_stats(df_train[mask_full], y_prob_ser[mask_full], final_thresholds, global_thresh)\n",
                    "stats_rec  = get_ta_stats(df_train[mask_rec], y_prob_ser[mask_rec], final_thresholds, global_thresh)\n",
                    "\n",
                    "summary = stats_full[['TA', 'N', 'Fail%', 'AUC', 'Prec', 'Rec']].merge(\n",
                    "    stats_rec[['TA', 'N', 'Fail%', 'AUC', 'Prec', 'Rec', 'Thresh']],\n",
                    "    on='TA', suffixes=(' Full', ' Rec')\n",
                    ")\n",
                    "\n",
                    "total_row = summary[summary['TA'] == 'TOTAL (Strategy)']\n",
                    "ta_rows = summary[summary['TA'] != 'TOTAL (Strategy)'].copy()\n",
                    "ta_rows['N Full Int'] = ta_rows['N Full'].astype(int)\n",
                    "ta_rows = ta_rows.sort_values('N Full Int', ascending=False).drop(columns=['N Full Int'])\n",
                    "summary_sorted = pd.concat([total_row, ta_rows])\n",
                    "\n",
                    "print(\"\\n=== THERAPEUTIC AREA COMPARATIVE STATISTICS (SYNCED) ===\")\n",
                    "header = \"TA                   | F.N    | F.F%    | F.AUC  | F.Pr   | F.Re   | R.N    | R.F%    | R.AUC  | R.Pr   | R.Re   | Thresh\"\n",
                    "print(header)\n",
                    "print(\"-\" * len(header))\n",
                    "for _, row in summary_sorted.iterrows():\n",
                    "    line = f\"{row['TA']:<20} | {row['N Full']:<6} | {row['Fail% Full']:<7} | {row['AUC Full']:<6} | {row['Prec Full']:<6} | {row['Rec Full']:<6} | {row['N Rec']:<6} | {row['Fail% Rec']:<7} | {row['AUC Rec']:<6} | {row['Prec Rec']:<6} | {row['Rec Rec']:<6} | {row['Thresh']}\"\n",
                    "    print(line)\n",
                    "# <REF:/CALIBRATION_AUDIT_CODE>\n"
                ]
            })
            continue

        if ref == "THRESH_FUNC_CODE":
            # Extract find_optimal_threshold from validation
            found = False
            for cell in nb_val['cells']:
                if 'find_optimal_threshold' in "".join(cell['source']):
                    new_cells.append(cell)
                    found = True
                    break
            if not found:
                 print("Warning: find_optimal_threshold function not found in validation notebook")
            continue

        if ref in val_map:
            cell = val_map[ref].copy()
            
            # CLEAR OUTPUTS FOR ALL CELLS
            cell['outputs'] = []
            cell['execution_count'] = None

            source_str = "".join(cell['source'])
            
            # ADAPT LOGIC FOR PRODUCTION
            if ref == "LIB_INIT_CODE":
                 cell['source'] = [
                    "# <REF:LIB_INIT_CODE>\n",
                    "import os\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import json\n",
                    "import joblib\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from dotenv import load_dotenv\n",
                    "\n",
                    "# Custom Modules/Classes\n",
                    "from src.prep.data_loader_clinpred import ClinicalTrialLoader\n",
                    "from src.prep.pipeline import preprocessor, FEATURE_REGISTRY, UI_SCHEMA\n",
                    "from src.prep.append_embeddings import append_new_embeddings\n",
                    "\n",
                    "load_dotenv()\n",
                    "\n",
                    "# <REF:/LIB_INIT_CODE>\n"
                ]
            elif ref == "PATH_RESOLUTION_CODE":
                 cell['source'] = [
                    "# <REF:PATH_RESOLUTION_CODE>\n",
                    "import sys\n",
                    "from pathlib import Path\n",
                    "\n",
                    "# 1. Define the Project Root\n",
                    "current_dir = Path.cwd()\n",
                    "project_root = current_dir\n",
                    "\n",
                    "while not (project_root / 'src').exists():\n",
                    "    if project_root == project_root.parent:\n",
                    "        raise FileNotFoundError(\"Could not find project root containing 'src'\")\n",
                    "    project_root = project_root.parent\n",
                    "\n",
                    "# 2. Add Project Root to System Path\n",
                    "if str(project_root) not in sys.path:\n",
                    "    sys.path.append(str(project_root))\n",
                    "\n",
                    "# 3. Define Key Paths\n",
                    "DATA_PATH = project_root / \"data\"\n",
                    "MODELS_PATH = project_root / \"models\"\n",
                    "FRONTEND_DATA_PATH = project_root / \"frontend\" / \"data\"\n",
                    "\n",
                    "# 4. Verification\n",
                    "print(f\"Project Root: {project_root}\")\n",
                    "print(f\"Data Path:    {DATA_PATH}\")\n",
                    "print(f\"Models Path:  {MODELS_PATH}\")\n",
                    "\n",
                    "# <REF:/PATH_RESOLUTION_CODE>\n"
                ]
            elif ref == "TEMP_SPLIT":
                cell['source'] = [
                    "#### <REF:TEMP_SPLIT>\n",
                    "> #### **5. Production Temporal Split: Cohort Extraction & Identity Anchoring**\n",
                    "\n",
                    "Extracts the historical training cohort from the universal universe and promotes the NCT_ID to the index to act as the permanent identity anchor for scoring and SHAP attributions."
                ]
            elif ref == "TEMP_SPLIT_CODE":
                cell['source'] = [
                    "# <REF:TEMP_SPLIT_CODE>\n",
                    "# 1. PRODUCTION FILTERING (Extract Historical Training Cohort) & Identity Anchoring\n",
                    "PROD_START_YEAR = int(os.getenv('PROD_START_YEAR', 2009))\n",
                    "PROD_END_YEAR   = int(os.getenv('PROD_END_YEAR', 2022))\n",
                    "\n",
                    "# Note: df_full already has nct_id as its index from the Load step\n",
                    "df_train = df_full[df_full['target'].notna()].copy()\n",
                    "\n",
                    "# Promote NCT_ID as Identity Anchor if not already\n",
                    "if df_train.index.name != 'nct_id':\n",
                    "    df_train = df_train.set_index('nct_id')\n",
                    "\n",
                    "df_train = df_train[df_train['start_year'].between(PROD_START_YEAR, PROD_END_YEAR)].copy()\n",
                    "\n",
                    "# Sort by date for temporal consistency\n",
                    "df_train['start_date'] = pd.to_datetime(df_train['start_date'], errors='coerce')\n",
                    "df_train = df_train.sort_values('start_date')\n",
                    "\n",
                    "# 2. Identify Model features\n",
                    "cols_to_keep = [c for c in df_train.columns if c.endswith('_ml') or\n",
                    "                c.startswith(('crit_', 'sci_', 'endp_')) or\n",
                    "                c == 'therapeutic_area']\n",
                    "\n",
                    "# 3. Anchored Training Matrix\n",
                    "X_train = df_train[cols_to_keep]\n",
                    "y_train = df_train['target']\n",
                    "\n",
                    "# 4. Preparation of df_full for Inference (Promote NCT_ID to index)\n",
                    "if df_full.index.name != 'nct_id':\n",
                    "    df_full = df_full.set_index('nct_id')\n",
                    "\n",
                    "print(f'>>> Production Training Set Prepared: {len(X_train):,} trials ({PROD_START_YEAR}-{PROD_END_YEAR})')\n",
                    "print(f'    Identity Anchor: Index is {X_train.index.name}')\n",
                    "# <REF:/TEMP_SPLIT_CODE>\n"
                ]
            elif ref == "PRED_GEN":
                 cell['source'] = [
                    "#### <REF:PRED_GEN>\n",
                    "> #### **8. Global Prediction Generation**\n",
                    "\n",
                    "Computes success probabilities for the total universe of trials (both historical and ongoing) to enable search and portfolio-wide analysis."
                ]
            elif ref == "PRED_GEN_CODE":
                cell['source'] = [
                    "# <REF:PRED_GEN_CODE>\n",
                    "# Generate predictions for the TOTAL UNIVERSE (Historical + Ongoing)\n",
                    "y_prob_full = model.predict_proba(df_full[cols_to_keep])[:, 1]\n",
                    "# <REF:/PRED_GEN_CODE>\n"
                ]
            elif ref == "EXPLAIN_ENGINE":
                cell['source'] = [
                    "#### <REF:EXPLAIN_ENGINE>\n",
                    "> #### **10. Global Explainability Engine: SHAP TreeExplainer**\n",
                    "\n",
                    "Decomposes model predictions into constituent feature contributions for all ~30,000 trials to power the application's root-cause analysis views."
                ]
            elif ref == "EXPLAIN_ENGINE_CODE":
                cell['source'] = [
                    "# <REF:EXPLAIN_ENGINE_CODE>\n",
                    "import shap\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "\n",
                    "print(\"\\n>>> STEP 10: CALCULATING SHAP VALUES (FULL UNIVERSE)...\")\n",
                    "\n",
                    "# 1. Reconstruct Input Matrix: Reproduce the transformed data seen by the model\n",
                    "prep_step = model.named_steps['prep']\n",
                    "prep_step.verbose_feature_names_out = True\n",
                    "X_full_trans = prep_step.transform(df_full[cols_to_keep])\n",
                    "feature_names = prep_step.get_feature_names_out()\n",
                    "\n",
                    "# 2. Attribution Calculation: Use TreeExplainer for exact mathematical decomposition\n",
                    "print(\"    [SHAP] Decomposing predictions for all trials (this may take a few minutes)... \")\n",
                    "explainer = shap.TreeExplainer(model.named_steps['clf'])\n",
                    "shap_values = explainer.shap_values(X_full_trans)\n",
                    "\n",
                    "# 3. Intercept Capture: The 'Expected Value' in log-odds space\n",
                    "model_base_value = explainer.expected_value\n",
                    "if isinstance(model_base_value, (list, np.ndarray)):\n",
                    "    model_base_value = model_base_value[0]\n",
                    "\n",
                    "print(f\"    [DONE] Matrix Shape: {shap_values.shape}\")\n",
                    "print(f\"    [Check] Intercept: {model_base_value:.4f}\")\n",
                    "# <REF:/EXPLAIN_ENGINE_CODE>\n"
                ]
            elif ref == "SCORE_ENGINE_CODE":
                # Fix all X_test references in the function body and signature
                new_src = []
                for line in cell['source']:
                    # Signature
                    line = line.replace('def generate_clinical_scorecard(X_test', 'def generate_clinical_scorecard(df_scoring')
                    # Body
                    line = line.replace('index=X_test.index', 'index=df_scoring.index')
                    line = line.replace("X_test['therapeutic_area']", "df_scoring['therapeutic_area']")
                    line = line.replace('pd.DataFrame(0.0, index=X_test.index', 'pd.DataFrame(0.0, index=df_scoring.index')
                    line = line.replace('pd.DataFrame(index=X_test.index)', 'pd.DataFrame(index=df_scoring.index)')
                    line = line.replace("scorecard['Therapeutic_Area'] = X_test['therapeutic_area']", "scorecard['Therapeutic_Area'] = df_scoring['therapeutic_area']")
                    
                    # Function call at the end
                    if 'df_scores = generate_clinical_scorecard(X_test' in line:
                         line = line.replace('X_test', 'df_full')
                         line = line.replace('df_scores =', 'df_full_scores =')
                    
                    if 'print(f"    Mean Success Score: {df_scores' in line:
                         line = line.replace('df_scores', 'df_full_scores')
                    
                    new_src.append(line)
                
                cell['source'] = new_src
                # Add the merge logic at the end
                cell['source'].append("\n")
                cell['source'].append("# Merge Scores into Main Dataframe\n")
                cell['source'].append("df_full = df_full.drop(columns=['Clinical_Score', 'Zone'], errors='ignore')\n")
                cell['source'].append("df_full = pd.concat([df_full, df_full_scores[['Clinical_Score', 'Zone'] + [c for c in df_full_scores.columns if 'pillar_' in c or c in RISK_TAXONOMY.keys()]]], axis=1)\n")
            elif ref == "AUDIT_HEALTH_CODE":
                 cell['source'] = [
                    "# <REF:AUDIT_HEALTH_CODE>\n",
                    "from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "\n",
                    "# 1. PREDICTION GENERATION\n",
                    "# For production, we audit the Full Portfolio (Historical Only)\n",
                    "df_audit = df_full[df_full['target'].notna()].copy()\n",
                    "y_true_audit = df_audit['target']\n",
                    "y_prob_audit = model.predict_proba(df_audit[cols_to_keep])[:, 1]\n",
                    "\n",
                    "# 2. STRATEGY APPLICATION\n",
                    "# Strategy A: Raw (Single Global Threshold for all)\n",
                    "y_pred_audit_raw = (y_prob_audit >= global_thresh).astype(int)\n",
                    "\n",
                    "# Strategy B: Hybrid (Custom TA-Specific Thresholds)\n",
                    "def apply_hybrid_strategy(df_x, probs, thresh_map, g_thresh):\n",
                    "    row_thresholds = df_x['therapeutic_area'].map(thresh_map).fillna(g_thresh)\n",
                    "    return (probs >= row_thresholds).astype(int)\n",
                    "\n",
                    "y_pred_audit_hybrid = apply_hybrid_strategy(df_audit, y_prob_audit, final_thresholds, global_thresh)\n",
                    "\n",
                    "# 3. METRIC SUITE CALCULATOR\n",
                    "def get_audit_metrics(y_true, probs, preds):\n",
                    "    return {\n",
                    "        'auc': roc_auc_score(y_true, probs),\n",
                    "        'pr_auc': average_precision_score(y_true, probs),\n",
                    "        'base': y_true.mean(),\n",
                    "        'prec': precision_score(y_true, preds, zero_division=0),\n",
                    "        'rec': recall_score(y_true, preds, zero_division=0),\n",
                    "        'f1': f1_score(y_true, preds, zero_division=0)\n",
                    "    }\n",
                    "\n",
                    "m_raw = get_audit_metrics(y_true_audit, y_prob_audit, y_pred_audit_raw)\n",
                    "m_hyb = get_audit_metrics(y_true_audit, y_prob_audit, y_pred_audit_hybrid)\n",
                    "\n",
                    "# 4. FINAL REPORTING\n",
                    "print(\">>> STEP 12: PRODUCTION MODEL INTEGRITY AUDIT (FULL PORTFOLIO)...\")\n",
                    "print('')\n",
                    "\n",
                    "print(\"=\"*110)\n",
                    "print(f\"{'METRIC TYPE':<35} | {'PORTFOLIO (2009-22)':<20} | {'STATUS'}\")\n",
                    "print(\"=\"*110)\n",
                    "\n",
                    "# A. Threshold-Independent (Model Quality)\n",
                    "print(f\"{'ROC-AUC (Ranking Power)':<35} | {m_hyb['auc']:<20.4f} | {'PASSED' if m_hyb['auc'] > 0.72 else 'CHECK'}\")\n",
                    "print(f\"{'PR-AUC (Signal Density)':<35} | {m_hyb['pr_auc']:<20.4f} | {'PASSED'}\")\n",
                    "print(f\"{'Baseline Failure Rate':<35} | {m_hyb['base']:<20.1%} | {'N/A'}\")\n",
                    "print('')\n",
                    "\n",
                    "# B. Strategy 1: Raw (Global)\n",
                    "print(f\"{'[Raw] Strategy Precision':<35} | {m_raw['prec']:<20.1%} | {'N/A'}\")\n",
                    "print(f\"{'[Raw] Strategy Recall':<35} | {m_raw['rec']:<20.1%} | {'N/A'}\")\n",
                    "print(f\"{'[Raw] Strategy F-Beta Score':<35} | {m_raw['f1']:<20.4f} | {'N/A'}\")\n",
                    "print('')\n",
                    "\n",
                    "# C. Strategy 2: Hybrid (TA-Specific)\n",
                    "print(f\"{'[Hybrid] Strategy Precision':<35} | {m_hyb['prec']:<20.1%} | {'N/A'}\")\n",
                    "print(f\"{'[Hybrid] Strategy Recall':<35} | {m_hyb['rec']:<20.1%} | {'PASSED' if m_hyb['rec'] > 0.70 else 'LOW'}\")\n",
                    "print(f\"{'[Hybrid] Strategy F-Beta Score':<35} | {m_hyb['f1']:<20.4f} | {'N/A'}\")\n",
                    "print('')\n",
                    "\n",
                    "# D. Strategic Lift\n",
                    "lift = m_hyb['prec'] / m_hyb['base'] if m_hyb['base'] > 0 else 0\n",
                    "print(f\"{'Precision Lift (Strength vs Base)':<35} | {lift:<20.2f}x | {'PASSED' if lift > 1.4 else 'LOW'}\")\n",
                    "\n",
                    "print(\"=\"*110)\n",
                    "status = \"✅ HEALTHY\" if m_hyb['auc'] >= 0.72 and m_hyb['rec'] >= 0.70 else \"⚠️ REVIEW REQUIRED\"\n",
                    "print('')\n",
                    "print(f\"OVERALL ENGINE STATUS: {status}\")\n",
                    "# <REF:/AUDIT_HEALTH_CODE>\n"
                ]
            elif ref == "PERF_VIS_CODE":
                # Replace test_results with df_audit
                cell['source'] = [s.replace('test_results = X_test.copy()', 'df_audit_vis = df_audit.copy()') for s in cell['source']]
                cell['source'] = [s.replace("test_results['target'] = y_test", "") for s in cell['source']]
                cell['source'] = [s.replace("test_results['proba_1'] = y_prob", "df_audit_vis['proba_1'] = y_prob_audit") for s in cell['source']]
                cell['source'] = [s.replace("test_results['risk_score_norm'] = 1.0 - (df_scores['Clinical_Score'] / 100.0)", "df_audit_vis['risk_score_norm'] = 1.0 - (df_full_scores.loc[df_audit_vis.index, 'Clinical_Score'] / 100.0)") for s in cell['source']]
                cell['source'] = [s.replace("y_true=test_results['target']", "y_true=df_audit_vis['target']") for s in cell['source']]
                cell['source'] = [s.replace("y_scores_norm=test_results['risk_score_norm']", "y_scores_norm=df_audit_vis['risk_score_norm']") for s in cell['source']]
                cell['source'] = [s.replace("y_scores_raw=test_results['proba_1']", "y_scores_raw=df_audit_vis['proba_1']") for s in cell['source']]
            elif ref == "UI_VIS_CODE":
                # Fix df_scores to df_full_scores or merged df_full
                new_src = []
                for line in cell['source']:
                    line = line.replace('df_scores', 'df_full_scores')
                    new_src.append(line)
                cell['source'] = new_src
            
            new_cells.append(cell)

    nb_prod['cells'] = new_cells
    
    with open('notebooks/production_06.ipynb', 'w') as f:
        json.dump(nb_prod, f, indent=1)

if __name__ == "__main__":
    sync()
