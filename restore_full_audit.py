import nbformat as nbf
import os

def restore_comprehensive_audit():
    val_nb = 'notebooks/model_validation.ipynb'
    prod_nb = 'notebooks/production.ipynb'
    
    for path in [val_nb, prod_nb]:
        if not os.path.exists(path): continue
        nb = nbf.read(path, as_version=4)
        is_prod = "production" in path.lower()
        
        # Determine labels based on notebook context
        # Validation: History (OOF 2009-2020) vs Test (2021-2022)
        # Production: History (OOF 2009-2022) - Note: Prod usually holds out 2022 or just OOFs
        hist_label = "History (OOF)"
        test_label = "Modern (2022)" if not is_prod else "Modern Fold"

        new_audit_source = [
            "# <REF:AUDIT_HEALTH_CODE>\n",
            "from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "\n",
            f"print(f\"\\n>>> STEP 8: COMPREHENSIVE FORENSIC AUDIT ({'PRODUCTION' if is_prod else 'VALIDATION'})...")\n",
            "\n",
            "# 1. PREPARE EVALUATION DATASETS\n",
            "# Strategy A: Raw (Global Baseline)\n",
            "# Strategy B: Hybrid (TA-Specific Specialized)\n",
            "\n",
            "# A. Training / History (OOF)\n",
            "df_oof_eval = df_oof.copy()\n",
            "df_oof_eval['pred_raw'] = (df_oof_eval['prob'] >= global_thresh).astype(int)\n",
            "df_oof_eval['pred_hybrid'] = df_oof_eval.apply(lambda x: 1 if x['prob'] >= final_thresholds.get(x['TA'], global_thresh) else 0, axis=1)\n",
            "\n",
            "# B. Test / Modern (2022)\n",
            "df_test_eval = pd.DataFrame({'TA': X_test['therapeutic_area'], 'true': y_test, 'prob': y_prob})\n",
            "df_test_eval['pred_raw'] = (df_test_eval['prob'] >= global_thresh).astype(int)\n",
            "df_test_eval['pred_hybrid'] = df_test_eval.apply(lambda x: 1 if x['prob'] >= final_thresholds.get(x['TA'], global_thresh) else 0, axis=1)\n",
            "\n",
            "def get_metrics(df, pred_col):\n",
            "    return {\n",
            "        'auc': roc_auc_score(df['true'], df['prob']),\n",
            "        'pr_auc': average_precision_score(df['true'], df['prob']),\n",
            "        'prec': precision_score(df['true'], df[pred_col], zero_division=0),\n",
            "        'rec': recall_score(df['true'], df[pred_col], zero_division=0),\n",
            "        'base': df['true'].mean()\n",
            "    }\n",
            "\n",
            "m_hist = get_metrics(df_oof_eval, 'pred_hybrid')\n",
            "m_test_raw = get_metrics(df_test_eval, 'pred_raw')\n",
            "m_test_hybrid = get_metrics(df_test_eval, 'pred_hybrid')\n",
            "\n",
            "print(\"\\n" + "="*100)\n",
            "print("      EXECUTIVE SCORECARD: HYBRID STRATEGY ON MODERN DATA (2022)")\n",
            "print("="*100)\n",
            "scorecard = {\n",
            "    'Metric': ['ROC-AUC (Model Power)', 'PR-AUC (Raw Signal)', 'Precision (Reliability)', 'Recall (Coverage)', 'Precision Lift'],\n",
            "    'Value': [\n",
            "        '{:.4f}'.format(m_test_hybrid['auc']), '{:.4f}'.format(m_test_hybrid['pr_auc']),\n",
            "        '{:.1%}'.format(m_test_hybrid['prec']), '{:.1%}'.format(m_test_hybrid['rec']),\n",
            "        '{:.2f}x'.format(m_test_hybrid['prec'] / m_test_hybrid['base'] if m_test_hybrid['base'] > 0 else 0)\n",
            "    ],\n",
            "    'Baseline': ['0.5000', '{:.4f}'.format(m_test_hybrid['base']), '{:.1%}'.format(m_test_hybrid['base']), '-', '1.00x'],\n",
            "    'Context': ['Ranking Power', 'Signal vs Noise', 'Trust in Score', 'Market Capture', 'Strength vs Random']\n",
            "}\n",
            "print(pd.DataFrame(scorecard).to_string(index=False))\n",
            "\n",
            "print(\"\\n" + ">>> Running Robustness Audit (OOF vs Test Set)...")\n",
            "print("="*100)\n",
            f"print(\"      ROBUSTNESS AUDIT: {hist_label} vs {test_label}\")\n",
            "print("="*100)\n",
            "robustness = {\n",
            "    'Metric': ['Baseline Failure Rate', 'ROC-AUC (Ranking)', 'Hybrid Precision', 'Hybrid Recall'],\n",
            "    'History (OOF)': ['{:.0%}'.format(m_hist['base']), '{:.4f}'.format(m_hist['auc']), '{:.1%}'.format(m_hist['prec']), '{:.1%}'.format(m_hist['rec'])],\n",
            "    'Modern (2022)': ['{:.0%}'.format(m_test_hybrid['base']), '{:.4f}'.format(m_test_hybrid['auc']), '{:.1%}'.format(m_test_hybrid['prec']), '{:.1%}'.format(m_test_hybrid['rec'])],\n",
            "    'Delta': [\n",
            "        '{:+.0%}'.format(m_test_hybrid['base'] - m_hist['base']), ",
            "        '{:+.4f}'.format(m_test_hybrid['auc'] - m_hist['auc']),
            "        '{:+.1%}'.format(m_test_hybrid['prec'] - m_hist['prec']),
            "        '{:+.1%}'.format(m_test_hybrid['rec'] - m_hist['rec'])
            "    ]
            "}
            "print(pd.DataFrame(robustness).to_string(index=False))
            
            "print("\n" + ">>> Head-to-Head Strategy Check (On 2022 Data)...")
            "print("="*100)
            "print("      STRATEGY LIFT: RAW (Global) vs HYBRID (TA-Specific)")
            "print("="*100)
            "strategy = {
            "    'Metric': ['Decision Precision', 'Decision Recall', 'Precision Lift (vs Base)']
            "    'Raw Strategy': ['{:.1%}'.format(m_test_raw['prec']), '{:.1%}'.format(m_test_raw['rec']), '{:.2f}x'.format(m_test_raw['prec']/m_test_raw['base'])]
            "    'Hybrid Strategy': ['{:.1%}'.format(m_test_hybrid['prec']), '{:.1%}'.format(m_test_hybrid['rec']), '{:.2f}x'.format(m_test_hybrid['prec']/m_test_hybrid['base'])]
            "    'Net Gain': ['{:+.1%}'.format(m_test_hybrid['prec'] - m_test_raw['prec']), '{:+.1%}'.format(m_test_hybrid['rec'] - m_test_raw['rec']), '{:+.2f}x'.format((m_test_hybrid['prec']/m_test_hybrid['base']) - (m_test_raw['prec']/m_test_raw['base']))]
            "}
            "print(pd.DataFrame(strategy).to_string(index=False))
            
            "status = "✅ HEALTHY" if m_test_hybrid['auc'] >= 0.74 and (m_test_hybrid['prec'] / m_test_hybrid['base']) >= 1.4 else "⚠️ DRIFT DETECTED"
            "print(f"\nOVERALL ENGINE STATUS: {status}")
            "# <REF:/AUDIT_HEALTH_CODE>"
        ]

        for cell in nb.cells:
            if cell.cell_type == 'code' and "# <REF:AUDIT_HEALTH_CODE>" in cell.source:
                cell.source = "".join(new_audit_source)
        
        nbf.write(nb, path)
        print(f"✅ Restored Comprehensive Audit in {path}")

if __name__ == '__main__':
    restore_comprehensive_audit()
