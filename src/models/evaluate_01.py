import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import learning_curve, TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report, RocCurveDisplay, PrecisionRecallDisplay
)

# Silence Scikit-Learn plotting warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Generates a comprehensive performance audit for Clinical Trial Risk Prediction.
    """

    # --- 1. GENERATE PREDICTIONS ---
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # --- 2. CALCULATE CORE METRICS ---
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    baseline_pr = y_test.mean()

    # --- 3. EXECUTIVE SUMMARY REPORT ---
    print(f"\n{'='*80}")
    print(f" PERFORMANCE AUDIT: {model_name}")
    print(f"{'='*80}")

    print(f"\n[1] DISCRIMINATORY POWER (ROC-AUC): {roc_auc:.4f}")
    print(f"    - Industry Benchmark: 0.70 (Public Data) | 0.78+ (Proprietary Data)")
    print(f"    - Status: {'✅ Strong' if roc_auc > 0.75 else '⚠️ Acceptable' if roc_auc > 0.70 else '❌ Weak'}")

    print(f"\n[2] FAILURE DETECTION CAPABILITY (PR-AUC): {pr_auc:.4f}")
    print(f"    - Baseline (Random Guess): {baseline_pr:.4f}")
    print(f"    - Lift: {pr_auc/baseline_pr:.1f}x better than random guessing.")

    print(f"\n[3] PROBABILITY RELIABILITY (Brier Score): {brier:.4f}")
    print(f"    - Goal: < 0.15 for high-confidence financial modeling.")

    print("-" * 80)
    print("DETAILED CLASSIFICATION REPORT (With Interpretation):")

    # --- CUSTOM TABLE FORMATTING ---
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=['Completed (0)', 'Terminated (1)'])

    lines = report_str.split('\n')
    for line in lines:
        if "Completed (0)" in line:
            rec = report_dict['0']['recall']
            print(f"{line.rstrip()}   <-- Correctly approves {rec:.0%} of safe trials")
        elif "Terminated (1)" in line:
            rec = report_dict['1']['recall']
            prec = report_dict['1']['precision']
            print(f"{line.rstrip()}   <-- Catches {rec:.0%} of failures (Precision: {prec:.0%})")
        else:
            print(line)

    # --- 4. VISUALIZATION SUITE ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    plt.suptitle(f"Model Diagnostics: {model_name}", fontsize=20, fontweight='bold', y=1.05)

    # --- PLOT A: CONFUSION MATRIX ---
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    labels = [f"{count}\n({perc:.0%})" for count, perc in zip(cm.flatten(), cm_norm.flatten())]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=axes[0], cbar=False,
                annot_kws={"size": 16, "weight": "bold"})

    axes[0].set_title("CONFUSION MATRIX\n(Recall View: % of Actuals)", fontsize=16, fontweight='bold', pad=15)
    axes[0].set_xlabel("Predicted Label", fontsize=14)
    axes[0].set_ylabel("Actual Label", fontsize=14)
    axes[0].set_xticklabels(['Success', 'Failure'], fontsize=12)
    axes[0].set_yticklabels(['Success', 'Failure'], fontsize=12)

    axes[0].text(0.5, 0.15, "True Negatives\n(Correctly Approved)", ha='center', va='center', color='white', fontsize=11)
    axes[0].text(1.5, 0.15, "False Positives\n(False Alarm)", ha='center', va='center', color='black', fontsize=11)
    axes[0].text(0.5, 1.15, "False Negatives\n(Missed Failure)", ha='center', va='center', color='black', fontsize=11)
    axes[0].text(1.5, 1.15, "True Positives\n(Risk Avoided)", ha='center', va='center', color='white', fontsize=11)

    # --- PLOT B: ROC CURVE ---
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[1], name=model_name, color='#1f77b4', linewidth=3)
    axes[1].set_title(f"ROC CURVE\nAUC = {roc_auc:.4f}", fontsize=16, fontweight='bold', pad=15)
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Random (0.50)")
    axes[1].legend(loc='lower right', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("False Positive Rate", fontsize=14)
    axes[1].set_ylabel("True Positive Rate", fontsize=14)

    # --- PLOT C: PRECISION-RECALL CURVE ---
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=axes[2], name=model_name, color='#ff7f0e', linewidth=3)
    axes[2].set_title(f"PRECISION-RECALL\nAP = {pr_auc:.4f}", fontsize=16, fontweight='bold', pad=15)
    axes[2].plot([0, 1], [baseline_pr, baseline_pr], 'k--', label=f'Baseline ({baseline_pr:.2f})', linewidth=2)
    axes[2].legend(loc='upper right', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel("Recall", fontsize=14)
    axes[2].set_ylabel("Precision", fontsize=14)

    plt.tight_layout()
    plt.show()

    return {'roc_auc': roc_auc, 'pr_auc': pr_auc, 'brier_score': brier}


def plot_learning_curve(model, X_train, y_train, cv=5):
    """
    Generates a Learning Curve using Temporal Cross-Validation.
    Displays two plots:
    1. Standard Learning Curve (Train vs CV Score)
    2. Overfitting Gap Evolution (Difference between Train and CV)
    """
    print(f"\n{'='*80}")
    print(f" DIAGNOSTIC: TEMPORAL LEARNING CURVE")
    print(f"{'='*80}")
    print("Computing learning curve... (This may take a moment)")

    tscv = TimeSeriesSplit(n_splits=cv)

    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=tscv,
        scoring='roc_auc',
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        shuffle=False
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    final_gap = train_mean[-1] - test_mean[-1]

    print(f"\n[RESULTS]")
    print(f"Final Training Score:   {train_mean[-1]:.4f}")
    print(f"Final Validation Score: {test_mean[-1]:.4f}")
    print(f"Generalization Gap:     {final_gap:.4f}")

    if final_gap > 0.10:
        print(">> DIAGNOSIS: High Overfitting. The model is memorizing historical noise.")
    elif final_gap < 0.02 and test_mean[-1] < 0.65:
        print(">> DIAGNOSIS: Underfitting. The model is too simple.")
    else:
        print(">> DIAGNOSIS: Balanced Fit. The model generalizes well.")

    # --- DUAL PLOT SETUP ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # PLOT 1: Standard Learning Curve
    axes[0].plot(train_sizes, train_mean, 'o-', color="#1f77b4", label="Training Score", linewidth=2)
    axes[0].plot(train_sizes, test_mean, 'o-', color="#ff7f0e", label="Validation Score", linewidth=2)
    axes[0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="#1f77b4")
    axes[0].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="#ff7f0e")

    axes[0].set_title("Temporal Learning Curve\n(Performance vs Data Size)", fontsize=16, fontweight='bold')
    axes[0].set_xlabel("Number of Training Samples", fontsize=14)
    axes[0].set_ylabel("ROC-AUC Score", fontsize=14)
    axes[0].legend(loc="best", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # PLOT 2: Gap Evolution (Overfitting Visual)
    gap = train_mean - test_mean
    axes[1].plot(train_sizes, gap, 'o-', color="#d62728", label="Overfitting Gap", linewidth=2)
    axes[1].fill_between(train_sizes, gap, 0, color="#d62728", alpha=0.1)

    axes[1].set_title("Generalization Gap Evolution\n(Lower is Better)", fontsize=16, fontweight='bold')
    axes[1].set_xlabel("Number of Training Samples", fontsize=14)
    axes[1].set_ylabel("Score Difference (Train - CV)", fontsize=14)
    axes[1].axhline(0.10, color='black', linestyle='--', label="Warning Threshold (0.10)")
    axes[1].legend(loc="best", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_business_slices(df_input, target_col='overall_status'):
    """
    Generates a Business Intelligence report.
    SHOWS ALL COMBINATIONS (No Top 10 Filter).
    """
    print(f"\n{'='*80}")
    print(f" 💼 BUSINESS INTELLIGENCE: CROSS-SEGMENT ANALYSIS (FULL LIST)")
    print(f"{'='*80}")

    # --- PART 0: SAFETY COPY & PREPROCESSING ---
    df_results = df_input.copy()

    # Auto-correct text target to numbers if needed
    if df_results[target_col].dtype == 'object':
        df_results[target_col] = df_results[target_col].apply(
            lambda x: 0 if str(x).upper().strip() == 'COMPLETED' else 1
        )

    # --- PART 1: PREPARE DATA ---
    # Create combo column
    if 'agency_class' in df_results.columns and 'phase' in df_results.columns:
        df_results['agency_phase_combo'] = (
            df_results['agency_class'].astype(str) + " / " + df_results['phase'].astype(str)
        )

    potential_segments = ['agency_class', 'phase', 'agency_phase_combo']
    segments = [col for col in potential_segments if col in df_results.columns]

    metrics_list = []

    # --- PART 2: CALCULATE METRICS ---
    for col in segments:
        for group in df_results[col].unique():
            mask = df_results[col] == group
            subset = df_results[mask]

            # LOWERED THRESHOLD: Show everything with at least 5 trials
            if len(subset) < 5: continue

            # 1. Calculate Fail Rate
            fail_rate = subset[target_col].mean()
            count = len(subset)

            # 2. Calculate AUC
            try:
                if len(subset[target_col].unique()) > 1:
                    auc = roc_auc_score(subset[target_col], subset['y_prob'])
                    pr = average_precision_score(subset[target_col], subset['y_prob'])
                else:
                    auc = 0.5
                    pr = 0.0
            except:
                auc = 0.5
                pr = 0.0

            dim_name = "CROSS_SEGMENT" if col == 'agency_phase_combo' else col.upper()

            metrics_list.append({
                'Dimension': dim_name,
                'Segment': group,
                'Count': count,
                'Fail_Rate': fail_rate,
                'ROC_AUC': auc,
                'PR_AUC': pr
            })

    if not metrics_list:
        print("No data found.")
        return

    res_df = pd.DataFrame(metrics_list).sort_values(['Dimension', 'ROC_AUC'], ascending=False)

    # --- PART 3: PRINT TABLES (FULL LIST) ---

    # 1. Main Dimensions
    print("\n[1] MAIN DIMENSIONS (Single Factors)")
    main_dims = res_df[res_df['Dimension'] != 'CROSS_SEGMENT']
    if not main_dims.empty:
        print(main_dims[['Dimension', 'Segment', 'Count', 'Fail_Rate', 'ROC_AUC']].to_string(index=False, formatters={
            'Fail_Rate': '{:.1%}'.format, 'ROC_AUC': '{:.3f}'.format
        }))

    # 2. Cross Segments - SHOWING ALL ROWS
    print("\n[2] CROSS-SEGMENTS (Agency + Phase Interaction) - ALL ROWS")
    cross_dims = res_df[res_df['Dimension'] == 'CROSS_SEGMENT'].sort_values('ROC_AUC', ascending=False)

    if not cross_dims.empty:
        # .to_string() prints the whole dataframe without truncating
        print(cross_dims[['Segment', 'Count', 'Fail_Rate', 'ROC_AUC']].to_string(index=False, formatters={
            'Fail_Rate': '{:.1%}'.format, 'ROC_AUC': '{:.3f}'.format
        }))

    # --- PART 4: VISUALIZATION ---
    # We still limit the plot to Top 20 to prevent it from looking messy,
    # but the print above shows everything.
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    if not cross_dims.empty:
        # Plot Top 20 for readability
        sns.barplot(data=cross_dims.head(20), x='ROC_AUC', y='Segment', ax=axes[0], palette='magma')
        axes[0].axvline(0.70, color='red', linestyle='--', label='Target (0.70)')
        axes[0].set_title("Top 20 Business Segments (By Performance)", fontsize=14, fontweight='bold')
        axes[0].set_xlim(0.4, 1.0)

    # Vintage Plot
    if 'start_year' in df_results.columns:
        years = sorted(df_results['start_year'].unique())
        vintage_scores = []
        for year in years:
            mask = df_results['start_year'] == year
            subset = df_results[mask]
            if len(subset) < 20: continue
            try:
                if len(subset[target_col].unique()) > 1:
                    auc = roc_auc_score(subset[target_col], subset['y_prob'])
                    vintage_scores.append({'Year': int(year), 'ROC_AUC': auc})
            except: pass

        if vintage_scores:
            vintage_df = pd.DataFrame(vintage_scores)
            sns.lineplot(data=vintage_df, x='Year', y='ROC_AUC', marker='o', linewidth=3, ax=axes[1])
            axes[1].set_title("Vintage Analysis: Stability Over Time", fontsize=14, fontweight='bold')
            axes[1].set_ylim(0.5, 1.0)
            axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
