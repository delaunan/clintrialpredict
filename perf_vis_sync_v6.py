# <REF:PERF_COL_CODE>
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc
from scipy.ndimage import gaussian_filter1d
from sklearn.isotonic import IsotonicRegression
from scipy.stats import gaussian_kde

print("\n>>> STEP 13: GLOBAL DATA SCIENCE DASHBOARD (VIBRANT PROFESSIONAL)...")

# --- 1. DEFINE UI COLOR PALETTE (VIBRANT ADJUSTMENT) ---
UI_COLORS = {
    # Red/Failure: Increased saturation, removed "muddy" grey undertones
    'fail_deep': '#A83232',  # Was #912323 (Too dark/brown)
    'fail_soft': '#F0A3A3',  # Was #E6B3B3 (Too dusty)

    # Blue/Success: Increased brightness, looks more "Royal/Tech"
    'succ_deep': '#1C5699',  # Was #144178 (Too black)
    'succ_soft': '#9ACBE8',  # Was #B3D1E6 (Too grey)

    # Neutrals
    'text_main': '#1f2a38',  # Dark Navy/Black
    'grid_grey': '#CFD8DC',
    'bg_white':  '#ffffff'
}

# --- 2. CREATE CUSTOM COLORMAPS ---
cmap_fail = mcolors.LinearSegmentedColormap.from_list("UI_Red", [UI_COLORS['fail_soft'], UI_COLORS['fail_deep']])
cmap_succ = mcolors.LinearSegmentedColormap.from_list("UI_Blue", [UI_COLORS['succ_soft'], UI_COLORS['succ_deep']])
cmap_conf = mcolors.LinearSegmentedColormap.from_list("UI_Conf", ["#ffffff", UI_COLORS['succ_deep']])

# --- 3. GLOBAL SETTINGS ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['text.color'] = UI_COLORS['text_main']
plt.rcParams['axes.labelcolor'] = UI_COLORS['text_main']
plt.rcParams['xtick.color'] = UI_COLORS['text_main']
plt.rcParams['ytick.color'] = UI_COLORS['text_main']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.color'] = UI_COLORS['grid_grey']

ref_line_style = dict(color='#555555', linestyle='--', linewidth=1.5, alpha=0.6)
box_style = dict(boxstyle="round,pad=0.4", fc="white", ec="#555555", lw=1, alpha=0.95)
arrow_style = dict(facecolor=UI_COLORS['text_main'], shrink=0.05, width=1, headwidth=5)

# <REF:/PERF_COL_CODE>

# <REF:PERF_VIS_CODE>
def plot_universal_professional_dashboard(y_true, y_scores_norm, y_scores_raw=None, threshold=0.5):
    """
    Generates the UI-Matched Data Science Dashboard.
    """
    if y_scores_raw is None:
        y_scores_raw = y_scores_norm

    # --- LAYOUT SETUP ---
    fig = plt.figure(figsize=(14, 25), facecolor='white')

    # Global Title
    fig.suptitle("PREMATURE TERMINATION RISK ENGINE",
                 fontsize=18, fontweight='bold', color=UI_COLORS['text_main'], y=0.98)
    fig.text(0.5, 0.955, "Training: 2009-2020 | Test: Clinical Trials with 2022 Start Date (Phase 2/3 Industry)",
             ha='center', fontsize=12, fontweight='bold', color=UI_COLORS['text_main'])

    # Grid Layout
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.25, top=0.92)

    ax1 = fig.add_subplot(gs[0, 0]) # Accuracy
    ax2 = fig.add_subplot(gs[0, 1]) # ROC
    ax3 = fig.add_subplot(gs[1, :]) # Matrix (Full Width)
    ax4 = fig.add_subplot(gs[2, 0]) # ROI
    ax5 = fig.add_subplot(gs[2, 1]) # PR Curve
    ax6 = fig.add_subplot(gs[3, 0]) # Strategy
    ax7 = fig.add_subplot(gs[3, 1]) # Separation

    # ==========================================================================
    # METRICS CALCULATION
    # ==========================================================================

    # 1. HYBRID STRATEGY (The actual Decision Policy)
    y_pred_hyb = (y_scores_norm >= threshold).astype(int)
    cm_hyb = confusion_matrix(y_true, y_pred_hyb)
    tn_h, fp_h, fn_h, tp_h = cm_hyb.ravel()

    rec_hyb = tp_h / (tp_h + fn_h) if (tp_h + fn_h) > 0 else 0
    prec_hyb = tp_h / (tp_h + fp_h) if (tp_h + fp_h) > 0 else 0
    fpr_hyb = fp_h / (tn_h + fp_h) if (tn_h + fp_h) > 0 else 0

    # 2. RAW MODEL FRONTIER (The Intelligence Curve)
    fpr_raw_arr, tpr_raw_arr, _ = roc_curve(y_true, y_scores_raw)
    p_raw_arr, r_raw_arr, _ = precision_recall_curve(y_true, y_scores_raw)

    # To place the star ON the line and keep numbers consistent with Panel 1:
    idx_roc = np.argmin(np.abs(tpr_raw_arr - rec_hyb))
    star_fpr = fpr_raw_arr[idx_roc]
    star_rec = tpr_raw_arr[idx_roc]

    idx_pr = np.argmin(np.abs(r_raw_arr - rec_hyb))
    star_prec = p_raw_arr[idx_pr]

    # ==========================================================================
    # PANEL 1: ACCURACY (Confusion Matrix)
    # ==========================================================================
    cm_perc = cm_hyb.astype('float') / cm_hyb.sum(axis=1)[:, np.newaxis]
    names = [['True Negative', 'False Positive'], ['False Negative', 'True Positive']]

    sns.heatmap(cm_perc, fmt="", cmap=cmap_conf, ax=ax1, cbar=False,
                linewidths=1, linecolor=UI_COLORS['grid_grey'])

    for i in range(2):
        for j in range(2):
            text_col = "white" if cm_perc[i, j] > 0.5 else UI_COLORS['text_main']
            ax1.text(j + 0.5, i + 0.4, f"{cm_hyb[i, j]}",
                     ha='center', va='center', color=text_col, fontsize=12, fontweight='bold')
            ax1.text(j + 0.5, i + 0.5, f"({cm_perc[i, j]:.0%})",
                     ha='center', va='center', color=text_col, fontsize=10, fontweight='bold')
            ax1.text(j + 0.5, i + 0.65, f"{names[i][j]}",
                     ha='center', va='center', color=text_col, fontsize=9)

    ax1.set_title(f"1. ACCURACY: Confusion Matrix", pad=15, color=UI_COLORS['text_main'])
    ax1.set_xlabel("Predicted Class")
    ax1.set_ylabel("Actual Class")
    ax1.set_xticklabels(['Success (0)', 'Failure (1)'])
    ax1.set_yticklabels(['Success (0)', 'Failure (1)'])

    # ==========================================================================
    # PANEL 2: RANKING (ROC Curve)
    # ==========================================================================
    roc_auc = auc(fpr_raw_arr, tpr_raw_arr)
    fpr_grid = np.linspace(0, 1, 1000)
    tpr_smooth = gaussian_filter1d(np.interp(fpr_grid, fpr_raw_arr, tpr_raw_arr), sigma=2)
    tpr_smooth[0], tpr_smooth[-1] = 0.0, 1.0

    ax2.plot([0, 1], [0, 1], label='Random (AUC=50%)', **ref_line_style)
    ax2.plot(fpr_grid, tpr_smooth, color=UI_COLORS['fail_deep'], lw=2.5, label=f'ROC Curve')
    ax2.fill_between(fpr_grid, tpr_smooth, alpha=0.15, color=UI_COLORS['fail_deep'])

    # Star on the line matching Strategy Recall
    ax2.plot(star_fpr, star_rec, marker='*', ms=18, color=UI_COLORS['text_main'],
             zorder=10, markeredgecolor='white')

    # FIXED: Explicit \n inside the f-string for visual line breaks
    note = (fr"Threshold: $\mathbf{{{threshold*100:.0f}}}$%" + "\n"
            fr"False Pos: $\mathbf{{{star_fpr*100:.1f}}}$%" + "\n"
            fr"True Pos: $\mathbf{{{star_rec*100:.1f}}}$%")
    ax2.annotate(note, xy=(star_fpr, star_rec), xytext=(star_fpr+0.1, star_rec-0.25),
                 arrowprops=arrow_style, bbox=box_style, fontsize=9)

    ax2.set_title(f"2. RANKING: Raw Model AUC = {roc_auc:.1%}", pad=15)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate (Recall)")
    ax2.legend(loc="lower right", frameon=True, fontsize=9)
    ax2.grid(True)

    # ==========================================================================
    # PANEL 3: RISK STRATIFICATION (VIBRANT GRADIENTS)
    # ==========================================================================
    df_matrix = pd.DataFrame({'target': y_true, 'prob': y_scores_norm})

    conditions = [
        (df_matrix['prob'] > 0.75),
        (df_matrix['prob'] > 0.50) & (df_matrix['prob'] <= 0.75),
        (df_matrix['prob'] > 0.25) & (df_matrix['prob'] <= 0.50),
        (df_matrix['prob'] <= 0.25)
    ]
    choices = ["1. High Risk", "2. Watchlist", "3. Good", "4. Robust"]
    df_matrix['Zone'] = np.select(conditions, choices, default="4. Robust")

    total_failures = df_matrix[df_matrix['target'] == 1].shape[0]
    total_successes = df_matrix[df_matrix['target'] == 0].shape[0]

    # FIXED: \n for proper headers
    row_labels_map = {
        0: fr"$\mathbf{{Actual\ Success}}$" + f"\n(Total N={total_successes})",
        1: fr"$\mathbf{{Actual\ Failure}}$" + f"\n(Total N={total_failures})"
    }
    df_matrix['target_label'] = df_matrix['target'].map(row_labels_map)

    col_order = choices
    row_order = [
        fr"$\mathbf{{Actual\ Failure}}$" + f"\n(Total N={total_failures})",
        fr"$\mathbf{{Actual\ Success}}$" + f"\n(Total N={total_successes})"
    ]

    matrix_pct = pd.crosstab(index=df_matrix['target_label'], columns=df_matrix['Zone'], normalize='index')[col_order] * 100
    matrix_counts = pd.crosstab(index=df_matrix['target_label'], columns=df_matrix['Zone'])[col_order]

    matrix_pct = matrix_pct.reindex(row_order).fillna(0)
    matrix_counts = matrix_counts.reindex(row_order).fillna(0)

    # FIXED: \n for zone descriptions
    detailed_labels = [
        fr"$\mathbf{{1.\ High\ Risk}}$" + "\n(Score < 25)",
        fr"$\mathbf{{2.\ Watchlist}}$" + "\n(Score 25-50)",
        fr"$\mathbf{{3.\ Good}}$" + "\n(Score 50-75)",
        fr"$\mathbf{{4.\ Robust}}$" + "\n(Score > 75)"
    ]

    rows = matrix_pct.index
    cols = matrix_pct.columns
    n_rows = len(rows)
    n_cols = len(cols)

    ax3.set_xlim(0, n_cols)
    ax3.set_ylim(0, n_rows)

    for j, col_label in enumerate(cols):
        if "High Risk" in col_label or "Watchlist" in col_label:
            current_cmap = cmap_fail
        else:
            current_cmap = cmap_succ

        val_fail = matrix_pct.loc[row_order[0], col_label]
        val_succ = matrix_pct.loc[row_order[1], col_label]
        max_val = max(val_fail, val_succ)
        if max_val == 0: max_val = 1.0

        for i, row_label in enumerate(rows):
            count = int(matrix_counts.loc[row_label, col_label])
            pct_display = matrix_pct.loc[row_label, col_label]

            intensity = pct_display / max_val
            # Start from 0.15 to keep light colors visible but clear
            rect_color = current_cmap(0.15 + (0.85 * intensity))

            rect = patches.Rectangle((j, n_rows - 1 - i), 1, 1, linewidth=1,
                                     edgecolor='white', facecolor=rect_color)
            ax3.add_patch(rect)

            text_color = 'white' if intensity > 0.5 else UI_COLORS['text_main']

            ax3.text(j + 0.5, n_rows - 1 - i + 0.55, f"{pct_display:.1f}%",
                     ha='center', va='center', fontsize=12, fontweight='bold', color=text_color)
            ax3.text(j + 0.5, n_rows - 1 - i + 0.40, f"(n={count})",
                     ha='center', va='center', fontsize=11, fontweight='normal', color=text_color)

    ax3.set_xticks(np.arange(n_cols) + 0.5)
    ax3.set_xticklabels(detailed_labels, fontsize=11)
    ax3.set_yticks(np.arange(n_rows) + 0.5)
    ax3.set_yticklabels(rows[::-1], fontsize=11, rotation=90, va='center')
    ax3.set_title("3. RISK STRATIFICATION: Distribution of Actual Outcomes per Predicted Zone", pad=15)
    ax3.set_xlabel("Predicted Risk Zone (Matches UI Gauge)", labelpad=10)
    for spine in ax3.spines.values():
        spine.set_visible(False)
    ax3.tick_params(length=0)

    # ==========================================================================
    # PANEL 4: ROI
    # ==========================================================================
    data = pd.DataFrame({'target': y_true, 'prob': y_scores_norm}).sort_values('prob', ascending=False)
    data['cum_pop'] = np.arange(1, len(data) + 1) / len(data)
    data['cum_gain'] = data['target'].cumsum() / data['target'].sum()

    ax4.plot(data['cum_pop'], data['cum_gain'], color=UI_COLORS['fail_deep'], lw=2.5, label='AI Capture of Failures')
    ax4.plot([0, 1], [0, 1], label='Random Baseline', **ref_line_style)
    ax4.fill_between(data['cum_pop'], data['cum_pop'], data['cum_gain'], color=UI_COLORS['fail_deep'], alpha=0.1)

    idx_20 = (data['cum_pop'] - 0.20).abs().idxmin()
    gain_20 = data.loc[idx_20, 'cum_gain']
    ax4.plot(0.20, gain_20, 'o', color=UI_COLORS['text_main'], ms=6)

    # FIXED: \n for ROI note
    ax4.annotate(fr"Top 20% Audit" + "\n" + fr"Catches $\mathbf{{{gain_20*100:.1f}}}$% Failures",
                 xy=(0.20, gain_20), xytext=(0.35, gain_20 - 0.15),
                 arrowprops=arrow_style, bbox=box_style, fontsize=9)
    ax4.set_title("4. ROI: Failure Interception Rate", pad=15)
    ax4.set_xlabel("% of Portfolio Audited")
    ax4.set_ylabel("% of Failures Caught")
    ax4.legend(loc="lower right", frameon=True, fontsize=9)
    ax4.grid(True)

    # ==========================================================================
    # PANEL 5: SIGNAL
    # ==========================================================================
    ap = average_precision_score(y_true, y_scores_raw)
    base = y_true.mean()
    r_sort = np.argsort(r_raw_arr)
    p_mono = IsotonicRegression(increasing=False).fit_transform(r_raw_arr[r_sort], p_raw_arr[r_sort])
    r_dense = np.linspace(0, 1, 1000)
    p_smooth = gaussian_filter1d(np.interp(r_dense, r_raw_arr[r_sort], p_mono), sigma=3)
    p_smooth[0], p_smooth[-1] = 1.0, base

    ax5.plot([0, 1], [base, base], label=f'Baseline ({base:.0%})', **ref_line_style)
    ax5.plot(r_dense, p_smooth, color=UI_COLORS['succ_deep'], lw=2.5, label=f'Av.Precision ({ap:.0%})')
    ax5.fill_between(r_dense, p_smooth, base, color=UI_COLORS['succ_deep'], alpha=0.1)

    # Star on the line matching Strategy Recall
    ax5.plot(star_rec, star_prec, marker='*', ms=18, color=UI_COLORS['fail_deep'],
             zorder=10, markeredgecolor='white')

    # FIXED: Explicit \n for PR note
    note_pr = (fr"Threshold: $\mathbf{{{threshold*100:.0f}}}$%" + "\n"
               fr"Recall: $\mathbf{{{star_rec*100:.1f}}}$%" + "\n"
               fr"Precision: $\mathbf{{{star_prec*100:.1f}}}$%")
    ax5.annotate(note_pr, xy=(star_rec, star_prec), xytext=(0.35, 0.65),
                 arrowprops=arrow_style, bbox=box_style, fontsize=9)
    ax5.set_title("5. SIGNAL: Raw Precision-Recall", pad=15)
    ax5.set_xlabel("Recall (Coverage)")
    ax5.set_ylabel("Precision (Reliability)")
    ax5.set_ylim(base-0.05, 1.05)
    ax5.legend(loc="upper right", frameon=True, fontsize=9)
    ax5.grid(True)

    # ==========================================================================
    # PANEL 6: STRATEGY
    # ==========================================================================
    p_trade, r_trade, t_trade = precision_recall_curve(y_true, y_scores_norm)
    p_trade, r_trade = p_trade[:-1], r_trade[:-1]
    t_grid = np.linspace(0, 1, 500)
    p_smooth_line = gaussian_filter1d(np.interp(t_grid, t_trade, IsotonicRegression(increasing=True).fit_transform(t_trade, p_trade)), sigma=10)
    r_smooth_line = gaussian_filter1d(np.interp(t_grid, t_trade, IsotonicRegression(increasing=False).fit_transform(t_trade, r_trade)), sigma=10)
    f1_smooth_line = 2*(p_smooth_line*r_smooth_line)/(p_smooth_line+r_smooth_line+1e-8)

    ax6.plot(t_grid, p_smooth_line, color=UI_COLORS['succ_deep'], linestyle='--', lw=2, label="Precision")
    ax6.plot(t_grid, r_smooth_line, color=UI_COLORS['fail_deep'], linestyle='-', lw=2, label="Recall")
    ax6.plot(t_grid, f1_smooth_line, color=UI_COLORS['text_main'], linestyle='-', lw=2.5, label="F1 Score")
    ax6.axvline(threshold, label=f'Threshold T={threshold:.1%}', **ref_line_style)

    idx_thresh = np.argmin(np.abs(t_grid - threshold))
    f1_at_thresh = f1_smooth_line[idx_thresh]
    ax6.plot(threshold, f1_at_thresh, marker='*', ms=18, color=UI_COLORS['fail_deep'],
             zorder=10, markeredgecolor='white')

    # FIXED: Explicit \n for Strategy note
    strat_text = (fr"Threshold: $\mathbf{{{threshold*100:.0f}}}$%" + "\n"
                  fr"Precision: $\mathbf{{{prec_hyb*100:.1f}}}$%" + "\n"
                  fr"Recall: $\mathbf{{{rec_hyb*100:.1f}}}$%")
    ax6.annotate(strat_text, xy=(threshold, f1_at_thresh), xytext=(0.62, 0.85),
                 arrowprops=arrow_style, bbox=box_style, fontsize=9)
    ax6.set_title("6. STRATEGY: Trade-off Analysis", pad=15)
    ax6.set_xlabel("Normalized Risk Score (0-1)")
    ax6.set_ylabel("Metric Score (0-1)")
    ax6.legend(loc="lower left", frameon=True, fontsize=9)
    ax6.grid(True)
    ax6.set_xlim(0, 1.0)

    # ==========================================================================
    # PANEL 7: SEPARATION
    # ==========================================================================
    x_grid = np.linspace(0, 1, 500)
    kde_succ = gaussian_kde(y_scores_norm[y_true == 0])
    kde_fail = gaussian_kde(y_scores_norm[y_true == 1])
    y_succ_balanced = kde_succ(x_grid)
    y_fail_balanced = kde_fail(x_grid)

    ax7.plot(x_grid, y_succ_balanced, color=UI_COLORS['succ_deep'], alpha=0.9, lw=2, label='Actual Success')
    ax7.fill_between(x_grid, 0, y_succ_balanced, color=UI_COLORS['succ_deep'], alpha=0.1)
    ax7.plot(x_grid, y_fail_balanced, color=UI_COLORS['fail_deep'], alpha=0.9, lw=2, label='Actual Failure')
    ax7.fill_between(x_grid, 0, y_fail_balanced, where=(x_grid >= threshold),
                     color=UI_COLORS['fail_deep'], alpha=0.4, label='Recall Zone (Caught)')
    ax7.fill_between(x_grid, 0, y_fail_balanced, where=(x_grid < threshold),
                     color=UI_COLORS['fail_deep'], alpha=0.05)
    ax7.axvline(threshold, **ref_line_style)

    max_density = max(y_succ_balanced.max(), y_fail_balanced.max())
    ax7.set_ylim(0, max_density * 1.3)

    # FIXED: \n for Separation note
    ax7.annotate(fr"Recall Zone" + "\n" + fr"(Catches $\mathbf{{{rec_hyb*100:.1f}}}$% of Failures)",
                        xy=(threshold + 0.05, y_fail_balanced[np.argmin(np.abs(x_grid - threshold))]),
                        xytext=(0.58, max_density * 1.1),
                        arrowprops=dict(facecolor=UI_COLORS['text_main'], shrink=0.05, width=1.5, headwidth=6),
                        bbox=box_style, fontsize=9)
    ax7.set_title("7. SEPARATION: Normalized Risk Distribution", pad=15)
    ax7.set_xlabel("Normalized Risk Score (0 = Safe, 1 = High Risk)")
    ax7.set_ylabel("Density (Normalized per Class)")
    ax7.legend(loc="upper left", frameon=True, fontsize=9)
    ax7.grid(True)
    ax7.set_xlim(0, 1.0)

    plt.show()

# --- EXECUTE ---
print(">>> Rendering Production Performance Audit Dashboard...")
# --- SYNC LOGIC: Forced Data Refresh ---
y_prob_prod = model.predict_proba(df_train[cols_to_keep])[:, 1]
risk_score_prod = 1.0 - (df_full_scores.loc[df_train.index, 'Clinical_Score'] / 100.0)

plot_universal_professional_dashboard(
    y_true=df_train['target'],
    y_scores_norm=risk_score_prod,
    y_scores_raw=y_prob_prod,
    threshold=0.5
)
# <REF:/PERF_VIS_CODE>
