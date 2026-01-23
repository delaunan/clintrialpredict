
import nbformat as nbf

def fix_header_and_logic():
    file_path = 'notebooks/production_01.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    target_marker = '# --- STEP 13: COMPREHENSIVE TA PERFORMANCE ANALYSIS ---'
    
    new_source = r"""# --- STEP 13: COMPREHENSIVE TA PERFORMANCE ANALYSIS ---
from sklearn.metrics import precision_score, recall_score, roc_auc_score

print('>>> STEP 13: COMPREHENSIVE THERAPEUTIC AREA ANALYSIS ...')
print('[SCOPE] Historical Columns: 2009-2020 Baseline')
print('[SCOPE] Recent Columns:     2021-2022 Modern Era Audit (In-Sample)')

def get_ta_stats(df_eval, probs, thresholds_map, g_thresh):
    results = []
    total_y = df_eval['target']
    total_p = (probs >= df_eval['therapeutic_area'].map(threshold_map).fillna(g_thresh)).astype(int)
    results.append({
        'TA': 'TOTAL (Strategy)',
        'N': len(df_eval),
        'Fail%': f'{total_y.mean():.1%}',
        'AUC': f'{roc_auc_score(total_y, probs):.3f}',
        'Prec': f'{precision_score(total_y, total_p, zero_division=0):.1%}',
        'Rec': f'{recall_score(total_y, total_p, zero_division=0):.1%}',
        'Thresh': f'{g_thresh:.4f}'
    })

    for ta in sorted(df_eval['therapeutic_area'].unique()):
        mask = df_eval['therapeutic_area'] == ta
        y_ta = df_eval.loc[mask, 'target']
        if len(y_ta) == 0 or y_ta.nunique() < 2:
            auc = 'N/A'
        else:
            auc = f'{roc_auc_score(y_ta, probs[mask]):.3f}'

        t_ta = thresholds_map.get(ta, g_thresh)
        p_ta = (probs[mask] >= t_ta).astype(int)

        results.append({
            'TA': ta,
            'N': len(y_ta),
            'Fail%': f'{y_ta.mean():.1%}',
            'AUC': auc,
            'Prec': f'{precision_score(y_ta, p_ta, zero_division=0):.1%}',
            'Rec': f'{recall_score(y_ta, p_ta, zero_division=0):.1%}',
            'Thresh': f'{t_ta:.4f}'
        })
    return pd.DataFrame(results)

mask_hist = df['start_year'] <= 2020
mask_test = df['start_year'] >= 2021

stats_hist = get_ta_stats(df[mask_hist], y_prob_train[mask_hist], final_thresholds, global_thresh)
stats_rec  = get_ta_stats(df[mask_test], y_prob_train[mask_test], final_thresholds, global_thresh)

# Merge and create final summary
summary = stats_hist[['TA', 'N', 'Fail%', 'AUC', 'Prec', 'Rec']].merge(
    stats_rec[['TA', 'N', 'Fail%', 'AUC', 'Prec', 'Rec', 'Thresh']],
    on='TA', suffixes=(' Hist', ' Rec')
)

total_row = summary[summary['TA'] == 'TOTAL (Strategy)']
ta_rows = summary[summary['TA'] != 'TOTAL (Strategy)'].copy()
ta_rows['N Rec Int'] = ta_rows['N Rec'].astype(int)
ta_rows = ta_rows.sort_values('N Rec Int', ascending=False).drop(columns=['N Rec Int'])
summary_sorted = pd.concat([total_row, ta_rows])

print('\n=== THERAPEUTIC AREA COMPARATIVE STATISTICS (SYNCED) ===')
header = f"{''TA'':<20} | {''H.N'':<6} | {''H.F%'':<7} | {''H.AUC'':<6} | {''H.Pr'':<6} | {''H.Re'':<6} | {''R.N'':<6} | {''R.F%'':<7} | {''R.AUC'':<6} | {''R.Pr'':<6} | {''R.Re'':<6} | {''Thresh''}"
print(header)
print('-' * len(header))
for _, row in summary_sorted.iterrows():
    line = f"{row['TA']:<20} | {row['N Hist']:<6} | {row['Fail% Hist']:<7} | {row['AUC Hist']:<6} | {row['Prec Hist']:<6} | {row['Rec Hist']:<6} | {row['N Rec']:<6} | {row['Fail% Rec']:<7} | {row['AUC Rec']:<6} | {row['Prec Rec']:<6} | {row['Rec Rec']:<6} | {row['Thresh']}"
    print(line)

print('\n>>> RENDERING TOP 6 THERAPEUTIC AREA DEEP DIVES (RESTORED FIDELITY) ...')" ""

    for cell in nb.cells:
        if cell.cell_type == 'code' and target_marker in cell.source:
            cell.source = new_source
            print("Successfully corrected the header logic.")
            break

    with open(file_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    fix_header_and_logic()
