
import nbformat as nbf

def fix_rogue_newlines():
    file_path = 'notebooks/production_01.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # Correct Step 12
    step12_id = "# --- STEP 12: DETAILED CALIBRATION REPORT ---"
    for cell in nb.cells:
        if step12_id in cell.source:
            cell.source = r"""# --- STEP 12: DETAILED CALIBRATION REPORT ---
print('=== STEP 12: RECENCY-WEIGHTED CALIBRATION (PRODUCTION FINAL) ===')
print(f'>>> Policy: Beta=1.15 | Recency Bias: 0.1 to 1.0')
print(f'🌍 Global Pool Threshold: {global_thresh:.4f} (Logit: {global_logit:.4f})')

audit_log = []
ta_counts = df_cal['TA'].value_counts()
for ta, count in ta_counts.items():
    df_ta = df_cal[df_cal['TA'] == ta]
    weighted_n = df_ta['weight'].sum()
    audit_log.append({
        'TA': ta, 
        'Raw_N': count, 
        'Eff_N': f"{weighted_n:.1f}", 
        'Threshold': f"{final_thresholds.get(ta, global_thresh):.6f}"
    })

print('\n--- CALIBRATION REPORT (WEIGHTED EFF_N) ---')
print(pd.DataFrame(audit_log).to_string(index=False))
print('\n✅ Logic Synced: Production thresholds are now locked for export.')"""

    # Correct Step 13 (Ensuring no rogue newlines in print statements)
    step13_id = "# --- STEP 13: COMPREHENSIVE TA PERFORMANCE ANALYSIS ---"
    for cell in nb.cells:
        if step13_id in cell.source:
            # We already fixed Step 13 in the previous turn, but let's be safe
            if "print('\n" in cell.source:
                cell.source = cell.source.replace("print('\n", "print('\\n")

    with open(file_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Successfully fixed rogue newlines in {file_path}")

if __name__ == "__main__":
    fix_rogue_newlines()
