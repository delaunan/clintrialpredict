import pandas as pd
import numpy as np
import sys

def deep_audit():
    print("# Deep Forensic Data Audit: Logical Integrity & Malfunction Scan")
    df = pd.read_csv('data/data_clinpred.csv', low_memory=False)
    
    # Standardize dates
    df['start_date_dt'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['completion_date_dt'] = pd.to_datetime(df['completion_date'], errors='coerce')
    
    malfunctions = []

    # --- 1. Chronological Malfunctions ---
    # completion_date < start_date
    chrono_err = df[df['completion_date_dt'] < df['start_date_dt']]
    if not chrono_err.empty:
        malfunctions.append(f"[CRITICAL] Chronological Reversal: {len(chrono_err)} trials have Completion Date before Start Date.")

    # --- 2. Identity & Structural Integrity ---
    # Duplicate NCT IDs
    dupes = df[df.duplicated('nct_id')]
    if not dupes.empty:
        malfunctions.append(f"[CRITICAL] Identity Violation: {len(dupes)} duplicate NCT_IDs found.")

    # --- 3. Phase vs. Strategic Intent Malfunctions ---
    # Phase 3 should rarely be "Safety/Dosing"
    phase3_safety = df[(df['phase_ui'] == 'Phase 3') & (df['strategic_ambition_ui'] == 'Early Phase / Dose Finding')]
    if not phase3_safety.empty:
        malfunctions.append(f"[LOGIC] Phase/Intent Mismatch: {len(phase3_safety)} Phase 3 trials categorized as 'Early Phase / Dose Finding'.")

    # --- 4. Physical Modality vs. Complexity Malfunctions ---
    # Biologic Mabs/ADCs/Cell Therapy are almost never "Simple Oral" (with very rare exceptions)
    biological_modalities = ['Biologic Mab', 'Biologic Adc', 'Cell Gene Therapy']
    oral_biologics = df[df['therapeutic_modality_ui'].isin(biological_modalities) & (df['administration_complexity_ui'] == 'Simple Oral (Pill/Tablet)')]
    if not oral_biologics.empty:
        malfunctions.append(f"[LOGIC] Physical Contradiction: {len(oral_biologics)} Biologic/Cell therapies categorized as 'Simple Oral'.")

    # --- 5. GBD Hierarchy Drift ---
    # Each gbd_cause_id_3 (Level 3) must map to exactly ONE gbd_cause_id_2 (Level 2)
    gbd_map = df.groupby('gbd_cause_id_3')['gbd_cause_id_2'].nunique()
    drift_ids = gbd_map[gbd_map > 1]
    if not drift_ids.empty:
        malfunctions.append(f"[STRUCTURAL] GBD Hierarchy Drift: {len(drift_ids)} Level 3 IDs map to multiple Level 2 parents.")

    # --- 6. Duration & Scaling Malfunctions ---
    # Duration > 180 months (Should be capped per v18.5)
    duration_uncapped = df[df['primary_duration_months_ml'] > 180.1] # Allow small epsilon
    if not duration_uncapped.empty:
        malfunctions.append(f"[PROTOCOL] Duration Cap Violation: {len(duration_uncapped)} trials exceed the 180-month cap.")
    
    # Duration <= 0
    duration_zero = df[df['primary_duration_months_ml'] <= 0]
    if not duration_zero.empty:
        malfunctions.append(f"[DATA] Zero Duration: {len(duration_zero)} trials have 0 or negative duration.")

    # --- 7. Sponsor Tier Consistency ---
    # One Canonical Sponsor should have exactly one Tier
    sponsor_tier_drift = df.groupby('lead_sponsor_canonical')['sponsor_tier_ml'].nunique()
    sponsor_errs = sponsor_tier_drift[sponsor_tier_drift > 1]
    if not sponsor_errs.empty:
        malfunctions.append(f"[STRUCTURAL] Sponsor Tier Instability: {len(sponsor_errs)} parent companies assigned multiple Tiers.")

    # --- 8. Target Leakage Scan (Heuristic) ---
    # If 'target' is 1 (Fail) but 'overall_status' is COMPLETED
    target_leak = df[(df['target'] == 1.0) & (df['overall_status'] == 'COMPLETED')]
    if not target_leak.empty:
        malfunctions.append(f"[CRITICAL] Target Label Malfunction: {len(target_leak)} COMPLETED trials labeled as FAILURE.")

    # --- Reporting ---
    if malfunctions:
        print("\n## Malfunction Alerts Detected")
        for m in malfunctions:
            print(f"- {m}")
    else:
        print("\n## Database Status: LOGIC-PURE")
        print("- No structural, physical, or chronological malfunctions detected.")

    print("\n## Detailed Metric Audit")
    print(f"- **Max Duration Found**: {df['primary_duration_months_ml'].max()} months")
    print(f"- **Min Arms Found**: {df['number_of_arms_ml'].min()}")
    print(f"- **Unique Sponsors**: {df['lead_sponsor_canonical'].nunique()}")
    
    # Check for NaNs in critical ML features
    ml_cols = [c for c in df.columns if c.endswith('_ml')]
    nan_report = df[ml_cols].isna().sum()
    nan_report = nan_report[nan_report > 0]
    if not nan_report.empty:
        print("\n## Missingness in ML Core (NaN Count)")
        for col, count in nan_report.items():
            print(f"- {col}: {count}")

if __name__ == "__main__":
    deep_audit()
