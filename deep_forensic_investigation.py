import pandas as pd
import numpy as np

def deep_forensic_investigation():
    df = pd.read_csv('data/data_clinpred.csv', low_memory=False)
    df['start_date_dt'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['completion_date_dt'] = pd.to_datetime(df['completion_date'], errors='coerce')

    print("# Deep Forensic Investigation: Root Cause Analysis\n")

    # --- 1. Chronological Reversal (4 trials) ---
    print("## 1. Chronological Reversal (Sample)")
    chrono_err = df[df['completion_date_dt'] < df['start_date_dt']]
    cols_to_show = ['nct_id', 'start_date', 'completion_date', 'overall_status', 'phase_ui']
    print(chrono_err[cols_to_show])
    print("\n")

    # --- 2. Phase/Intent Mismatch (Sample 5) ---
    print("## 2. Phase/Intent Mismatch (Sample 5)")
    mismatch = df[(df['phase_ui'] == 'Phase 3') & (df['strategic_ambition_ui'] == 'Early Phase / Dose Finding')]
    cols_mismatch = ['nct_id', 'brief_title', 'phase_ui', 'strategic_ambition_ui', 'strategic_ambition']
    print(mismatch[cols_mismatch].head(5))
    print("\n")

    # --- 3. Physical Contradiction (Sample 5) ---
    print("## 3. Physical Contradiction (Sample 5)")
    biological_modalities = ['Biologic Mab', 'Biologic Adc', 'Cell Gene Therapy']
    contradiction = df[df['therapeutic_modality_ui'].isin(biological_modalities) & (df['administration_complexity_ui'] == 'Simple Oral (Pill/Tablet)')]
    cols_contra = ['nct_id', 'alpha_drug_name', 'therapeutic_modality_ui', 'administration_complexity_ui', 'administration_complexity']
    print(contradiction[cols_contra].head(5))
    print("\n")

    # --- 4. Zero Duration (Sample 5) ---
    print("## 4. Zero Duration (Sample 5)")
    zero_dur = df[df['primary_duration_months_ml'] <= 0]
    cols_zero = ['nct_id', 'start_date', 'completion_date', 'primary_duration_months_ml', 'overall_status']
    print(zero_dur[cols_zero].head(5))
    print("\n")

    # --- 5. Sponsor Tier Instability (Sample 3 companies) ---
    print("## 5. Sponsor Tier Instability (Sample 3 Companies)")
    sponsor_tier_counts = df.groupby('lead_sponsor_canonical')['sponsor_tier_ml'].nunique()
    unstable_sponsors = sponsor_tier_counts[sponsor_tier_counts > 1].index[:3]
    for sponsor in unstable_sponsors:
        subset = df[df['lead_sponsor_canonical'] == sponsor]
        print(f"### Sponsor: {sponsor}")
        print(subset[['nct_id', 'lead_sponsor', 'sponsor_tier_ui', 'sponsor_tier_ml']].drop_duplicates(['sponsor_tier_ui']))
        print("\n")

    # --- 6. GBD Hierarchy Drift (The Specific ID) ---
    print("## 6. GBD Hierarchy Drift")
    gbd_map = df.groupby('gbd_cause_id_3')['gbd_cause_id_2'].nunique()
    drift_ids = gbd_map[gbd_map > 1].index
    for drift_id in drift_ids:
        subset = df[df['gbd_cause_id_3'] == drift_id]
        print(f"### GBD Level 3 ID: {drift_id}")
        print(subset[['gbd_indication_name_3', 'gbd_cause_id_2', 'gbd_indication_name_2']].drop_duplicates())

if __name__ == "__main__":
    deep_forensic_investigation()
