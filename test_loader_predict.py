import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.prep.data_loader_predict import ClinicalTrialLoaderPredict

def validate_loader():
    print("--- Initializing Predict Loader ---")
    loader = ClinicalTrialLoaderPredict(data_path='data/')
    
    # 1. Load and Clean
    df_raw = loader.load_and_clean()
    
    print(f"\n--- Post Load & Clean Audit ---")
    print(f"Total Rows: {len(df_raw)}")
    print(f"Segment Distribution:\n{df_raw['trial_segment'].value_counts()}")
    print(f"Status Distribution Sample:\n{df_raw['overall_status'].value_counts().head(10)}")
    print(f"Year Range: {df_raw['start_year'].min()} to {df_raw['start_year'].max()}")
    
    # 2. Add Features (Limited sample for speed if needed, but let's try full first)
    print("\n--- Engineering Features (Enrichment) ---")
    df_feat = loader.add_features(df_raw)
    
    print(f"\n--- Final Feature Audit ---")
    cols_to_check = ['nct_id', 'overall_status', 'trial_segment', 'enrollment', 'number_of_facilities', 'ui_title']
    available_cols = [c for c in cols_to_check if c in df_feat.columns]
    
    print("Sample Data (Enriched):")
    print(df_feat[available_cols].head(10))
    
    print("\nMetadata Reliability Check:")
    print(f"Non-null Enrollment: {df_feat['enrollment'].notna().sum()}")
    print(f"Non-null Facilities: {df_feat['number_of_facilities'].notna().sum()}")
    
    # 3. Check for 2026 specifically
    trials_2026 = df_feat[df_feat['start_year'] == 2026]
    print(f"\nTrials starting in 2026: {len(trials_2026)}")
    if len(trials_2026) > 0:
        print(trials_2026[['nct_id', 'overall_status', 'start_date']].head(5))

if __name__ == "__main__":
    validate_loader()
