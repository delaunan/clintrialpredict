import pandas as pd
import numpy as np

def analyze_mathematical_parity(csv_path='frontend/data/search_registry.csv'):
    print(f"Starting Forensic Audit of {csv_path}...\n")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    pillars = [
        "Therapeutic Context", 
        "Scientific Challenge", 
        "Execution Framework", 
        "Patient Profile"
    ]
    
    # Check if all columns exist
    missing = [p for p in pillars + ["Clinical_Score"] if p not in df.columns]
    if missing:
        print(f"Critical Error: Missing columns in CSV: {missing}")
        return

    # 1. ANALYSIS: Gauge (Clinical_Score) vs Sum of Pillars
    # Logic from api/main.py: final_score = 50.0 + sum(pillar_impacts)
    df['calculated_score'] = 50.0 + df[pillars].sum(axis=1)
    df['score_diff'] = (df['Clinical_Score'] - df['calculated_score']).round(2)
    
    # Identify discrepancies
    discrepancies = df[df['score_diff'].abs() > 0.15] # Allowing for small floating point / rounding jitter
    
    total_trials = len(df)
    parity_trials = total_trials - len(discrepancies)
    
    print(f"--- 1. GAUGE vs PILLAR PARITY ---")
    print(f"Definition: Clinical_Score == 50.0 + sum(Pillars)")
    print(f"Total Trials Audited: {total_trials:,}")
    print(f"Trials with Perfect Parity: {parity_trials:,} ({parity_trials/total_trials:.2%})")
    
    if len(discrepancies) > 0:
        print(f"\nWARNING: {len(discrepancies)} trials failed parity check.")
        print("Sample of discrepancies (First 5):")
        print(discrepancies[['nct_id', 'Clinical_Score', 'calculated_score', 'score_diff']].head())
        
        # Analyze distribution of errors
        print("\nError Magnitude Distribution:")
        print(df['score_diff'].value_counts().head(10))
    else:
        print("SUCCESS: All trials show perfect mathematical alignment between Gauge and Pillars.")

    # 2. TREEMAP vs BAR CHART Consistency
    # In plot.py, Treemap uses pillar_impacts for parent nodes and subcat_impacts for leaves.
    # Bar Chart uses pillar_impacts. 
    # Since subcategories aren't in the CSV (they are dynamic SHAP aggregations), 
    # we verify that the Pillar sums (Bar Chart) are what drive the Gauge.
    
    print(f"\n--- 2. UI COMPONENT ALIGNMENT (Theoretical) ---")
    print("Based on @frontend/utils/plot.py and @api/main.py:")
    print("- Treemap Root ('ALL DRIVERS'): Sum(Pillars) [Relative to 0 center]")
    print("- Gauge Value: 50.0 + Sum(Pillars)")
    print("- Impact Bar: Individual Pillar Values")
    
    # Verify Pillar column ranges
    print("\nImpact Ranges per Pillar:")
    for p in pillars:
        print(f"  {p:22}: Min {df[p].min():+5.1f} | Max {df[p].max():+5.1f} | Avg {df[p].mean():+5.2f}")

    # 3. IDENTIFY DRIFT
    # Check if Clinical_Score was calculated with a different intercept or threshold in some cases
    if len(discrepancies) > 0:
        intercept_drift = (df['Clinical_Score'] - df[pillars].sum(axis=1))
        unique_intercepts = intercept_drift.round(1).unique()
        print(f"\nDetected Potential Alternate Intercepts (Expected 50.0):")
        print(unique_intercepts)

if __name__ == "__main__":
    analyze_mathematical_parity()
