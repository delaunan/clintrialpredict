import pandas as pd
import numpy as np
import os
import sys

def run_master_audit(data_path):
    """
    Generates a deep audit report with console mirroring and error trapping.
    Ensures that even if the file write fails, the output appears in the terminal.
    """
    file_path = os.path.join(data_path, 'project_data.csv')
    output_file = os.path.join(data_path, 'audit_features.txt')

    if not os.path.exists(file_path):
        print(f"CRITICAL: File not found at {file_path}")
        return

    print(f">>> Loading data from {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)

    if df.empty:
        print("CRITICAL: Dataframe is empty. Check your data loader.")
        return

    # Open file with explicit encoding and buffering
    with open(output_file, 'w', encoding='utf-8', buffering=1) as f:
        def log(msg):
            # Write to file
            f.write(msg + "\n")
            # Mirror to console for immediate debugging
            print(msg)

        def section(title):
            log("\n" + "="*80)
            log(f" {title.upper()}")
            log("="*80)

        def get_power_rating(value, metric_type):
            val = abs(value)
            if metric_type in ['spread', 'diff']:
                if val > 20.0: return "■ VERY STRONG"
                if val > 10.0: return "□ STRONG"
                if val > 5.0:  return "  AVERAGE"
            if metric_type == 'corr':
                if val > 0.85: return "!!! LEAKAGE ALERT !!!"
                if val > 0.15: return "■ VERY STRONG"
                if val > 0.05: return "□ STRONG"
            return "  POOR/WEAK"

        try:
            # ==============================================================================
            # 1. DATASET HEALTH
            # ==============================================================================
            section("1. DATASET HEALTH & INTEGRITY")
            log(f"Dimensions: {df.shape[0]} Rows x {df.shape[1]} Columns")

            if 'target' not in df.columns:
                log("CRITICAL ERROR: 'target' column missing from project_data.csv")
                return

            global_fail_rate = df['target'].mean()
            log(f"Global Failure Baseline: {global_fail_rate*100:.1f}%")

            # Fallback Effectiveness
            if 'therapeutic_subgroup_name' in df.columns:
                unclass_count = df['therapeutic_subgroup_name'].isin(['Unclassified Condition', 'nan', 'UNKNOWN']).sum()
                unclass_pct = (unclass_count / len(df)) * 100
                log(f"Therapeutic Fallback: {unclass_pct:.1f}% remain Unclassified")

            # ==============================================================================
            # 2. CATEGORICAL SIGNALS
            # ==============================================================================
            section("2. CATEGORICAL PREDICTIVE SIGNAL")
            cat_features = ['agent_category', 'phase', 'sponsor_tier', 'is_fda_regulated_drug', 'is_sick_only']

            for col in cat_features:
                if col in df.columns:
                    stats = df.groupby(col)['target'].agg(['count', 'mean'])
                    stats = stats[stats['count'] > 5].copy()
                    if not stats.empty:
                        spread = (stats['mean'].max() - stats['mean'].min()) * 100
                        log(f"\n>>> {col} (Spread: {spread:.1f}%) -> {get_power_rating(spread, 'spread')}")
                        for idx, row in stats.sort_values('mean', ascending=False).iterrows():
                            log(f"{str(idx)[:30]:<30} | {int(row['count']):<6} | {row['mean']*100:.1f}%")

            # ==============================================================================
            # 3. TEMPORAL GATING
            # ==============================================================================
            section("3. TEMPORAL STABILITY & GATING")
            if 'start_year' in df.columns and 'is_fda_regulated_drug' in df.columns:
                pre_2017 = df[df['start_year'] < 2017]['is_fda_regulated_drug'].mean() * 100
                post_2017 = df[df['start_year'] >= 2017]['is_fda_regulated_drug'].mean() * 100
                log(f"FDA Regulated Rate (Pre-2017):  {pre_2017:.1f}% (Target: 0.0%)")
                log(f"FDA Regulated Rate (Post-2017): {post_2017:.1f}%")

            # ==============================================================================
            # 4. CORRELATION (VARIANCE PROTECTED)
            # ==============================================================================
            section("4. CORRELATION LEADERBOARD")
            numeric_df = df.select_dtypes(include=[np.number]).copy()
            # Drop columns with zero variance
            numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]

            if 'target' in numeric_df.columns:
                corrs = numeric_df.corrwith(df['target']).dropna().sort_values(key=abs, ascending=False)
                for feat, val in corrs.head(15).items():
                    if feat != 'target':
                        log(f"{feat:<35} | Corr: {val:>6.3f} | {get_power_rating(val, 'corr')}")

            # ==============================================================================
            # 5. RISK ZONES
            # ==============================================================================
            section("5. BUSINESS LOGIC: RISK ZONES")
            if 'competition_broad' in df.columns and 'design_rigor_score' in df.columns:
                q80 = df['competition_broad'].quantile(0.8)
                death_zone = (df['competition_broad'] > q80) & (df['design_rigor_score'] <= 1)

                log(f"Trials in High-Competition/Low-Rigor Zone: {death_zone.sum()}")
                if death_zone.any():
                    log(f"Death Zone Failure Rate: {df[death_zone]['target'].mean()*100:.1f}%")

        except Exception as e:
            log(f"\nCRITICAL ERROR DURING AUDIT: {str(e)}")
            import traceback
            log(traceback.format_exc())

    print(f"\n>>> Audit process finished. Check the output above and the file at: {output_file}")
