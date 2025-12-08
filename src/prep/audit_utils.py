import pandas as pd
import numpy as np
import os
import sys

def run_master_audit(data_path):
    """
    Generates a deep audit report with Predictive Power ratings.
    Args:
        data_path (str): The folder containing 'project_data.csv'
    """
    file_path = os.path.join(data_path, 'project_data.csv')
    output_file = os.path.join(data_path, 'audit_features.txt')

    print(f"Reading {file_path}...")
    if not os.path.exists(file_path):
        print(f"CRITICAL: File not found at {file_path}")
        return

    df = pd.read_csv(file_path, low_memory=False)

    with open(output_file, 'w') as f:
        def log(msg):
            # print(msg)
            f.write(msg + "\n")

        def section(title):
            log("\n" + "="*80)
            log(f" {title.upper()}")
            log("="*80)

        def get_power_rating(value, metric_type):
            """Returns a discrete rating based on the metric."""
            # ■ = Very Strong, □ = Strong, (Empty) = Average/Poor

            if metric_type == 'spread': # For Categories (Difference in %)
                if value > 20.0: return "■ VERY STRONG"
                if value > 10.0: return "□ STRONG"
                if value > 5.0:  return "  AVERAGE"
                return "  POOR"

            if metric_type == 'diff': # For Numerics (% Difference)
                if value > 20.0: return "■ VERY STRONG"
                if value > 10.0: return "□ STRONG"
                if value > 5.0:  return "  AVERAGE"
                return "  POOR"

            if metric_type == 'corr': # For Correlation Coefficient
                val = abs(value)
                if val > 0.15: return "■ VERY STRONG"
                if val > 0.05: return "□ STRONG"
                if val > 0.02: return "  AVERAGE"
                return "  POOR"

        # ==============================================================================
        # 1. HIGH LEVEL HEALTH CHECK
        # ==============================================================================
        print("[1/9] Checking Dataset Health...")
        section("1. DATASET HEALTH CHECK")
        log(f"Dimensions: {df.shape[0]} Rows x {df.shape[1]} Columns")

        if 'target' in df.columns:
            counts = df['target'].value_counts()
            global_fail_rate = df['target'].mean()
            ratio = counts.get(0, 0) / counts.get(1, 1) if counts.get(1, 1) > 0 else 0

            log(f"\nTARGET DISTRIBUTION:")
            log(f"   0 (Completed):  {counts.get(0, 0)} ({1-global_fail_rate:.1%})")
            log(f"   1 (Terminated): {counts.get(1, 0)} ({global_fail_rate:.1%})")
            log(f"   Global Failure Baseline: {global_fail_rate*100:.1f}%")
            log(f"   Imbalance Ratio: 1:{ratio:.2f} (Use scale_pos_weight={ratio:.2f})")
        else:
            log("[CRITICAL] 'target' column missing.")
            return

        # ==============================================================================
        # 2. CATEGORICAL FEATURE VALUE (RISK MULTIPLIER)
        # ==============================================================================
        print("[2/9] Analyzing Categorical Risk Signals...")
        section("2. CATEGORICAL PREDICTIVE SIGNAL")
        log("Metric: Risk Spread (Max Failure % - Min Failure %)")
        log("Interpretation: Multiplier > 1.0 = Higher Risk than Baseline.")
        log("-" * 80)

        cat_features = [
            'agent_category', 'sponsor_tier', 'phase',
            'agency_class', 'therapeutic_area', 'has_dmc',
            'is_fda_regulated_drug'
        ]

        # Store results for the leaderboard later
        cat_scores = []

        for col in cat_features:
            if col in df.columns:
                stats = df.groupby(col)['target'].agg(['count', 'mean'])
                stats.rename(columns={'mean': 'failure_rate'}, inplace=True)
                stats = stats[stats['count'] > 50].copy() # Filter noise

                if not stats.empty:
                    stats['failure_pct'] = (stats['failure_rate'] * 100).round(1)
                    stats['risk_multiplier'] = (stats['failure_rate'] / global_fail_rate).round(2)
                    stats = stats.sort_values('failure_rate', ascending=False)

                    spread = (stats['failure_rate'].max() - stats['failure_rate'].min()) * 100
                    rating = get_power_rating(spread, 'spread')

                    # Save for leaderboard
                    cat_scores.append((col, spread, rating))

                    log(f"\n>>> {col} (Spread: {spread:.1f}%) -> {rating}")
                    log(f"{'CATEGORY':<30} | {'COUNT':<6} | {'FAIL %':<6} | {'MULTIPLIER'}")
                    log("-" * 65)
                    for idx, row in stats.iterrows():
                        cat_name = str(idx)[:30]
                        if row['risk_multiplier'] > 1.2:
                            risk_flag = "▲ HIGH"
                        elif row['risk_multiplier'] < 0.8:
                            risk_flag = "▼ LOW "
                        else:
                            risk_flag = "  AVG "

                        log(f"{cat_name:<30} | {int(row['count']):<6} | {row['failure_pct']:<6} | {row['risk_multiplier']:<5} {risk_flag}")
                else:
                    log(f"\n>>> {col} (Not enough data)")
            else:
                log(f"\n[MISSING] {col}")

        # ==============================================================================
        # 3. NUMERICAL FEATURE VALUE (IMPACT ANALYSIS)
        # ==============================================================================
        print("[3/9] Analyzing Numerical Impact...")
        section("3. NUMERICAL PREDICTIVE SIGNAL")
        log("Metric: % Difference between Completed and Terminated trials")
        log("-" * 80)

        num_features = [
            'design_rigor_score', 'eligibility_strictness_score',
            'competition_broad', 'competition_niche',
            'num_primary_endpoints', 'number_of_arms', 'criteria_len_log'
        ]

        log(f"{'FEATURE':<30} | {'AVG(0)':<8} | {'AVG(1)':<8} | {'DIFF %':<8} | {'POWER'}")
        log("-" * 80)

        for col in num_features:
            if col in df.columns:
                means = df.groupby('target')[col].mean()
                avg_0 = means.get(0, 0)
                avg_1 = means.get(1, 0)

                if avg_0 != 0:
                    diff_pct = ((avg_1 - avg_0) / avg_0) * 100
                else:
                    diff_pct = 0

                rating = get_power_rating(abs(diff_pct), 'diff')
                direction = "HIGHER" if diff_pct > 0 else "LOWER"

                log(f"{col:<30} | {avg_0:<8.2f} | {avg_1:<8.2f} | {abs(diff_pct):<6.1f}%  | {rating} ({direction} in Failures)")
            else:
                log(f"{col:<30} | [MISSING]")

        # ==============================================================================
        # 4. PREPROCESSING RECOMMENDATIONS
        # ==============================================================================
        print("[4/9] Generating Preprocessing Strategy...")
        section("4. PREPROCESSING STRATEGY")
        log(f"{'COLUMN':<30} | {'SKEW':<6} | {'UNIQUE':<8} | {'RECOMMENDATION'}")
        log("-" * 80)

        all_cols = df.columns.tolist()
        exclude = ['target', 'nct_id', 'start_date', 'official_title', 'txt_tags', 'txt_criteria', 'why_stopped', 'lead_sponsor', 'sponsor_clean']

        for col in all_cols:
            if col in exclude or col.startswith('emb_'): continue

            dtype = str(df[col].dtype)
            nunique = df[col].nunique()

            if "float" in dtype or "int" in dtype:
                if nunique < 10:
                    rec = "OneHot (Categorical)"
                    skew = "-"
                else:
                    skew_val = df[col].skew()
                    skew = f"{skew_val:.2f}"
                    if abs(skew_val) > 1.0: rec = "Log1p + StandardScaler"
                    else: rec = "StandardScaler"
            else:
                skew = "-"
                if nunique > 50: rec = "TargetEncoder"
                else: rec = "OneHotEncoder"

            log(f"{col:<30} | {skew:<6} | {nunique:<8} | {rec}")

        # ==============================================================================
        # 5. CORRELATION LEADERBOARD (NUMERIC)
        # ==============================================================================
        print("[5/9] Calculating Correlations...")
        section("5. CORRELATION LEADERBOARD (NUMERIC & BOOLEAN)")
        log("Features mathematically most linked to Failure (Target=1)")
        log("-" * 80)

        numeric_df = df.select_dtypes(include=[np.number])
        initial_cols = [c for c in numeric_df.columns if not c.startswith('emb_') and c != 'target']
        cols_to_corr = [c for c in initial_cols if df[c].std() > 0]

        if cols_to_corr:
            correlations = numeric_df[cols_to_corr].corrwith(df['target']).sort_values(key=abs, ascending=False)

            log(f"{'FEATURE':<35} | {'CORR':<8} | {'POWER'}")
            log("-" * 65)
            for feat, corr_val in correlations.head(20).items():
                rating = get_power_rating(corr_val, 'corr')
                log(f"{feat:<35} | {corr_val:<8.3f} | {rating}")
        else:
            log("No numeric columns found for correlation.")

        # ==============================================================================
        # 5b. CATEGORICAL LEADERBOARD (NEW!)
        # ==============================================================================
        print("[6/9] Ranking Categorical Features...")
        section("5b. CATEGORICAL LEADERBOARD (RISK SPREAD)")
        log("Ranking text features by their ability to separate Risk.")
        log("-" * 80)

        # Sort by Spread (Descending)
        cat_scores.sort(key=lambda x: x[1], reverse=True)

        log(f"{'FEATURE':<35} | {'SPREAD':<8} | {'POWER'}")
        log("-" * 65)
        for feat, spread, rating in cat_scores:
            log(f"{feat:<35} | {spread:<8.1f}% | {rating}")

        # ==============================================================================
        # 6. CORRELATION LOSERBOARD
        # ==============================================================================
        section("6. CORRELATION LOSERBOARD (LOW SIGNAL)")
        log("Features with near-zero correlation (< 0.01). Candidates for removal.")
        log("-" * 80)

        if cols_to_corr:
            losers = correlations.abs().sort_values(ascending=True).head(10)
            for feat, corr_val in losers.items():
                original_val = correlations[feat]
                log(f"{feat:<35} | {original_val:<8.4f}")

        # ==============================================================================
        # 7. FEATURE COLLINEARITY
        # ==============================================================================
        print("[7/9] Checking Collinearity...")
        section("7. FEATURE COLLINEARITY (REDUNDANCY CHECK)")
        log("Pairs with Correlation > 0.7 (Risk of Overfitting)")
        log("-" * 80)

        if cols_to_corr:
            corr_matrix = numeric_df[cols_to_corr].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_pairs = [
                (column, index, upper.loc[index, column])
                for index in upper.index
                for column in upper.columns
                if upper.loc[index, column] > 0.7
            ]
            high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

            if high_corr_pairs:
                log(f"{'FEATURE A':<30} <--> {'FEATURE B':<30} | {'CORR'}")
                log("-" * 80)
                for feat_a, feat_b, val in high_corr_pairs:
                    log(f"{feat_a:<30} <--> {feat_b:<30} | {val:.2f}")
            else:
                log("No highly correlated features found.")

        # ==============================================================================
        # 8. BUSINESS LOGIC & RED FLAGS
        # ==============================================================================
        print("[8/9] Running Business Logic Checks...")
        section("8. BUSINESS LOGIC: RED FLAG DETECTOR")
        log("Simulating 'Due Diligence' rules to see how many trials get flagged.")
        log("-" * 80)

        high_risk_agents = ['PI3K_INHIBITOR', 'CHEMOTHERAPY', 'CELL_THERAPY']
        flag_agent = df['agent_category'].isin(high_risk_agents)
        flag_rigor = df['design_rigor_score'] <= 1
        flag_academic = df['agency_class'] == 'OTHER'
        comp_90 = df['competition_broad'].quantile(0.90)
        flag_crowded = df['competition_broad'] > comp_90

        log(f"Total Trials: {len(df)}")
        log(f"• Flagged for High Risk Agent:     {flag_agent.sum()} ({flag_agent.mean():.1%})")
        log(f"• Flagged for Low Design Rigor:    {flag_rigor.sum()} ({flag_rigor.mean():.1%})")
        log(f"• Flagged for Academic Sponsor:    {flag_academic.sum()} ({flag_academic.mean():.1%})")
        log(f"• Flagged for Crowded Market:      {flag_crowded.sum()} ({flag_crowded.mean():.1%})")

        death_zone = flag_academic & flag_rigor & flag_agent
        death_fail_rate = df[death_zone]['target'].mean()

        log(f"\n[THE DEATH ZONE COMBINATION]")
        log("Criteria: Academic Sponsor + Low Rigor + High Risk Agent")
        log(f"Trials in Death Zone: {death_zone.sum()}")
        log(f"Failure Rate in Death Zone: {death_fail_rate:.1%} (Baseline: {global_fail_rate:.1%})")
        if death_fail_rate > global_fail_rate * 1.5:
            log("-> VERDICT: This combination is a massive Red Flag.")
        else:
            log("-> VERDICT: This combination is not significantly riskier.")

        # ==============================================================================
        # 9. FEATURE DEFINITIONS
        # ==============================================================================
        print("[9/9] Writing Documentation...")
        section("9. FEATURE DEFINITIONS & LOGIC")
        log("Formal documentation of calculated features.")
        log("-" * 80)

        definitions = {
            "SCIENTIFIC / PROTOCOL": [
                ("design_rigor_score", "Sum of ordinal scores for Masking (0-3), Allocation (0-1), and Model (0-1). Higher score indicates a more robust, bias-resistant protocol."),
                ("agent_category", "Classification of the intervention molecule (e.g., Biologic, Small Molecule, Gene Therapy) based on regex analysis of drug names and AACT intervention types."),
                ("scientific_success", "Binary flag (1/0). Derived from 'min_p_value'. True if the trial reported at least one Primary Outcome with p <= 0.05. (Analysis only)."),
                ("min_p_value", "The minimum P-value reported across all Primary Outcomes for the trial. Represents the strongest statistical signal observed.")
            ],
            "OPERATIONAL / RECRUITMENT": [
                ("competition_broad", "Count of other trials starting in the same Therapeutic Area within the same year and the previous year. Proxy for market saturation."),
                ("competition_niche", "Count of other trials starting in the same specific Therapeutic Subgroup (e.g., 'Breast Neoplasms') within the same timeframe."),
                ("eligibility_strictness_score", "Sum of binary restrictions: Gender (Not All), Healthy Volunteers (No), Children (Excluded), Elderly (Excluded). Higher score implies harder recruitment."),
                ("criteria_len_log", "Natural log of the character count of the eligibility criteria text. Proxy for protocol complexity and administrative burden.")
            ],
            "REGULATORY / SPONSOR": [
                ("includes_us", "Binary flag. True if the trial lists at least one facility in the United States. Proxy for FDA regulatory oversight."),
                ("is_fda_regulated_drug", "Binary flag from AACT. Indicates if the trial is explicitly flagged as subject to FDA drug regulations."),
                ("sponsor_tier", "Categorical (Tier 1 vs Tier 2). Tier 1 includes top 20 global pharma companies by revenue. Proxy for financial stability and operational resources."),
                ("agency_class", "Categorical (Industry, NIH, Other). Distinguishes between commercial (Industry) and academic/non-profit (Other) sponsorship.")
            ],
            "INTRINSIC RISK": [
                ("has_dmc", "Binary flag. True if the trial has a Data Monitoring Committee. Proxy for trial safety risk or complexity (DMCs are required for high-risk interventions)."),
                ("is_sick_only", "Binary flag. True if 'Healthy Volunteers' is 'No'. Indicates the trial targets a patient population with a specific condition.")
            ]
        }

        for category, items in definitions.items():
            log(f"\n[{category}]")
            for name, desc in items:
                log(f" • {name}:")
                log(f"   {desc}")

    print(f"\nDone. Audit saved to {output_file}")

if __name__ == "__main__":
    current_dir = os.getcwd()
    data_path = os.path.join(current_dir, 'data')
    run_master_audit(data_path)
