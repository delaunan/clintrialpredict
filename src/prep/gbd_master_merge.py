import pandas as pd
import numpy as np
import os
import json
import sys
from pathlib import Path

def gbd_master_merge():
    """
    Production script version of notebooks/llm_in_00_create_hier.ipynb.
    Integrates IHME GBD metrics with the clinical hierarchy and generates the LLM menu.
    """
    # [STEP 1] Path Resolution
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    DATA_PATH = PROJECT_ROOT / "data"
    REF_PATH  = DATA_PATH / "reference"
    PROC_PATH = DATA_PATH / "processed"
    DOCS_PATH = PROJECT_ROOT / "docs"

    print(">>> Initializing GBD Master Merge Pipeline (v5.1)...")

    # [STEP 2] Load Hierarchy and Metrics
    hierarchy_path = REF_PATH / "hier_gbd.csv"
    metrics_path = REF_PATH / "IHME-GBD_2023_ALL.csv"
    json_path = REF_PATH / "gbd_descriptions.json"

    if not all(p.exists() for p in [hierarchy_path, metrics_path]):
        print(f"[!] ERROR: Required files missing in {REF_PATH}")
        return

    df_hier = pd.read_csv(hierarchy_path)
    df_metrics = pd.read_csv(metrics_path)

    df_hier['Cause ID'] = df_hier['Cause ID'].astype(int)
    df_metrics['cause_id'] = df_metrics['cause_id'].astype(int)

    # [STEP 3] Prepare Metrics Pivot (Feature Vector)
    target_measures = {
        'DALYs (Disability-Adjusted Life Years)': 'daly',
        'YLDs (Years Lived with Disability)': 'yld',
        'YLLs (Years of Life Lost)': 'yll'
    }
    target_locations = {
        'Global': 'global',
        'High SDI': 'high_income'
    }

    df_filtered = df_metrics[
        df_metrics['measure_name'].isin(target_measures.keys()) &
        df_metrics['location_name'].isin(target_locations.keys()) &
        (df_metrics['age_name'] == 'All ages') &
        (df_metrics['metric_name'] == 'Rate')
    ].copy()

    df_filtered['measure_key'] = df_filtered['measure_name'].map(target_measures)
    df_filtered['location_key'] = df_filtered['location_name'].map(target_locations)
    df_filtered['final_col'] = df_filtered['measure_key'] + "_" + df_filtered['location_key']

    df_pivot = df_filtered.pivot_table(index='cause_id', columns='final_col', values='val', aggfunc='mean').reset_index()

    # [STEP 4] Master Merge
    df_master = pd.merge(df_hier, df_pivot, left_on='Cause ID', right_on='cause_id', how='left')
    if 'cause_id' in df_master.columns:
        df_master.drop(columns=['cause_id'], inplace=True)

    # [STEP 5] Cascading Data Inheritance (Bulletproof Fallback v6.0)
    metric_groups = ['global', 'high_income']
    base_metrics = ['daly', 'yld', 'yll']

    # Triplet Inheritance (Level 1 -> 4)
    for level in [1, 2, 3, 4]:
        for suffix in metric_groups:
            cols = [f'{m}_{suffix}' for m in base_metrics]
            missing_mask = (df_master['Level'] == level) & (df_master[f'daly_{suffix}'].isna())
            if missing_mask.any():
                for idx in df_master[missing_mask].index:
                    parent_id = df_master.loc[idx, 'Parent ID']
                    parent_row = df_master[df_master['Cause ID'] == parent_id]
                    if not parent_row.empty and not parent_row[f'daly_{suffix}'].isna().all():
                        for col in cols:
                            df_master.loc[idx, col] = parent_row[col].values[0]

    # Clinical Overrides & Identity Reconciliation
    for suffix in metric_groups:
        df_master.loc[df_master['YLL Only'] == 'X', f'yld_{suffix}'] = 0.0
        df_master.loc[df_master['YLD Only'] == 'X', f'yll_{suffix}'] = 0.0
        
        # Fill mean for absolute gaps
        for m in base_metrics:
            col = f'{m}_{suffix}'
            df_master[col] = df_master[col].fillna(df_master[col].mean())
        
        # Enforce DALY = YLL + YLD
        df_master[f'daly_{suffix}'] = df_master[f'yld_{suffix}'] + df_master[f'yll_{suffix}']

    # [STEP 6] Valuation Ratios
    epsilon = 1e-10
    df_master['chronic_ratio_global'] = (df_master['yld_global'] / (df_master['daly_global'] + epsilon)).clip(0, 1)
    df_master['market_skew_index'] = (df_master['daly_high_income'] / (df_master['daly_global'] + epsilon))

    # [STEP 7] Therapeutic Area Mapping
    mapping_path = REF_PATH / "gbd_ta_mapping.csv"
    if mapping_path.exists():
        df_ta_map = pd.read_csv(mapping_path)
        ta_prefixes = df_ta_map.set_index('GBD Anchor Outline')['Model TA'].to_dict()
        sorted_prefixes = sorted(ta_prefixes.keys(), key=len, reverse=True)

        def assign_model_ta(outline):
            if pd.isna(outline): return "Other/Unclassified"
            outline_str = str(outline).strip()
            for prefix in sorted_prefixes:
                if outline_str == prefix or outline_str.startswith(prefix + "."):
                    return ta_prefixes[prefix]
            return "Other/Unclassified"
        
        df_master['model_ta'] = df_master['Cause Outline'].apply(assign_model_ta)
    else:
        df_master['model_ta'] = "Other/Unclassified"

    # [STEP 8] Export Reference
    output_file = REF_PATH / "gbd_stats.csv"
    df_master.to_csv(output_file, index=False)
    print(f"> SUCCESS: GBD Reference created at {output_file}")

    # [STEP 9] LLM Menu Generation
    NL = chr(10)
    menu_output = DOCS_PATH / 'prompts' / 'gbd_codes.md'
    gbd_map = {}
    if json_path.exists():
        with open(json_path, 'r') as f:
            gbd_map = json.load(f)

    # Filter for L2, L3 and L4 for the menu
    menu_df = df_master[df_master['Level'].isin([2, 3, 4])].sort_values(['model_ta', 'Sort Order'])

    lines = [
        f"# **IHME GBD 2021: Hierarchical Indication Menu (v02)**{NL}",
        f"Use this list to map trials to the most granular ID possible.{NL}",
        f"**Logic**: Find L4 match first -> Fallback to L3 -> Fallback to the Group's [L2 Safety Net] ID -> Final Fallback [ID: 0].{NL}",
        f"{NL}---{NL}{NL}"
    ]

    # Group by Model TA
    for ta, group in menu_df.groupby('model_ta'):
        lines.append(f"### **GROUP: {ta}**{NL}")

        # Identify the L2 Safety Net for this group
        l2_rows = group[group['Level'] == 2]
        if not l2_rows.empty:
            for _, l2 in l2_rows.iterrows():
                cid = str(int(l2['Cause ID']))
                name = l2['Cause Name']
                desc = gbd_map.get(cid, f"Broad safety net category for {ta}. Use if no specific L3/L4 match exists below.")
                lines.append(f"**[Safety Net] [L2] [ID: {cid}] {name}** | {desc}{NL}{NL}")

        # Now list L3 and L4
        sub_group = group[group['Level'].isin([3, 4])]
        for _, row in sub_group.iterrows():
            lvl = row['Level']
            cid = str(int(row['Cause ID']))
            name = row['Cause Name']
            desc = gbd_map.get(cid, "No clinical description available.")

            # Indentation logic: L3 is root, L4 is nested
            prefix = "- " if lvl == 3 else "  - "
            lines.append(f"{prefix}[L{lvl}] [ID: {cid}] {name} | {desc}{NL}")

        lines.append(f"{NL}---{NL}{NL}")

    # Final Fallback
    lines.append(f"### **GROUP: UNKNOWN / OTHER**{NL}")
    lines.append(f"- [L3] [ID: 0] UNKNOWN / OTHER | Catch-all for indications not otherwise listed or scientific areas outside the GBD hierarchy.{NL}")

    with open(menu_output, 'w') as f:
        f.writelines(lines)
    print(f"> SUCCESS: LLM Menu generated at {menu_output}")

if __name__ == "__main__":
    gbd_master_merge()