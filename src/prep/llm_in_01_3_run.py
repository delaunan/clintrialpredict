
import pandas as pd
import os
import re
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
V0_FILE = os.path.join(PROJECT_ROOT, 'data/processed/llm_out_00.csv')
MASTER_REF = os.path.join(PROJECT_ROOT, 'data/reference/gbd_stats.csv')
FINAL_EXPORT = os.path.join(PROJECT_ROOT, 'data/processed/llm_out_01.csv')

ALLOWED_TAS = ["Oncology", "Cardiovascular", "Metabolic", "Neurology", "Infections", "Immunology", "Gastrointestinal", "Renal/Urology", "Psychiatry", "Dermatology", "Respiratory", "Ophthalmology", "Musculoskeletal", "Hematology", "Reproductive", "Genetic", "Dental", "Ear/Nose/Throat", "Unclassified"]

def main():
    print(f"> Run 1.3: Linking statistics and recovering hierarchy...")
    df_results = pd.read_csv(V0_FILE)
    df_master = pd.read_csv(MASTER_REF)

    # 1. Statistical Join
    cols_to_keep = [
        'Cause ID', 'Cause Name', 'Level', 'model_ta', 'Parent ID', 
        'daly_global', 'daly_high_income', 'yld_global', 'yld_high_income', 
        'yll_global', 'yll_high_income', 'chronic_ratio_global', 
        'chronic_ratio_high_income', 'market_skew_index'
    ]
    df_merged = pd.merge(df_results, df_master[cols_to_keep], left_on='gbd_cause_id', right_on='Cause ID', how='left')
    
    # 2. Hierarchy Recovery
    parent_map = df_master.set_index('Cause ID')['Parent ID'].to_dict()
    name_map = df_master.set_index('Cause ID')['Cause Name'].to_dict()
    
    df_merged['gbd_cause_id_4'] = 0
    df_merged['gbd_cause_id_3'] = 0
    df_merged['gbd_cause_id_2'] = 0
    df_merged.rename(columns={'Level': 'gbd_hierarchy_level'}, inplace=True)

    for i, row in df_merged.iterrows():
        lvl = row['gbd_hierarchy_level']
        cid = int(row['gbd_cause_id'])
        if cid == 0 or pd.isna(lvl): continue
        if lvl == 4:
            df_merged.at[i, 'gbd_cause_id_4'] = cid
            p3 = parent_map.get(cid, 0)
            df_merged.at[i, 'gbd_cause_id_3'] = p3
            df_merged.at[i, 'gbd_cause_id_2'] = parent_map.get(p3, 0)
        elif lvl == 3:
            df_merged.at[i, 'gbd_cause_id_3'] = cid
            df_merged.at[i, 'gbd_cause_id_2'] = parent_map.get(cid, 0)
        elif lvl == 2:
            df_merged.at[i, 'gbd_cause_id_2'] = cid

    # 3. Recover Indication Names for UI
    for level in [4, 3, 2]:
        df_merged[f'gbd_indication_name_{level}'] = df_merged[f'gbd_cause_id_{level}'].map(lambda x: name_map.get(x, "") if x != 0 else "")

    # [REVISED STATISTICAL ASSIGNMENT]
    stat_cols = [
        'daly_global', 'daly_high_income', 'yld_global', 'yld_high_income', 
        'yll_global', 'yll_high_income', 'chronic_ratio_global', 
        'chronic_ratio_high_income', 'market_skew_index'
    ]
    
    # A. Level 0 / ID 0: Set all stats to 0.0
    mask_0 = (df_merged['gbd_cause_id'] == 0)
    df_merged.loc[mask_0, stat_cols] = 0.0
    df_merged.loc[mask_0, 'gbd_hierarchy_level'] = 1.0 # Standardize level
    df_merged.loc[mask_0, 'gbd_indication_name'] = "UNKNOWN / OTHER"
    print(f"  - Cleared stats for {mask_0.sum()} Level 0 trials.")

    # B. Level 2 (Broad): Assign MEAN of L3 Children
    mask_l2 = (df_merged['gbd_hierarchy_level'] == 2)
    if mask_l2.any():
        print(f"  - Re-calculating representative stats for {mask_l2.sum()} Level 2 trials...")
        # Calculate L3 Means per TA/Category from the master reference
        l3_stats = df_master[df_master['Level'] == 3]
        category_means = l3_stats.groupby('Parent ID')[stat_cols].mean().to_dict('index')
        
        for i, row in df_merged[mask_l2].iterrows():
            cid = int(row['gbd_cause_id'])
            if cid in category_means:
                for col in stat_cols:
                    df_merged.at[i, col] = category_means[cid][col]

    # [IRON GATE] Apply strict quoting for data integrity
    import csv
    df_merged.to_csv(FINAL_EXPORT, index=False, quoting=csv.QUOTE_ALL)
    
    # 4. FINAL FORENSIC AUDIT
    print(f"\n{'='*40}\nMASTER FORENSIC AUDIT: Run 1.3 (Final)\n{'='*40}")
    print("HIERARCHY DISTRIBUTION:")
    lvl_counts = df_merged['gbd_hierarchy_level'].fillna(0).value_counts().sort_index()
    for lvl, count in lvl_counts.items():
        desc = {1: "Global Baseline", 2: "Category (Broad)", 3: "Indication", 4: "Subtype"}.get(int(lvl), "Unknown")
        print(f"  - Level {int(lvl)} ({desc:<15}): {count:>5} trials")

    # Stats Audit
    id0_daly = df_merged[df_merged['gbd_cause_id'] == 0]['daly_global'].mean()
    l2_daly = df_merged[df_merged['gbd_hierarchy_level'] == 2]['daly_global'].mean()
    print(f"\nSTATISTICAL VALIDATION:")
    print(f"  - Mean DALY for ID 0 (Expected 0.0): {id0_daly:.2f}")
    print(f"  - Mean DALY for L2 (Representative):  {l2_daly:.2f}")
    
    print(f"\n> Run 1.3 Audit Complete. Master Dataset: {FINAL_EXPORT}")

if __name__ == "__main__":
    main()
