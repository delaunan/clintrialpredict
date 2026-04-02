import pandas as pd
import numpy as np

# Define the 21 root determinants
determinants = [
    "nct_id", "title", "summary_ui", "criteria_ui", "conditions_ui", "interventions_ui", 
    "primary_outcomes_ui", "therapeutic_area", "start_date", "minimum_age", "maximum_age", 
    "phase", "number_of_arms", "allocation", "intervention_model", "masking", "has_dmc", 
    "has_placebo", "gender", "healthy_volunteers", "includes_us"
]

# Load the data
df = pd.read_csv("frontend/data/search_registry.csv", usecols=determinants)

char_stats = []

for col in determinants:
    # Treat all values as strings to count characters, dropping truly empty rows
    data = df[col].astype(str).replace(['nan', 'None', ''], np.nan).dropna()
    
    if data.empty:
        char_stats.append({
            "Root Determinant": col,
            "Min Chars": 0,
            "Max Chars": 0,
            "Mean Chars": 0,
            "Avg Chars": 0,
            "Char Distribution": "No Data"
        })
        continue

    lengths = data.str.len()
    
    # Calculate Distribution (Top 3 character counts by frequency)
    dist = lengths.value_counts(normalize=True).head(3)
    dist_str = ", ".join([f"{int(k)} len: {int(v*100)}%" for k, v in dist.items()])
    
    char_stats.append({
        "Root Determinant": col,
        "Min Chars": lengths.min(),
        "Max Chars": lengths.max(),
        "Mean Chars": round(lengths.mean(), 1),
        "Avg Chars": round(lengths.mean(), 1),
        "Char Distribution": dist_str
    })

# Load existing mapping
existing_df = pd.read_excel("simulation_determinants_mapping.xlsx")

# Drop old stat columns if they exist from previous turns to avoid duplicates
cols_to_drop = ["Min", "Max", "Mean", "Average", "Distribution", "Min Chars", "Max Chars", "Mean Chars", "Avg Chars", "Char Distribution"]
existing_df = existing_df.drop(columns=[c for c in cols_to_drop if c in existing_df.columns], errors='ignore')

# Merge
stats_df = pd.DataFrame(char_stats)
final_df = existing_df.merge(stats_df, on="Root Determinant", how="left")

# Save
final_df.to_excel("simulation_determinants_mapping.xlsx", index=False)
print("Updated Excel with character-specific stats for all 21 fields.")
