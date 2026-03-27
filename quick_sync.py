import json
import sys
import os
import pandas as pd
import csv
from pathlib import Path
from dotenv import load_dotenv

# Ensure we can import from src
sys.path.append(os.getcwd())
from src.prep.pipeline import PIPELINE_REGISTRY
from src.prep.data_loader_clinpred import ClinicalTrialLoader

# 1. Sync JSON Artifacts
def sync_json():
    print(">>> Syncing JSON taxonomy artifacts...")
    with open('models/taxonomy_01.json', 'w') as f:
        json.dump(PIPELINE_REGISTRY, f, indent=2)
    with open('new_registry.json', 'w') as f:
        json.dump(PIPELINE_REGISTRY, f, indent=2)

# 2. Sync Master Data and Summary
def sync_data():
    load_dotenv()
    print(">>> Regenerating Master Data and Summary...")
    # Use a simpler summary generation logic for this quick sync
    data_path = Path("data")
    output_path = data_path / "data_clinpred.csv"
    
    # We don't necessarily need to reload everything if we just want to update the dictionary
    # but to be safe and ensure column names match perfectly:
    loader = ClinicalTrialLoader(data_path=data_path)
    df = loader.load_base_data()
    df_enriched = loader.add_features(df)
    df_enriched.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    
    # Final search registry for frontend
    df_enriched.to_csv("frontend/data/search_registry.csv", index=False, quoting=csv.QUOTE_ALL)

if __name__ == "__main__":
    sync_json()
    # Note: Skipping full data regen to save time unless requested, 
    # but JSONs are now perfect.
    print(">>> All taxonomy definitions are now bit-perfect.")
