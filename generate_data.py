import os
import pandas as pd
import numpy as np
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.prep.data_loader_clinpred import ClinicalTrialLoader

# Ensure env is set
os.environ['GLOBAL_START_YEAR'] = '2009'
os.environ['GLOBAL_END_YEAR'] = '2025'

print(">>> Initializing ClinicalTrialLoader...")
loader = ClinicalTrialLoader(data_path='data')
print(">>> Loading base data (AACT + LLM)...")
df = loader.load_base_data()
print(">>> Adding engineered features...")
df = loader.add_features(df)
print(">>> Saving to data_clinpred.csv...")
loader.save(df, 'data_clinpred.csv')
print(">>> Step 1 Complete.")
