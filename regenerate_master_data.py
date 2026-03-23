from src.prep.data_loader_clinpred import ClinicalTrialLoader
import os

# Initialize loader
loader = ClinicalTrialLoader(data_path='data/')

print(">>> Regenerating Master Data (data_clinpred.csv)...")
try:
    # 1. Load base data (merges LLM runs)
    df = loader.load_base_data()
    
    # 2. Add features (Hybrid engineering)
    df = loader.add_features(df)
    
    # 3. Save final master file
    loader.save(df)
    
    print("✅ SUCCESS: Master data regenerated and correctly aligned.")
except Exception as e:
    print(f"❌ FAILURE during regeneration: {e}")
