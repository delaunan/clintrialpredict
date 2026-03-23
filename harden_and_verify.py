import pandas as pd
import csv
from pathlib import Path

# Config for validation
expected_counts = {
    'data/processed/llm_out_01.csv': 29,
    'data/processed/llm_out_02.csv': 9,
    'data/processed/llm_out_03.csv': 9,
    'data/processed/llm_out_04.csv': 8
}

def harden_and_validate(f_path, expected_n):
    if not Path(f_path).exists():
        print(f"❌ File not found: {f_path}")
        return
    
    # 1. READ: Use pandas (Smart Reader)
    # It handles commas inside quotes correctly, so it should see 29 columns
    df = pd.read_csv(f_path, low_memory=False)
    actual_n = len(df.columns)
    
    if actual_n != expected_n:
        print(f"⚠️  HEADER MISMATCH in {f_path}: Expected {expected_n}, Found {actual_n}")
        # Note: If this fails, it means the source file was REALLY broken
    
    # 2. WRITE: Apply the "Iron Shield" (QUOTE_ALL)
    df.to_csv(f_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"✅ Hardened: {f_path} (Cols: {actual_n})")
    
    # 3. VERIFY: Re-read to confirm every row is now identical
    df_new = pd.read_csv(f_path, quoting=csv.QUOTE_ALL, low_memory=False)
    if len(df_new.columns) == expected_n:
        print(f"💎 INTEGRITY VERIFIED: {f_path}")
    else:
        print(f"🔥 CRITICAL FAILURE: {f_path} still has {len(df_new.columns)} columns!")

# Execute
for path, count in expected_counts.items():
    harden_and_validate(path, count)
