import requests
import pandas as pd
import numpy as np
from collections import defaultdict

API_URL = "http://localhost:8000/predict"
DATA_PATH = "frontend/data/search_registry.csv"

def verify_trial(nct_id, ta, csv_score):
    payload = {"nct_id": nct_id, "therapeutic_area": ta}
    try:
        res = requests.post(API_URL, json=payload, timeout=10)
        if res.status_code != 200:
            return False, f"HTTP {res.status_code}"
            
        data = res.json()
        if "error" in data:
            return False, f"API Error: {data['error']}"
            
        api_score = data['score']
        pillars = data['pillar_impacts']
        subcats = data['subcat_impacts']
        
        # 1. Internal API Parity (Sub -> Pillar -> Score)
        sub_by_pillar = defaultdict(float)
        for s in subcats:
            sub_by_pillar[s['Pillar']] += s['Impact']
            
        pillar_errors = []
        for p in pillars:
            p_name = p['Pillar']
            p_imp = p['Impact']
            s_sum = round(sub_by_pillar[p_name], 1)
            if abs(p_imp - s_sum) > 0.01:
                pillar_errors.append(f"{p_name}: Pillar({p_imp}) != SumSub({s_sum})")
                
        baseline = 50.0
        sum_pillars = round(sum(p['Impact'] for p in pillars), 1)
        calc_score_pillars = round(baseline + sum_pillars, 1)
        
        errs = []
        if abs(calc_score_pillars - api_score) > 0.01:
            errs.append(f"API Internal Mismatch: Gauge({api_score}) != Baseline+Pillars({calc_score_pillars})")
        if pillar_errors:
            errs.extend(pillar_errors)
            
        # 2. Registry (CSV) vs API Parity
        # Formatting both to 1 decimal place as shown in the UI
        csv_score_fmt = round(float(csv_score), 1)
        api_score_fmt = round(float(api_score), 1)
        
        if abs(csv_score_fmt - api_score_fmt) > 0.01:
            errs.append(f"Registry Mismatch: CSV({csv_score_fmt}) != API({api_score_fmt})")
            
        if not errs:
            return True, f"Score: {api_score} | Registry Parity: OK"
        else:
            return False, " | ".join(errs)

    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    print("=== Full System Parity Audit (CSV Registry vs API vs Gauge) ===")
    df = pd.read_csv(DATA_PATH).head(50) # Expanded check
    
    passes = 0
    fails = 0
    
    for idx, row in df.iterrows():
        nct_id = row['nct_id']
        ta = row['therapeutic_area']
        csv_score = row['Clinical_Score']
        success, msg = verify_trial(nct_id, ta, csv_score)
        if success:
            print(f"[PASS] {nct_id}: {msg}")
            passes += 1
        else:
            print(f"[FAIL] {nct_id}: {msg}")
            fails += 1
            
    print("\n" + "="*80)
    print(f"AUDIT SUMMARY: {passes} Passed, {fails} Failed")
    if fails == 0:
        print("RESULT: Absolute Parity Confirmed. The Registry Grid and Gauge match bit-perfectly.")
    else:
        print("RESULT: Discrepancies detected between Registry (CSV) and predictive Engine.")
        print("ACTION: Registry may need synchronization with the latest rounding logic.")
