import pandas as pd
import os
import asyncio
import json
import re
import logging
from google import genai
from google.genai import types
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

# [STEP 1] Setup
load_dotenv()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
INPUT_MASTER = os.path.join(PROJECT_ROOT, 'data/llm_in_01.csv')
V0_FILE = os.path.join(PROJECT_ROOT, 'data/processed/llm_out_00.csv')
REF_CONTEXT = os.path.join(PROJECT_ROOT, 'data/llm_in_01_2.csv')
HIER_FILE = os.path.join(PROJECT_ROOT, 'data/reference/hier_gbd.csv')
STATS_FILE = os.path.join(PROJECT_ROOT, 'data/reference/gbd_stats.csv')
MODEL_NAME = "gemini-3-flash-preview"

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

ALLOWED_TAS = [
    "Oncology", "Cardiovascular", "Metabolic", "Neurology", "Infections",
    "Immunology", "Gastrointestinal", "Renal/Urology", "Psychiatry",
    "Dermatology", "Respiratory", "Ophthalmology", "Musculoskeletal",
    "Hematology", "Reproductive", "Genetic", "Dental", "Ear/Nose/Throat", "Unclassified"
]

RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "nct_id": {"type": "STRING"},
            "mapping_logic": {"type": "STRING"},
            "gbd_cause_id": {"type": "INTEGER"},
            "gbd_indication_name": {"type": "STRING"},
            "therapeutic_area": {"type": "STRING", "enum": ALLOWED_TAS},
            "is_rare_disease": {"type": "BOOLEAN"}
        },
        "required": ["nct_id", "mapping_logic", "gbd_cause_id", "gbd_indication_name", "therapeutic_area", "is_rare_disease"]
    }
}

async def process_batch(semaphore, batch_df, cache_name, refined_results, prev_map):
    async with semaphore:
        contexts = "".join([f"--- {row['nct_id']} ---\n[PREV]: {prev_map.get(row['nct_id'])}\n[CTX]: {row['context']}\n\n" for _, row in batch_df.iterrows()])
        for _ in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=f"REFINE THESE TRIALS:\n{contexts}",
                    config=types.GenerateContentConfig(cached_content=cache_name, response_mime_type="application/json", response_schema=RESPONSE_SCHEMA, temperature=0.0)
                )
                results = json.loads(response.text)
                for res in results: refined_results[res['nct_id']] = res
                return True
            except: await asyncio.sleep(5); continue
        return False

async def main():
    # 1. Filter Targets
    hier = pd.read_csv(HIER_FILE)
    broad_ids = hier[hier['Level'] <= 2]['Cause ID'].tolist()
    df_v0 = pd.read_csv(V0_FILE)
    mask = (df_v0['gbd_cause_id'] == 0) | (df_v0['gbd_cause_id'].isin(broad_ids))
    target_ids = df_v0[mask]['nct_id'].unique()
    
    df_in = pd.read_csv(INPUT_MASTER)
    df_todo = df_in[df_in['nct_id'].isin(target_ids)].copy()
    df_todo.to_csv(REF_CONTEXT, index=False)

    # 2. Prepare Prompts & Cache
    with open(os.path.join(PROJECT_ROOT, 'docs/prompts/llm_prompt_in_01_2.md'), 'r') as f: prompt = f.read()
    with open(os.path.join(PROJECT_ROOT, 'docs/prompts/gbd_codes.md'), 'r') as f: menu = f.read()
    system_instr = f"{prompt}\n\n### GBD MENU ###\n{menu}"
    
    cache = client.caches.create(model=MODEL_NAME, config=types.CreateCachedContentConfig(display_name="ref_cache", system_instruction=system_instr, ttl='3600s'))
    prev_map = df_v0.set_index('nct_id').apply(lambda x: f"{x['gbd_indication_name']} ({x['gbd_cause_id']})", axis=1).to_dict()
    refined_results = {}
    
    try:
        print(f"> Run 1.2: Refining {len(df_todo)} trials...")
        semaphore = asyncio.Semaphore(20)
        tasks = [process_batch(semaphore, df_todo.iloc[i:i+5], cache.name, refined_results, prev_map) for i in range(0, len(df_todo), 5)]
        await tqdm.gather(*tasks)
    finally:
        if cache_name := cache.name:
            try: client.caches.delete(name=cache_name)
            except: pass

    # 3. Surgical Update & Forensic Audit
    menu_map = {int(m.group(1)): m.group(2).strip() for m in [re.search(r'\[ID:\s*(\d+)\]\s*([^|]+)', line) for line in menu.split('\n')] if m}
    df_stats = pd.read_csv(STATS_FILE)
    id_to_ta_map = df_stats.set_index('Cause ID')['model_ta'].to_dict()

    # [INTEGRITY GUARD] Explicit field separation
    UPDATE_FIELDS = ['mapping_logic', 'gbd_cause_id', 'gbd_indication_name', 'therapeutic_area', 'is_rare_disease']
    IMMUTABLE_FIELDS = ['clinical_evidence', 'patient_severity', 'line_of_therapy']
    df_orig_snapshot = df_v0.copy()  # Snapshot for final verification

    updates = 0
    logic_conflicts = 0
    hallucinations = 0
    
    branch_map = {"A": "Unmapped Recovery (L0 -> L3/4)", "B": "True Broad Preservation (L2 Sum)", "C": "Multi-Indication Assignment (Main L3)", "D": "Specific Sister Fallback (Other L3)", "None": "Undefined Branch"}
    branch_counts = {v: 0 for v in branch_map.values()}

    for nct_id, res in refined_results.items():
        mask_v0 = df_v0['nct_id'] == nct_id
        if mask_v0.any():
            cid = int(res['gbd_cause_id'])
            
            # Hallucination Check (ID-Name)
            if cid != 0 and cid in menu_map:
                if res['gbd_indication_name'].lower().strip() not in menu_map[cid].lower().strip(): hallucinations += 1
            
            # TA-Identity Check
            expected_ta = id_to_ta_map.get(cid, "Unclassified")
            if cid != 0 and expected_ta != res['therapeutic_area']:
                logic_conflicts += 1
                logging.error(f"TA CONFLICT {nct_id}: ID {cid} is {expected_ta} but LLM said {res['therapeutic_area']}")

            # Branch Tally
            logic = res.get('mapping_logic', '').upper()
            found_branch = False
            for char, name in branch_map.items():
                if f"BRANCH {char}" in logic:
                    branch_counts[name] += 1
                    found_branch = True
                    break
            if not found_branch: branch_counts["Undefined Branch"] += 1

            # Surgical Update: ONLY targeting defined UPDATE_FIELDS
            official_name = menu_map.get(cid, res['gbd_indication_name'])
            df_v0.loc[mask_v0, UPDATE_FIELDS] = \
                [res['mapping_logic'], cid, official_name, res['therapeutic_area'], res['is_rare_disease']]
            updates += 1

    # [FINAL INTEGRITY AUDIT]
    if len(df_v0) != len(df_orig_snapshot):
        raise ValueError(f"CRITICAL: Row count mismatch! (Orig: {len(df_orig_snapshot)}, New: {len(df_v0)})")
    
    for field in IMMUTABLE_FIELDS:
        if not df_v0[field].equals(df_orig_snapshot[field]):
            raise ValueError(f"CRITICAL: Immutable field '{field}' was corrupted during refinement update.")

    # [IRON GATE] Apply strict quoting for data integrity
    import csv
    df_v0.to_csv(V0_FILE, index=False, quoting=csv.QUOTE_ALL)
    
    print(f"\n{'='*40}\nFORENSIC AUDIT: Run 1.2 (Refinement)\n{'='*40}")
    print(f"  - Refined trials: {updates}")
    print(f"  - Logic Branch Utilization:")
    for name, count in branch_counts.items(): print(f"    - {name:<40}: {count}")
    print(f"  - GBD Code/Name Hallucinations: {hallucinations}")
    print(f"  - TA-Identity Conflicts:       {logic_conflicts}")
    print(f"  - Unmapped (ID 0) remaining:   {len(df_v0[df_v0['gbd_cause_id'] == 0])}")
    print(f"\n> Run 1.2 Audit Complete. Surgical update successful.")

if __name__ == "__main__":
    asyncio.run(main())