import pandas as pd
import os
import json
import asyncio
import csv
import logging
import re
import sys
from tqdm.asyncio import tqdm
from google import genai
from google.genai import types
from dotenv import load_dotenv

# [STEP 1] Setup environment
load_dotenv()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
sys.path.append(PROJECT_ROOT)

# Paths
DATA_PATH = os.path.join(PROJECT_ROOT, 'data/')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data/processed/')
INPUT_FILE = os.path.join(DATA_PATH, 'llm_in_01.csv')
OUTPUT_FILE = os.path.join(OUTPUT_PATH, 'llm_out_00.csv')
LOG_FILE = os.path.join(PROJECT_ROOT, 'data/logs/enrichment_v1_errors.log')

# Config
MODEL_NAME = "gemini-2.0-flash"
CONCURRENCY_LIMIT = 40
BATCH_SIZE = 1 
FIELDNAMES = [
    "nct_id", "clinical_evidence", "mapping_logic", "gbd_cause_id", "gbd_indication_name",
    "therapeutic_area", "patient_severity", "line_of_therapy", "is_rare_disease"
]

ALLOWED_TAS = [
    "Oncology", "Cardiovascular", "Metabolic", "Neurology", "Infections",
    "Immunology", "Gastrointestinal", "Renal/Urology", "Psychiatry",
    "Dermatology", "Respiratory", "Ophthalmology", "Musculoskeletal",
    "Hematology", "Reproductive", "Genetic", "Dental", "Ear/Nose/Throat", "Unclassified"
]

# Schema
RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "nct_id": {"type": "STRING"},
            "clinical_evidence": {"type": "STRING"},
            "mapping_logic": {"type": "STRING"},
            "gbd_cause_id": {"type": "INTEGER"},
            "gbd_indication_name": {"type": "STRING"},
            "therapeutic_area": {"type": "STRING", "enum": ALLOWED_TAS},
            "patient_severity": {"type": "STRING"},
            "line_of_therapy": {"type": "STRING"},
            "is_rare_disease": {"type": "BOOLEAN"}
        },
        "required": ["nct_id", "clinical_evidence", "mapping_logic", "gbd_cause_id", "gbd_indication_name", "therapeutic_area", "patient_severity", "line_of_therapy", "is_rare_disease"]
    }
}

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

async def process_batch(semaphore, batch_df, cache_name, writer, f_handle):
    async with semaphore:
        contexts_payload = "".join([f"--- {row['nct_id']} ---\n{row['context']}\n\n" for _, row in batch_df.iterrows()])
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=f"EXTRACT DATA FOR THESE {len(batch_df)} TRIALS:\n{contexts_payload}",
                    config=types.GenerateContentConfig(cached_content=cache_name, response_mime_type="application/json", response_schema=RESPONSE_SCHEMA, temperature=0.0)
                )
                results = json.loads(response.text)
                for res in results:
                    writer.writerow({f: str(res.get(f)).replace("\n", " ") if f != "is_rare_disease" else res.get(f) for f in FIELDNAMES})
                f_handle.flush()
                return True
            except Exception as e:
                await asyncio.sleep(5); continue
        return False

async def main():
    with open(os.path.join(PROJECT_ROOT, 'docs/prompts/llm_prompt_in_01.md'), 'r') as f: prompt_instr = f.read()
    with open(os.path.join(PROJECT_ROOT, 'docs/prompts/gbd_codes.md'), 'r') as f: menu = f.read()
    system_instr = f"{prompt_instr}\n\n### GBD MENU ###\n{menu}"

    df_input = pd.read_csv(INPUT_FILE)
    if os.path.exists(OUTPUT_FILE):
        df_old = pd.read_csv(OUTPUT_FILE, usecols=['nct_id'])
        df_todo = df_input[~df_input['nct_id'].isin(df_old['nct_id'].unique())]
    else:
        df_todo = df_input

    if len(df_todo) == 0:
        print("> All trials already processed.")
        run_integrity_check()
        return

    print(f"> Run 1: Processing {len(df_todo)} trials...")
    
    # [CACHE ENFORCEMENT]
    cache_name = None
    try:
        cache = client.caches.create(model=MODEL_NAME, config=types.CreateCachedContentConfig(display_name="run1_cache", system_instruction=system_instr, ttl='3600s'))
        if not cache or not cache.name:
            raise ValueError("Cache creation failed: No cache name returned.")
        cache_name = cache.name
        print(f"> Cache Active: {cache_name}")
    except Exception as e:
        print(f"[!] CRITICAL CACHE FAILURE: {e}")
        print("> Run blocked to prevent high token costs. Check API status/quotas.")
        return

    try:
        with open(OUTPUT_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
            if f.tell() == 0: writer.writeheader()
            semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
            tasks = [process_batch(semaphore, df_todo.iloc[i:i+BATCH_SIZE], cache_name, writer, f) for i in range(0, len(df_todo), BATCH_SIZE)]
            await tqdm.gather(*tasks)
    finally:
        if cache_name:
            try:
                client.caches.delete(name=cache_name)
                print(f"> Cache deleted: {cache_name}")
            except Exception as e:
                print(f"> Warning: Cache deletion failed: {e}")

    run_integrity_check()

def run_integrity_check():
    print(f"\n{'='*40}")
    print("FORENSIC AUDIT: Run 1 (Baseline)")
    print(f"{'='*40}")
    df = pd.read_csv(OUTPUT_FILE)
    
    # Load hierarchy for level checking
    hier = pd.read_csv(os.path.join(DATA_PATH, 'reference/hier_gbd.csv'))
    l2_ids = set(hier[hier['Level'] == 2]['Cause ID'])
    
    # 1. TA Hallucination Check
    invalid_ta = df[~df['therapeutic_area'].isin(ALLOWED_TAS)]
    print(f"  - TA Hallucinations: {len(invalid_ta)}")

    # 2. Refinement Target Quantification
    id_0_count = len(df[df['gbd_cause_id'] == 0])
    l2_count = len(df[df['gbd_cause_id'].isin(l2_ids)])
    
    print(f"  - Unmapped (ID 0):   {id_0_count} trials")
    print(f"  - Safety Net (L2):   {l2_count} trials")
    print(f"  - TOTAL TARGETS FOR RUN 1.2: {id_0_count + l2_count}")
    
    print(f"\n> Run 1 Audit Complete. {len(df)} trials verified.")

if __name__ == "__main__":
    asyncio.run(main())
