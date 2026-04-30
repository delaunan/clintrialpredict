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
PROJECT_ROOT = "/home/delaunan/code/delaunan/clintrialpredict"
sys.path.append(PROJECT_ROOT)

DATA_PATH = os.path.join(PROJECT_ROOT, 'data/')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data/processed/')
INPUT_FILE = os.path.join(DATA_PATH, 'llm_in_04.csv')
OUTPUT_FILE = os.path.join(OUTPUT_PATH, 'llm_out_04.csv')
LOG_FILE = os.path.join(PROJECT_ROOT, 'data/logs/enrichment_v4_run4_errors.log')

# [STEP 3] Global Helpers
NL = chr(10)
MODEL_NAME = "gemini-2.5-flash-lite"
CONCURRENCY_LIMIT = 20
BATCH_SIZE = 1         # One-by-one for maximum structural precision
BUDGET_LIMIT_USD = 50.00
CONSECUTIVE_FAIL_LIMIT = 5

stats = {"input_new": 0, "input_cached": 0, "output": 0, "success": 0, "fail_streak": 0, "total_cost": 0.0}

# LOGIC-LOCK SCHEMA: structural_forensic_monologue at second position
FIELDNAMES = [
    "nct_id", "structural_forensic_monologue", "lead_sponsor_canonical", "sponsor_tier",
    "primary_duration_value", "primary_duration_unit", "primary_duration_months", "is_duration_unknown"
]

def calculate_duration_months(value, unit):
    """Robust conversion from value/unit to float months, with a 15-year (180 month) cap."""
    try:
        val = float(value)
        u = str(unit).upper().strip()
        months = 0.0
        if u == "YEARS": months = val * 12.0
        elif u == "MONTHS": months = val
        elif u == "WEEKS": months = val / 4.348 # Average 4.348 weeks per month
        elif u == "DAYS": months = val / 30.4375 # Average 30.4375 days per month
        
        # [CAP RULE] Maximum 180 months (15 years) to prevent extreme outliers (e.g. 50yr follow-up) from skewing ML
        return round(min(months, 180.0), 2)
    except:
        pass
    return 0.0

RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "nct_id": {"type": "STRING"},
            "structural_forensic_monologue": {"type": "STRING"},
            "lead_sponsor_canonical": {"type": "STRING"},
            "sponsor_tier": {"type": "STRING", "enum": ["TIER 1", "MID_CAP", "BIOTECH"]},
            "primary_duration_value": {"type": "NUMBER"},
            "primary_duration_unit": {"type": "STRING", "enum": ["DAYS", "WEEKS", "MONTHS", "YEARS", "UNKNOWN"]}
        },
        "required": ["nct_id", "structural_forensic_monologue", "lead_sponsor_canonical", "sponsor_tier", "primary_duration_value", "primary_duration_unit"]
    }
}

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

def safe_json_loads(text):
    try: 
        res = json.loads(text)
        if isinstance(res, dict): return [res] # Auto-wrap
        return res
    except:
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            try: 
                res = json.loads(match.group(1))
                if isinstance(res, dict): return [res]
                return res
            except: pass
        try:
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1: return json.loads(text[start:end+1])
            start_obj = text.find('{')
            end_obj = text.rfind('}')
            if start_obj != -1 and end_obj != -1:
                res = json.loads(text[start_obj:end_obj+1])
                return [res]
        except: pass
    return None

def wash_input_text(text):
    """Normalizes Greek and special characters in input to prevent JSON corruption."""
    if not isinstance(text, str): return text
    charmap = {'\u03b1': 'alpha', '\u0391': 'Alpha', '\u03b2': 'beta', '\u0392': 'Beta', '\u03b3': 'gamma', '\u0393': 'Gamma', '\u03b4': 'delta', '\u0394': 'Delta', '\u03ba': 'kappa', '\u039a': 'Kappa', '\u00ae': '', '\u2122': '', '\u2264': '<=', '\u2265': '>='}
    for char, replacement in charmap.items(): text = text.replace(char, replacement)
    return text

async def process_batch(semaphore, batch_df, cache_name, writer, f_handle):
    async with semaphore:
        if stats["total_cost"] > BUDGET_LIMIT_USD: return "BUDGET_EXCEEDED"
        if stats["fail_streak"] >= CONSECUTIVE_FAIL_LIMIT: return "CRITICAL_FAILURE_STREAK"

        # [ISOLATION] Use clear markers for each trial
        contexts_payload = ""
        for _, row in batch_df.iterrows():
            clean_ctx = wash_input_text(row['context'])
            contexts_payload += f"### DATA_START_FOR_{row['nct_id']} ###{NL}{clean_ctx}{NL}### DATA_END_FOR_{row['nct_id']} ###{NL}{NL}"

        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=f"EXTRACT DATA FOR THESE {len(batch_df)} TRIALS. FOLLOW THE V18.3 STEEL SHIELD LOGIC.{NL}{contexts_payload}",
                    config=types.GenerateContentConfig(
                        cached_content=cache_name,
                        response_mime_type="application/json",
                        response_schema=RESPONSE_SCHEMA,
                        temperature=0.0,
                        max_output_tokens=1024
                    )
                )

                if response.usage_metadata:
                    usage = response.usage_metadata
                    cached = getattr(usage, 'cached_content_token_count', 0)
                    new_input = usage.prompt_token_count
                    output = usage.candidates_token_count
                    stats["input_new"] += new_input
                    stats["input_cached"] += cached
                    stats["output"] += output
                    stats["total_cost"] += (new_input / 1e6 * 0.10) + (cached / 1e6 * 0.025) + (output / 1e6 * 0.40)

                results = safe_json_loads(response.text)
                if results is None: raise ValueError("JSON Parse Error")

                requested_ids = batch_df['nct_id'].tolist()
                result_map = {r.get('nct_id'): r for r in results if r.get('nct_id')}

                for nct_id in requested_ids:
                    if nct_id in result_map:
                        res = result_map[nct_id]
                        
                        # [STEP 5] HOMOGENEOUS CONVERSION (Weeks/Months to Months)
                        val = res.get("primary_duration_value", 0)
                        unit = res.get("primary_duration_unit", "UNKNOWN")
                        res["primary_duration_months"] = calculate_duration_months(val, unit)
                        res["is_duration_unknown"] = 1 if str(unit).upper() == "UNKNOWN" else 0
                        
                        clean_res = {f: str(res.get(f, "UNKNOWN")).replace(chr(10), " ").replace(",", ";").strip() for f in FIELDNAMES}
                        writer.writerow(clean_res)
                        stats["success"] += 1
                    else:
                        logging.error(f"ID mismatch: {nct_id} missing in LLM response.")

                f_handle.flush()
                stats["fail_streak"] = 0
                return True
            except Exception as e:
                if "429" in str(e): await asyncio.sleep(15 * (attempt + 1))
                else:
                    stats["fail_streak"] += 1
                    logging.error(f"Batch Error: {e}")
                    break
        return False

async def main():
    # Load Prompts
    with open(os.path.join(PROJECT_ROOT, 'docs/prompts/llm_prompt_in_04.md'), 'r') as f: prompt_instr = f.read()
    with open(os.path.join(PROJECT_ROOT, 'docs/prompts/llm_prompt_in_04_ex.md'), 'r') as f: few_shots = f.read()

    batch_refinement = [
        f"{NL}### [BATCH_STRATEGY_V18.3_RULES] ###",
        "1. ID-ANCHOR MANDATE: Every trial result MUST be indexed by its exact NCT_ID.",
        "2. LOGIC-LOCK: Use [STEP-X-...] result markers in structural_forensic_monologue.",
        "3. BIT-PERFECT CAPS: Sponsor names must be ALL CAPS with no suffixes.",
        "4. LONGEST DURATION: Mathematically compare all units (90 Days < 6 Months).",
        "5. TIER STRICTNESS: Only use TIER 1, MID_CAP, or BIOTECH labels."
    ]

    system_instr = NL.join([prompt_instr, NL.join(batch_refinement), f"{NL}### [EXAMPLES] ###{NL}", few_shots])

    df_input = pd.read_csv(INPUT_FILE, dtype=str)

    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            df_old = pd.read_csv(OUTPUT_FILE, usecols=['nct_id'], dtype=str)
            processed_ids = set(df_old['nct_id'].unique())
        except: pass

    df_todo = df_input[~df_input['nct_id'].isin(processed_ids)]

    if len(df_todo) == 0:
        print("> All Run 4 trials already processed.")
        return

    print(f"> Starting Run 4 (Structural Anchor) v18.3 | {len(df_todo)} trials remaining")

    try:
        cache = client.caches.create(model=MODEL_NAME, config=types.CreateCachedContentConfig(display_name="run4_v18_cache", system_instruction=system_instr, ttl='43200s'))
        print(f"> Cache Active: {cache.name}")
        try:
            with open(OUTPUT_FILE, 'a', newline='') as f:
                # [IRON GATE] Apply strict quoting for data integrity
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
                if f.tell() == 0: writer.writeheader()
                
                semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
                tasks = [process_batch(semaphore, df_todo.iloc[i:i+BATCH_SIZE], cache.name, writer, f) for i in range(0, len(df_todo), BATCH_SIZE)]
                await tqdm.gather(*tasks)
        finally:
            print(f"> Deleting Cache: {cache.name}")
            client.caches.delete(name=cache.name)
    except Exception as e:
        print(f"[!] ERROR: {e}")

def homogenize_all_durations(output_file):
    """Post-processing: Reads the entire file and ensures primary_duration_months is calculated for all rows."""
    if not os.path.exists(output_file): return
    print(f"> Homogenizing all durations in {output_file}...")
    df = pd.read_csv(output_file, dtype=str)
    
    # Recalculate everything for consistency
    df['primary_duration_months'] = df.apply(
        lambda x: calculate_duration_months(x.get('primary_duration_value', 0), x.get('primary_duration_unit', 'UNKNOWN')), 
        axis=1
    )
    df['is_duration_unknown'] = df['primary_duration_unit'].apply(lambda x: 1 if str(x).upper() == "UNKNOWN" else 0)
    
    # Ensure correct column order
    df = df[FIELDNAMES]
    df.to_csv(output_file, index=False)
    print(f"> Done. {len(df)} rows homogenized.")

def run_final_audit(input_file, output_file):
    print(NL + "="*50)
    print("=== FINAL MASTER AUDIT: RUN 4 (STRUCTURAL ANCHOR) ===")
    if not os.path.exists(output_file): return

    df_out = pd.read_csv(output_file, dtype=str)
    total_count = len(df_out)
    print(f"{NL}[1] RECONCILIATION: {total_count} trials processed.")

    print(f"{NL}[2] SPONSOR TIER DISTRIBUTION")
    print(df_out['sponsor_tier'].value_counts())

    print(f"{NL}[3] DURATION UNIT DISTRIBUTION")
    unit_counts = df_out['primary_duration_unit'].value_counts()
    print(unit_counts)
    
    # Specific Unknown reporting
    unknown_count = unit_counts.get('UNKNOWN', 0)
    unknown_pct = (unknown_count / total_count * 100) if total_count > 0 else 0
    print(f"--- UNKNOWN DURATIONS: {unknown_count} ({unknown_pct:.2f}%) ---")

    print(f"{NL}[4] DURATION (MONTHS) STATS (Excluding 0.0/UNKNOWN)")
    # Convert to numeric and filter out 0.0 for cleaner stats
    dur_numeric = pd.to_numeric(df_out['primary_duration_months'], errors='coerce')
    valid_durations = dur_numeric[dur_numeric > 0]
    if not valid_durations.empty:
        print(valid_durations.describe())
    else:
        print("No valid durations found.")

    print(f"{NL}[5] OPERATIONAL METRICS")
    print(f"ESTIMATED TOTAL COST: ${stats['total_cost']:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
    homogenize_all_durations(OUTPUT_FILE)
    run_final_audit(INPUT_FILE, OUTPUT_FILE)
