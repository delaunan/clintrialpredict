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
from collections import defaultdict

# [STEP 1] Setup environment
load_dotenv()
PROJECT_ROOT = "/home/delaunan/code/delaunan/clintrialpredict"
sys.path.append(PROJECT_ROOT)

DATA_PATH = os.path.join(PROJECT_ROOT, 'data/')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data/processed/')
INPUT_FILE = os.path.join(DATA_PATH, 'llm_in_03.csv')
OUTPUT_FILE = os.path.join(OUTPUT_PATH, 'llm_out_03.csv')
LOG_FILE = os.path.join(PROJECT_ROOT, 'data/logs/enrichment_v3_run3_errors.log')

# [STEP 3] Global Helpers
NL = chr(10)
MODEL_NAME = "gemini-2.0-flash"
CONCURRENCY_LIMIT = 40
BATCH_SIZE = 3
BUDGET_LIMIT_USD = 100.00
CONSECUTIVE_FAIL_LIMIT = 5

stats = {"input_new": 0, "input_cached": 0, "output": 0, "success": 0, "fail_streak": 0, "total_cost": 0.0}

FIELDNAMES = [
    "nct_id", "strategist_logic", "endpoint_rigor", "endpoint_structure",
    "comparator_benchmark", "strategic_ambition", "administration_complexity",
    "innovation_tier", "adaptive_design"
]

RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "nct_id": {"type": "STRING"},
            "strategist_logic": {"type": "STRING"},
            "endpoint_rigor": {"type": "STRING", "enum": ["HARD_CLINICAL", "SURROGATE", "SUBJECTIVE_PRO"]},
            "endpoint_structure": {"type": "STRING", "enum": ["SINGLE_GOAL", "MULTI_COMPOSITE"]},
            "comparator_benchmark": {"type": "STRING", "enum": ["PLACEBO", "ACTIVE_MODERN_STANDARD", "ACTIVE_LEGACY_STANDARD", "NO_CONTROL_GROUP"]},
            "strategic_ambition": {"type": "STRING", "enum": ["PIVOTAL_INTENT", "SIGNAL_SEARCH", "SAFETY_DOSING"]},
            "administration_complexity": {"type": "STRING", "enum": ["SIMPLE_ORAL", "ROUTINE_INFUSION", "INTENSIVE_MANAGEMENT"]},
            "innovation_tier": {"type": "STRING", "enum": ["FIRST_IN_CLASS", "NEXT_GEN_OPTIMIZED", "ESTABLISHED_COPY"]},
            "adaptive_design": {"type": "STRING", "enum": ["ADAPTIVE", "STATIC"]}
        },
        "required": ["nct_id", "strategist_logic", "endpoint_rigor", "endpoint_structure", "comparator_benchmark", "strategic_ambition", "administration_complexity", "innovation_tier", "adaptive_design"]
    }
}

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

def safe_json_loads(text):
    try: return json.loads(text)
    except:
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except: pass
    return None

async def process_batch(semaphore, batch_df, cache_name, writer, f_handle):
    async with semaphore:
        if stats["total_cost"] > BUDGET_LIMIT_USD: return "BUDGET_EXCEEDED"
        if stats["fail_streak"] >= CONSECUTIVE_FAIL_LIMIT: return "CRITICAL_FAILURE_STREAK"

        contexts_payload = ""
        for _, row in batch_df.iterrows():
            contexts_payload += f"[NCT_ID]: {row['nct_id']}{NL}{row['context']}{NL}{NL}"

        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=f"EXTRACT DATA FOR THESE {len(batch_df)} TRIALS. FOLLOW THE V17.5 LOGIC FOR EVERY TRIAL.{NL}{contexts_payload}",
                    config=types.GenerateContentConfig(
                        cached_content=cache_name,
                        response_mime_type="application/json",
                        response_schema=RESPONSE_SCHEMA,
                        temperature=0.0
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
                        writer.writerow({f: str(res.get(f, "UNKNOWN")).replace("\n", " ").replace(",", ";").strip() for f in FIELDNAMES})
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
    with open(os.path.join(PROJECT_ROOT, 'docs/prompts/llm_prompt_in_03.md'), 'r') as f: prompt_instr = f.read()
    with open(os.path.join(PROJECT_ROOT, 'docs/prompts/llm_prompt_in_03_ex.md'), 'r') as f: few_shots = f.read()

    batch_refinement = [
        f"{NL}### [BATCH_STRATEGY_V17.5_RULES] ###",
        "1. ID-ANCHOR MANDATE: Every trial result MUST be indexed by its exact NCT_ID from the context. DO NOT use generic labels like 'TRIAL 1'.",
        "2. LOGIC-LOCK: Use result markers [STEP-X-RESULT: ...] in strategist_logic monologue.",
        "3. THE PILL RULE: Oral = SIMPLE_ORAL. Hospital monitoring NEVER upgrades complexity.",
        "4. ONCOLOGY RIGOR: PFS/ORR are SURROGATE. OS is HARD_CLINICAL.",
        "5. FORBIDDEN UNKNOWN: Resolve all fields using Title/Strategy Clues. No UNKNOWN allowed."
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
        print("> All Run 3 trials already processed.")
        return

    print(f"> Starting Run 3 (Strategist) v17.5 | {len(df_todo)} trials remaining")

    try:
        cache = client.caches.create(model=MODEL_NAME, config=types.CreateCachedContentConfig(display_name="run3_v17_cache", system_instruction=system_instr, ttl='43200s'))
        print(f"> Cache Active: {cache.name}")
        try:
            with open(OUTPUT_FILE, 'a', newline='') as f:
                # [IRON GATE] Apply strict quoting for data integrity
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
                if not os.path.exists(OUTPUT_FILE) or os.stat(OUTPUT_FILE).st_size == 0: writer.writeheader()
            semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
            tasks = [process_batch(semaphore, df_todo.iloc[i:i+BATCH_SIZE], cache.name, writer, f) for i in range(0, len(df_todo), BATCH_SIZE)]
            await tqdm.gather(*tasks)
        client.caches.delete(name=cache.name)
    except Exception as e:
        print(f"[!] ERROR: {e}")

def run_master_audit(input_file, output_file):
    """Performs a comprehensive integrity and quality audit on the Run 3 (Strategist) results."""
    print("\n" + "="*50)
    print("=== FINAL MASTER AUDIT: RUN 3 (STRATEGIST) ===")

    if not os.path.exists(output_file):
        print(f"❌ ERROR: Output file {output_file} not found.")
        return

    try:
        df_in = pd.read_csv(input_file, usecols=['nct_id'], dtype=str)
        df_out = pd.read_csv(output_file, dtype=str)
    except Exception as e:
        print(f"❌ Error loading files for audit: {e}")
        return

    unique_in = df_in['nct_id'].nunique()
    unique_out = df_out['nct_id'].nunique()

    print("\n[1] RECONCILIATION")
    print(f"Expected Unique IDs: {unique_in}")
    print(f"Actual Unique IDs:   {unique_out}")
    if unique_in == unique_out:
        print("✅ SUCCESS: 100% of trials accounted for.")
    else:
        print(f"❌ FAILURE: {unique_in - unique_out} trials are still missing!")

    enums = {
        'endpoint_rigor': ["HARD_CLINICAL", "SURROGATE", "SUBJECTIVE_PRO"],
        'endpoint_structure': ["SINGLE_GOAL", "MULTI_COMPOSITE"],
        'comparator_benchmark': ["PLACEBO", "ACTIVE_MODERN_STANDARD", "ACTIVE_LEGACY_STANDARD", "NO_CONTROL_GROUP"],
        'strategic_ambition': ["PIVOTAL_INTENT", "SIGNAL_SEARCH", "SAFETY_DOSING"],
        'administration_complexity': ["SIMPLE_ORAL", "ROUTINE_INFUSION", "INTENSIVE_MANAGEMENT"],
        'innovation_tier': ["FIRST_IN_CLASS", "NEXT_GEN_OPTIMIZED", "ESTABLISHED_COPY"],
        'adaptive_design': ["ADAPTIVE", "STATIC"]
    }

    print("\n[2] ENUM INTEGRITY (SCHEMA ADHERENCE)")
    total_violations = 0
    for field, valid_values in enums.items():
        if field not in df_out.columns:
            print(f"❌ FIELD MISSING: {field}")
            total_violations += 1
            continue

        invalid = df_out[~df_out[field].isin(valid_values)]
        if len(invalid) > 0:
            print(f"❌ {field}: Found {len(invalid)} invalid values! (e.g., {invalid[field].unique()[:3]})")
            total_violations += len(invalid)
        else:
            print(f"✅ {field}: 100% adherence.")

    print("\n[3] LOGIC QUALITY (FORENSIC MONOLOGUE)")
    # Check if strategist_logic contains the mandatory [STEP-X-RESULT] markers
    logic_field = 'strategist_logic'
    missing_markers = df_out[~df_out[logic_field].str.contains(r'\[STEP-\d-RESULT:', na=False)]
    if len(missing_markers) == 0:
        print("✅ SUCCESS: 100% of rows contain mandatory Logic-Lock markers.")
    else:
        print(f"⚠️  WARNING: {len(missing_markers)} rows are missing Logic-Lock markers in strategist_logic.")

    print("\n[4] OPERATIONAL METRICS")
    print(f"Total Trials Audited: {len(df_out)}")
    print(f"Tokens - New Input:    {stats['input_new']:,}")
    print(f"Tokens - Cached Input: {stats['input_cached']:,}")
    print(f"Tokens - Output:       {stats['output']:,}")
    print(f"ESTIMATED TOTAL COST:  ${stats['total_cost']:.4f}")

    print("\n" + "="*50)
    print("MASTER AUDIT COMPLETE")

if __name__ == "__main__":
    asyncio.run(main())
    run_master_audit(INPUT_FILE, OUTPUT_FILE)
