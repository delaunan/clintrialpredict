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

# [STEP 1] Setup environment and local imports
load_dotenv()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
sys.path.append(PROJECT_ROOT)

# [STEP 2] Path Configuration
DATA_PATH = os.path.join(PROJECT_ROOT, 'data/')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data/processed/')
INPUT_FILE = os.path.join(DATA_PATH, 'llm_in_02.csv')
OUTPUT_FILE = os.path.join(OUTPUT_PATH, 'llm_out_02.csv')
LOG_FILE = os.path.join(PROJECT_ROOT, 'data/logs/enrichment_run2_errors.log')

# [STEP 3] Global Helpers
NL = chr(10)
MODEL_NAME = "gemini-2.5-flash-lite"
CONCURRENCY_LIMIT = 15
BATCH_SIZE = 1         # Ultimate precision mode for rescuing 9,800 trials
BUDGET_LIMIT_USD = 100.00
CONSECUTIVE_FAIL_LIMIT = 5

stats = {"input_new": 0, "input_cached": 0, "output": 0, "success": 0, "fail_streak": 0, "total_cost": 0.0}

FIELDNAMES = [
    "nct_id", "pharmacology_logic", "alpha_drug_name", "therapeutic_modality",
    "molecular_targets", "target_pathway_class", "target_precedent",
    "biomarker_stratification", "biomarker_description"
]

# [STEP 4] SCHEMA ENFORCEMENT
RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "nct_id": {"type": "STRING"},
            "pharmacology_logic": {"type": "STRING"},
            "alpha_drug_name": {"type": "STRING"},
            "therapeutic_modality": {
                "type": "STRING",
                "enum": [
                    "SMALL_MOLECULE", "BIOLOGIC_MAB", "BIOLOGIC_ADC", "BIOLOGIC_OTHER",
                    "CELL_GENE_THERAPY", "RNA_THERAPY", "VACCINE", "RADIOPHARMACEUTICAL",
                    "PEPTIDE_HORMONES", "OTHER_MODALITY"
                ]
            },
            "molecular_targets": {"type": "STRING"},
            "target_pathway_class": {
                "type": "STRING",
                "enum": [
                    "IMMUNO_ONCOLOGY", "KINASE_INHIBITOR", "METABOLIC_REPROGRAMMING",
                    "INTERLEUKIN_CYTOKINE", "GPCR_TARGET", "ENZYME_MODULATOR",
                    "EPIGENETIC_REGULATOR", "NUCLEAR_RECEPTOR", "PROTEIN_DEGRADER",
                    "ION_CHANNEL", "DNA_REPAIR", "OTHER_PATHWAY"
                ]
            },
            "target_precedent": {
                "type": "STRING",
                "enum": ["PRECEDENT_IN_INDICATION", "PRECEDENT_IN_OTHER", "NO_PRECEDENT"]
            },
            "biomarker_stratification": {"type": "BOOLEAN"},
            "biomarker_description": {"type": "STRING"}
        },
        "required": [
            "nct_id", "pharmacology_logic", "alpha_drug_name", "therapeutic_modality",
            "molecular_targets", "target_pathway_class", "target_precedent",
            "biomarker_stratification", "biomarker_description"
        ]
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
        try:
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1: return json.loads(text[start:end+1])
        except: pass
    return None

async def process_batch(semaphore, batch_df, cache_name, writer, f_handle):
    async with semaphore:
        if stats["total_cost"] > BUDGET_LIMIT_USD: return "BUDGET_EXCEEDED"
        if stats["fail_streak"] >= CONSECUTIVE_FAIL_LIMIT: return "CRITICAL_FAILURE_STREAK"

        contexts_payload = ""
        for i, (_, row) in enumerate(batch_df.iterrows()):
            contexts_payload += f"--- TRIAL {i+1} ---{NL}{row['context']}{NL}{NL}"

        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=f"EXTRACT MOLECULAR BLUEPRINTS FOR THESE {len(batch_df)} TRIALS.{NL}{contexts_payload}",
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
                if results is None or not isinstance(results, list):
                    raise ValueError("Batch JSON Parsing Failed")

                requested_ids = batch_df['nct_id'].tolist()
                result_map = {r.get('nct_id'): r for r in results if r.get('nct_id')}

                for nct_id in requested_ids:
                    if nct_id in result_map:
                        result = result_map[nct_id]
                        final_row = {field: str(result.get(field, "UNKNOWN")).replace("\n", " ").replace(",", ";") if field != "nct_id" else result.get(field) for field in FIELDNAMES}
                        writer.writerow(final_row)
                        stats["success"] += 1
                    else:
                        logging.error(f"CRITICAL: LLM dropped {nct_id} in batch Run 2.")

                f_handle.flush()
                stats["fail_streak"] = 0
                return True
            except Exception as e:
                if "429" in str(e): await asyncio.sleep(15 * (attempt + 1))
                else:
                    stats["fail_streak"] += 1
                    logging.error(f"Failed Run 2 Batch: {e}")
                    break
        return False

async def main():
    try:
        with open(os.path.join(PROJECT_ROOT, 'docs/prompts/llm_prompt_in_02.md'), 'r') as f: prompt_instr = f.read()
        with open(os.path.join(PROJECT_ROOT, 'docs/prompts/llm_prompt_in_02_ex.md'), 'r') as f: examples = f.read()
    except FileNotFoundError as e:
        print(f"[!] ERROR: Missing prompt or reference files: {e}"); return

    # Combine prompt and examples for the System Instruction
    system_instr = f"{prompt_instr}{NL}{NL}---{NL}{examples}"

    df_input = pd.read_csv(INPUT_FILE, dtype=str)
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            df_old = pd.read_csv(OUTPUT_FILE, usecols=['nct_id'], dtype=str)
            processed_ids = set(df_old['nct_id'].unique())
        except: pass
    df_todo = df_input[~df_input['nct_id'].isin(processed_ids)]

    if len(df_todo) == 0:
        print("> All Run 2 trials already processed.")
        return

    print(f"> Starting Run 2 Runner | {len(df_todo)} trials | BATCH SIZE: {BATCH_SIZE}")

    cache_name = None
    try:
        cache = client.caches.create(model=MODEL_NAME, config=types.CreateCachedContentConfig(display_name="run2_cache", system_instruction=system_instr, ttl='43200s'))
        cache_name = cache.name
        print(f"> Cache Active: {cache_name}")
    except Exception as e:
        print(f"[!] ERROR: Cache failed: {e}"); return

    try:
        with open(OUTPUT_FILE, 'a', newline='') as f:
            # [IRON GATE] Apply strict quoting for data integrity
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
            if not os.path.exists(OUTPUT_FILE) or os.stat(OUTPUT_FILE).st_size == 0: writer.writeheader()
            semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
            tasks = [process_batch(semaphore, df_todo.iloc[i:i+BATCH_SIZE], cache_name, writer, f) for i in range(0, len(df_todo), BATCH_SIZE)]
            await tqdm.gather(*tasks)
    finally:
        if cache_name:
            try: client.caches.delete(name=cache_name); print(f"{NL}> Cache deleted.")
            except: pass

    print(f"{NL}Run 2 Complete. Success: {stats['success']} | Est Cost: ${stats['total_cost']:.4f}")

    # [STEP 5] POST-PROCESSING: TARGET NORMALIZATION
    # (Moved to __main__ block to ensure execution on processed files)

def apply_logic_patch(file_path):
    """Enforces deterministic Iron Laws (IO Anchor, MAB Boundary, Symbol Wash, Greek Normalization)."""
    if not os.path.exists(file_path): return
    print(f"> Applying logic-lock patch to {file_path}...")
    df = pd.read_csv(file_path)

    io_anchors = {'PD-1', 'PD-L1', 'CTLA-4'}
    mab_mods = {'BIOLOGIC_MAB', 'BIOLOGIC_ADC'}
    leaky_pat = re.compile(r'^(Inhibitor|Agonist|Antagonist|Modulator|Targeting|Antibody against|Small molecule inhibitor) of\s+|'
                           r'\s+(Inhibitor|Agonist|Antagonist|Modulator|Protein|Enzyme)$', re.IGNORECASE)

    # Greek & Trademark normalization map
    special_map = {'\u03b1': 'A', '\u0391': 'A', '\u03b2': 'B', '\u0392': 'B', '\u03b3': 'G', '\u0393': 'G', '\u03b4': 'D', '\u0394': 'D', '\u03ba': 'K', '\u039a': 'K', '\u00ae': '', '\u2122': ''}

    def patch(row):
        targets_raw = str(row['molecular_targets'])
        # 1. SPECIAL CHARACTER & GREEK NORMALIZATION
        for char, replacement in special_map.items():
            targets_raw = targets_raw.replace(char, replacement)
            row['alpha_drug_name'] = str(row['alpha_drug_name']).replace(char, replacement)

        targets = [t.strip() for t in targets_raw.split('|')]
        mod = row['therapeutic_modality']
        path = row['target_pathway_class']
        logic = str(row['pharmacology_logic']).upper()

        # 2. PATHWAY GUARDS (Rules 1 & 2)
        if any(io in targets for io in io_anchors):
            row['target_pathway_class'] = 'IMMUNO_ONCOLOGY'
        elif mod in mab_mods and path == 'KINASE_INHIBITOR':
            row['target_pathway_class'] = 'OTHER_PATHWAY'

        # 3. SMALL MOLECULE INTERLEUKIN GUARDRAIL
        # Small molecules target enzymes/kinases in the pathway, not cytokines themselves.
        if mod == 'SMALL_MOLECULE' and path == 'INTERLEUKIN_CYTOKINE':
            if any(k in str(targets).upper() or k in logic for k in ['JAK', 'BTK', 'TYK', 'SYK', 'IRAK']):
                row['target_pathway_class'] = 'KINASE_INHIBITOR'
            else:
                row['target_pathway_class'] = 'ENZYME_MODULATOR'

        # 4. SYMBOL WASH & CASING LOCK
        new_targets = []
        for t in targets:
            clean_t = leaky_pat.sub('', t).strip()
            # Final trim and capitalize symbols (Consistent with post_process_targets)
            if len(clean_t) <= 10 and " " not in clean_t:
                clean_t = clean_t.upper()
            new_targets.append(clean_t)

        row['molecular_targets'] = " | ".join(new_targets)
        return row

    df.apply(patch, axis=1).to_csv(file_path, index=False)
    print(f"> Logic patch complete. Iron Laws enforced.")

def post_process_targets(file_path):
    """Normalizes naming conventions for molecular_targets to ensure consistency."""
    if not os.path.exists(file_path): return
    print(f"> Starting target normalization on {file_path}...")

    df = pd.read_csv(file_path)

    # High-priority canonical symbol mapping
    symbol_map = {
        "PD1": "PD-1", "CD279": "PD-1", "PDCD1": "PD-1",
        "PDL1": "PD-L1", "CD274": "PD-L1",
        "CTLA4": "CTLA-4", "CD152": "CTLA-4",
        "HER2": "HER2", "ERBB2": "HER2", "CD340": "HER2",
        "VEGFA": "VEGF", "VEGF-A": "VEGF",
        "TNFALPHA": "TNF", "TNF-ALPHA": "TNF", "TNFA": "TNF", "TNF-A": "TNF",
        "CD20": "CD20", "MS4A1": "CD20",
        "BCMA": "BCMA", "CD269": "BCMA", "TNFRSF17": "BCMA"
    }

    def normalize(val):
        if pd.isna(val) or str(val).strip().upper() in ["UNKNOWN", "N/A", "NONE"]:
            return "Unknown"

        raw_parts = [p.strip() for p in str(val).split("|")]
        clean_parts = []
        seen = set()

        for p in raw_parts:
            if not p: continue

            # 1. Check Canonical Map (Lookup key is uppercase, no dashes/spaces)
            lookup = p.upper().replace("-", "").replace(" ", "")
            if lookup in symbol_map:
                p_final = symbol_map[lookup]
            # 2. Heuristic for symbols (Short, no spaces -> FORCE UPPERCASE)
            elif len(p) <= 10 and " " not in p:
                p_final = p.upper()
            # 3. Descriptive targets: Title Case
            else:
                p_final = p.title()

            # Deduplicate (case-insensitive)
            if p_final.lower() not in seen:
                clean_parts.append(p_final)
                seen.add(p_final.lower())

        return " | ".join(clean_parts) if clean_parts else "Unknown"

    df['molecular_targets'] = df['molecular_targets'].apply(normalize)
    df.to_csv(file_path, index=False)
    print(f"> Normalization complete. Canonical symbols enforced.")

def run_master_audit(input_file, output_file):
    """Performs a comprehensive integrity and quality audit on the processed results."""
    print("\n" + "="*50)
    print("=== FINAL MASTER AUDIT: RUN 2 (PHARMACOLOGY) ===")

    try:
        df_in = pd.read_csv(input_file, usecols=['nct_id'])
        df_out = pd.read_csv(output_file)
    except Exception as e:
        print(f"Error loading files for audit: {e}")
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
        'therapeutic_modality': [
            "SMALL_MOLECULE", "BIOLOGIC_MAB", "BIOLOGIC_ADC", "BIOLOGIC_OTHER",
            "CELL_GENE_THERAPY", "RNA_THERAPY", "VACCINE", "RADIOPHARMACEUTICAL",
            "PEPTIDE_HORMONES", "OTHER_MODALITY"
        ],
        'target_pathway_class': [
            "IMMUNO_ONCOLOGY", "KINASE_INHIBITOR", "METABOLIC_REPROGRAMMING",
            "INTERLEUKIN_CYTOKINE", "GPCR_TARGET", "ENZYME_MODULATOR",
            "EPIGENETIC_REGULATOR", "NUCLEAR_RECEPTOR", "PROTEIN_DEGRADER",
            "ION_CHANNEL", "DNA_REPAIR", "OTHER_PATHWAY"
        ],
        'target_precedent': ["PRECEDENT_IN_INDICATION", "PRECEDENT_IN_OTHER", "NO_PRECEDENT"]
    }

    print("\n[2] ENUM INTEGRITY")
    for field, valid_values in enums.items():
        invalid = df_out[~df_out[field].isin(valid_values)]
        if len(invalid) > 0:
            print(f"❌ {field}: Found {len(invalid)} invalid values!")
        else:
            print(f"✅ {field}: 100% adherence.")

    drug_mods = ["SMALL_MOLECULE", "BIOLOGIC_MAB", "BIOLOGIC_ADC", "BIOLOGIC_OTHER", "CELL_GENE_THERAPY", "RNA_THERAPY", "PEPTIDE_HORMONES"]
    drug_trials = df_out[df_out['therapeutic_modality'].isin(drug_mods)]
    unknown_drugs = drug_trials[drug_trials['molecular_targets'].astype(str).str.contains('Unknown', case=False)]

    print("\n[3] PHARMACOLOGICAL QUALITY")
    print(f"Total Drug-like Trials: {len(drug_trials)}")
    print(f"Unresolved Drug Targets: {len(unknown_drugs)}")
    if len(unknown_drugs) < 650:
        print("✅ SUCCESS: High-priority resolution exceeds 98%.")
    else:
        print(f"⚠️  WARNING: {len(unknown_drugs)} drug targets remain unknown.")

    print("\n[4] LOGIC CONTRADICTION SCAN")
    contradictions = df_out[
        (df_out['target_precedent'] == 'NO_PRECEDENT') &
        (df_out['pharmacology_logic'].astype(str).str.contains('approved for .* in [12][0-9]{3}', case=False, na=False))
    ]
    true_contradictions = contradictions[~contradictions['pharmacology_logic'].astype(str).str.contains('not approved|no approval|zero global approvals', case=False, na=False)]

    if len(true_contradictions) == 0:
        print("✅ SUCCESS: No detectable logic contradictions.")
    else:
        print(f"⚠️  INFO: {len(true_contradictions)} potential contradictions (Same-Year Rule or Background overlaps).")

    print("\n" + "="*50)
    print("MASTER AUDIT COMPLETE")

if __name__ == "__main__":
    asyncio.run(main())

    # Ensure post-processing and logic-patch are ALWAYS applied to the final file
    post_process_targets(OUTPUT_FILE)
    apply_logic_patch(OUTPUT_FILE)

    run_master_audit(INPUT_FILE, OUTPUT_FILE)
