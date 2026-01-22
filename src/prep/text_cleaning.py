import pandas as pd
import numpy as np
import re
import os

def day_zero_reconstructor(text, pillar_label="text"):
    """
    Final optimized cleaner for all NLP pillars.
    Protects medical terminology and purges administrative leakage.
    """
    if not isinstance(text, str) or not text.strip() or text.lower() == 'nan':
        return f"No {pillar_label} provided"

    # 1. PROTECT: Medical terms and our manual [SEP] tokens
    text = text.replace("[SEP]", " __SEP_TOKEN__ ")

    protected_terms = ["RECIST", "CTCAE", "NYHA", "ECOG", "HbA1c", "pfs", "os", "orr"]
    for i, term in enumerate(protected_terms):
        text = re.sub(rf'(?i)\b{term}\b', f"__PROT_{i}__", text)

    # 2. CLEAN AACT ARTIFACTS (Replaced with spaces to prevent word merging)
    text = text.replace('~', ' ')
    text = text.replace('\\>', ' ')
    text = text.replace('*', ' ')
    text = re.sub(r'-{2,}', ' ', text)

    # 3. LEAKAGE PURGE (Administrative Redaction)
    admin_patterns = [
        r'(?i)\bamendment\b.*?(?=[.;:]|$)',
        r'(?i)\bprotocol\s+v(?:er|ersion)?\.?\s*[\d\.]+',
        r'(?i)\brevised\s+(?:per|on|by)\b.*?(?=[.;:]|$)',
        r'(?i)\bupdated\s+(?:as\s+of|on|protocol)\b.*?(?=[.;:]|$)',
        r'(?i)\bmodified\s+(?:on|date)\b.*?(?=[.;:]|$)'
    ]
    for pattern in admin_patterns:
        text = re.sub(pattern, ' ', text)

    # 4. RESTORE & NORMALIZE
    for i, term in enumerate(protected_terms):
        text = text.replace(f"__PROT_{i}__", term.upper())

    text = text.replace("__SEP_TOKEN__", "[SEP]")
    text = re.sub(r'\s+', ' ', text).strip()

    return text if len(text) > 5 else f"No {pillar_label} provided"

# ==============================================================================
# PROTECTED EXECUTION BLOCK
# This code ONLY runs if you execute this file directly (python text_cleaning.py)
# It is IGNORED when you import the function into your notebook.
# ==============================================================================
if __name__ == "__main__":
    input_path = 'data/project_data.csv'
    if not os.path.exists(input_path):
        print(f"ERROR: Could not find {input_path}")
    else:
        df = pd.read_csv(input_path)
        print(">>> Executing Final Scientific Sanitization...")
        pillars = ['txt_scientific_essence', 'txt_criteria', 'txt_primary_endpoints']
        for col in pillars:
            if col in df.columns:
                label = col.replace('txt_', '')
                df[col] = df[col].apply(lambda x: day_zero_reconstructor(x, label))

        output_path = 'data/project_data_nlp_light.csv'
        df[['nct_id', 'target', 'txt_scientific_essence', 'txt_criteria', 'txt_primary_endpoints']].to_csv(output_path, index=False)
        print(f"[SUCCESS] File saved: {output_path}")
