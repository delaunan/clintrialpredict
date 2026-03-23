import csv
import pandas as pd
import numpy as np
import os
from pathlib import Path

def is_float(val):
    try:
        float(str(val).replace(',', '.'))
        return True
    except:
        return False

def forensic_repair_llm_01(input_path, output_path):
    print(f"> Starting Forensic Audit & Repair: {input_path}")
    
    repaired_rows = []
    audit_log = {"total": 0, "clean": 0, "repaired": 0, "failed": []}
    
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        expected_len = len(header) # Should be 29
        
        for i, row in enumerate(reader):
            audit_log["total"] += 1
            if len(row) == expected_len:
                repaired_rows.append(row)
                audit_log["clean"] += 1
                continue
            
            # --- START FORENSIC REPAIR (SQUEEZE) ---
            try:
                new_row = [None] * expected_len
                
                # 1. ANCHOR: NCT_ID (Index 0)
                new_row[0] = row[0]
                
                # 2. ANCHOR: THE TAIL (Indices 23-28 / Last 6)
                # gbd_cause_id_4, gbd_cause_id_3, gbd_cause_id_2, gbd_indication_name_4, 3, 2
                new_row[23:] = row[-6:]
                
                # 3. ANCHOR: THE FLOATS (Indices 14-22 / 9 columns)
                # Search backwards from the tail to find the block of 9 floats
                float_block = []
                idx = len(row) - 7
                while len(float_block) < 9 and idx >= 0:
                    if is_float(row[idx]):
                        float_block.insert(0, row[idx])
                    idx -= 1
                new_row[14:23] = float_block
                
                # 4. ANCHOR: THE ENUMS (Index 8: is_rare_disease)
                # Search for 'True' or 'False' specifically
                rare_idx = -1
                for j in range(len(row)):
                    if str(row[j]).strip().upper() in ['TRUE', 'FALSE']:
                        rare_idx = j
                        break
                
                if rare_idx != -1:
                    new_row[8] = row[rare_idx]
                    new_row[7] = row[rare_idx - 1] # line_of_therapy
                    new_row[6] = row[rare_idx - 2] # patient_severity
                    new_row[5] = row[rare_idx - 3] # therapeutic_area
                
                # 5. TEXT GLUE: Clinical Evidence & Mapping Logic (Indices 1, 2)
                # Everything between NCT_ID and gbd_cause_id (Index 3)
                # We find gbd_cause_id by looking for the first integer after NCT_ID
                id_idx = 1
                for j in range(1, len(row)):
                    if row[j].isdigit() and int(row[j]) > 0:
                        id_idx = j
                        break
                
                new_row[3] = row[id_idx] # gbd_cause_id
                
                # Join everything in between for Evidence & Logic
                middle_text = " ".join(row[1:id_idx])
                if "(1)" in middle_text:
                    parts = middle_text.split("(1)", 1)
                    new_row[1] = parts[0].strip()
                    new_row[2] = "(1) " + parts[1].strip()
                else:
                    new_row[1] = middle_text
                    new_row[2] = "NOT_SPECIFIED"

                # 6. TEXT GLUE: Indication Names (Indices 4, 9, 10)
                # Join everything between gbd_cause_id and therapeutic_area
                ta_idx = rare_idx - 3
                new_row[4] = ", ".join(row[id_idx+1 : ta_idx])
                new_row[9] = new_row[3] # Cause ID (Duplicate)
                new_row[10] = new_row[4] # Cause Name (Duplicate)
                new_row[11] = row[rare_idx + 1] # gbd_hierarchy_level
                new_row[12] = row[rare_idx + 2] # model_ta
                new_row[13] = row[rare_idx + 3] # Parent ID

                repaired_rows.append(new_row)
                audit_log["repaired"] += 1
                
            except Exception as e:
                audit_log["failed"].append(f"Row {i+1} (NCT: {row[0]}): {str(e)}")

    # Write REPAIRED file
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        writer.writerows(repaired_rows)
        
    return audit_log

# Run the audit
input_f = "data/processed/llm_out_01.csv"
output_f = "data/processed/llm_out_01_REPAIRED.csv"
stats = forensic_repair_llm_01(input_f, output_f)

print("\n--- REPAIR AUDIT REPORT ---")
print(f"Total Rows Processed: {stats['total']}")
print(f"Clean Rows Kept:      {stats['clean']}")
print(f"Corrupted Rows Fixed: {stats['repaired']}")
print(f"Total Failed Rows:    {len(stats['failed'])}")
if stats['failed']:
    print("\nSample Failures:")
    for err in stats['failed'][:3]: print(f"  [!] {err}")
