import nbformat
import json
import os

def update_taxonomy_in_nb(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    with open('models/taxonomy_01.json', 'r') as f:
        master_taxonomy = json.load(f)
    
    found = False
    for cell in nb.cells:
        if cell.cell_type == 'code' and ('RISK_TAXONOMY =' in cell.source):
            tax_str = json.dumps(master_taxonomy, indent=4)
            # Find the gain factor value from the existing cell
            gain_line = [l for l in cell.source.split('\n') if 'GAIN_FACTOR =' in l]
            gain_val = gain_line[0] if gain_line else "GAIN_FACTOR = 25.0"
            
            # Reconstruct cell with REF tags
            if 'model_validation' in nb_path:
                new_source = f"# <REF:SCORE_ENGINE_CODE>\nimport pandas as pd\nimport numpy as np\nfrom scipy.special import logit\n
print(\"\n>>> STEP 11: SCORING ENGINE (SUCCESS-ORIENTED UI)...")\n
RISK_TAXONOMY = {tax_str}\n
{gain_val} # Scales 1 logit unit to 25 UI points\n"
                # Keep the function and calling code
                remaining = cell.source.split('def generate_clinical_scorecard')[1]
                cell.source = new_source + "\ndef generate_clinical_scorecard" + remaining
            else:
                new_source = f"# <REF:PROD_SCORE_ENGINE_CODE>\nRISK_TAXONOMY = {tax_str}\n
{gain_val}\n"
                remaining = cell.source.split('def generate_production_scorecard')[1]
                cell.source = new_source + "\ndef generate_production_scorecard" + remaining
                
            found = True
            break
            
    if found:
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Updated full taxonomy in {nb_path}")

update_taxonomy_in_nb('notebooks/model_validation.ipynb')
update_taxonomy_in_nb('notebooks/production_01.ipynb')
