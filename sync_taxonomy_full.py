import nbformat
import os

def update_taxonomy_in_nb(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Load the target taxonomy from our fixed file
    with open('models/taxonomy_01.json', 'r') as f:
        master_taxonomy = json.load(f)
    
    found = False
    for cell in nb.cells:
        if cell.cell_type == 'code' and ('RISK_TAXONOMY = {' in cell.source or '"1. Therapeutic Context": {' in cell.source):
            # We recreate the cell source precisely
            # Determine which taxonomy variable name is used
            var_name = "RISK_TAXONOMY" if "RISK_TAXONOMY =" in cell.source else "taxonomy"
            
            # Format the dictionary nicely as Python code
            import json
            tax_str = json.dumps(master_taxonomy, indent=4)
            
            # Inject it back into the cell
            # We keep the leading comment and variable name
            new_source = f"{var_name} = {tax_str}"
            
            # Re-add common boilerplate like GAIN_FACTOR if present
            if "GAIN_FACTOR =" in cell.source:
                gain_val = cell.source.split("GAIN_FACTOR =")[1].split("\n")[0].strip()
                new_source += f"\n\nGAIN_FACTOR ={gain_val}"
            
            # Re-add REF tags if present
            if "# <REF:" in cell.source:
                start_tag = cell.source.split("\n")[0]
                end_tag = cell.source.split("\n")[-1]
                cell.source = f"{start_tag}\n{new_source}\n{end_tag}"
            else:
                cell.source = new_source
                
            found = True
            break
            
    if found:
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Updated full taxonomy in {nb_path}")

import json
update_taxonomy_in_nb('notebooks/model_validation.ipynb')
update_taxonomy_in_nb('notebooks/production_01.ipynb')
