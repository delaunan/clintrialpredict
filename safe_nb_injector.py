import json
import sys
import os

# ==============================================================================
# MANDATORY USAGE PROTOCOL (v1.0)
# ------------------------------------------------------------------------------
# 1. ALWAYS use this script instead of direct text 'replace' on .ipynb files.
# 2. To update a notebook: Write code to a .tmp file, then execute this script.
# 3. Purpose: Prevents API 400 'INVALID_ARGUMENT' errors and JSON corruption.
# ==============================================================================

def inject_cell(notebook_path, source_file, cell_index=None, target_string=None):
    """
    Surgically injects code from a source file into a Jupyter Notebook cell.
    Prevents 400 INVALID_ARGUMENT errors by avoiding large text replacements via API.
    """
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook {notebook_path} not found.")
        sys.exit(1)
    
    if not os.path.exists(source_file):
        print(f"Error: Source file {source_file} not found.")
        sys.exit(1)

    # Read the new code to inject
    with open(source_file, 'r') as f:
        new_source = f.read().splitlines(keepends=True)

    # Load the notebook as JSON
    with open(notebook_path, 'r') as f:
        try:
            nb = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: {notebook_path} is not a valid JSON file. {e}")
            sys.exit(1)

    updated = False
    
    # Logic 1: Use direct cell index
    if cell_index is not None:
        if 0 <= cell_index < len(nb['cells']):
            nb['cells'][cell_index]['source'] = new_source
            updated = True
        else:
            print(f"Error: Cell index {cell_index} is out of range (Total cells: {len(nb['cells'])}).")
            sys.exit(1)
            
    # Logic 2: Search for a unique target string within cells
    elif target_string is not None:
        for i, cell in enumerate(nb['cells']):
            source_text = "".join(cell.get('source', []))
            if target_string in source_text:
                print(f"Target found in cell {i}. Injecting...")
                cell['source'] = new_source
                updated = True
                break
    
    if updated:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully updated {notebook_path} via JSON injection.")
    else:
        print(f"Error: Target string '{target_string}' not found in any cell.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python safe_nb_injector.py <notebook_path> <source_file> <target_string_or_index>")
        sys.exit(1)
    
    nb_path = sys.argv[1]
    src_file = sys.argv[2]
    target = sys.argv[3]
    
    try:
        # Check if the target is a numeric index
        idx = int(target)
        inject_cell(nb_path, src_file, cell_index=idx)
    except ValueError:
        # Otherwise, treat it as a search string
        inject_cell(nb_path, src_file, target_string=target)
