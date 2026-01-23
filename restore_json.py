import json

def restore_notebook_json():
    path = 'notebooks/production.ipynb'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Manually fix the specific corrupted line that breaks JSON parsing
    # The error was at line 387: '2. Scientific Design': {
    content = content.replace("'2. Scientific Design': {", '"2. Scientific Design": {')
    # Fix the single quotes inside the RISK_TAXONOMY dict while we are at it
    content = content.replace("'1. Therapeutic Context': {", '"1. Therapeutic Context": {')
    content = content.replace("'features': ['cat_onehot__therapeutic_area'", '"features": ["cat_onehot__therapeutic_area"')
    content = content.replace("'logic': 'The combined risk profile", '"logic": "The combined risk profile')
    
    # Try to parse it. If it fails, we will try a more aggressive approach.
    try:
        data = json.loads(content)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print("Successfully restored JSON structure.")
    except Exception as e:
        print(f"JSON restore failed: {e}")
        # Aggressive: the notebook is currently a mess of mixed quotes because of 'replace'.
        # I will overwrite it with a clean template if it remains unparsable.

if __name__ == '__main__':
    restore_notebook_json()
