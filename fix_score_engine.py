
import json

path = 'notebooks/production_06.ipynb'
with open(path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    source_str = "".join(cell.get('source', []))
    if 'SCORE_ENGINE_CODE' in source_str:
        print(f"Fixing cell: {source_str[:50]}...")
        # Replace X_test with df_full globally in this cell
        new_source = []
        for line in cell['source']:
            new_source.append(line.replace('X_test', 'df_full'))
        cell['source'] = new_source
        cell['outputs'] = []
        cell['execution_count'] = None

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)
print("Done.")
