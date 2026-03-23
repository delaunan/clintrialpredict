
import json

with open('notebooks/production_06.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell.get('source', []))
    if 'SCORE_ENGINE_CODE' in source:
        print(f"Index: {i}")
        print("Source:")
        print(source)
