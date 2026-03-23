
import json

with open('notebooks/production_06.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell.get('execution_count') == 15:
        print(f"Index: {i}")
        print("Source:")
        print("".join(cell.get('source', [])))
        break
