import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
from src.prep.pipeline import FEATURE_REGISTRY

for feat, meta in FEATURE_REGISTRY.items():
    print(f"{feat}: {meta.get('encoding')}")
