import requests
import json

FEATURES_URL = "http://127.0.0.1:8000/features"

features = requests.get(FEATURES_URL).json()["features"]

payload = {
    "features": {}
}

for f in features:
    payload["features"][f] = 0  # safe default

with open("payload.json", "w") as f:
    json.dump(payload, f, indent=2)

print("✅ payload.json created")
