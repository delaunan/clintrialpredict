import joblib
import json

# Check SHAP values
shap_data = joblib.load("models/shap_values_01.joblib")
first_key = list(shap_data.keys())[0]
shap_val = shap_data[first_key]
print(f"SHAP array length: {len(shap_val)}")

# Check Taxonomy
with open("models/taxonomy_01.json", 'r') as f:
    taxonomy = json.load(f)

all_features = []
for pillar, subcats in taxonomy.items():
    for subcat, info in subcats.items():
        all_features.extend(info['features'])

print(f"Total features in taxonomy: {len(all_features)}")
# Remove CALIBRATION_OFFSET if it's a virtual feature
real_features = [f for f in all_features if f != "CALIBRATION_OFFSET"]
print(f"Real features in taxonomy: {len(real_features)}")

# Check threshold data
with open("models/thresholds_01.json", 'r') as f:
    thresholds = json.load(f)
print(f"Base value (Intercept): {thresholds.get('base_value')}")

# Check pipeline feature names if possible
model = joblib.load("models/model_prod_01.joblib")
prep = model.named_steps['prep']
try:
    feature_names = prep.get_feature_names_out()
    print(f"Pipeline feature names length: {len(feature_names)}")
except:
    print("Could not get feature names from prep")