import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# --- CONFIGURATION ---

# Add CORS (Cross-Origin Resource Sharing) middleware.
# This is a security feature that allows your Streamlit frontend (running on a different server)
# to successfully send requests to this API. Without this, the browser would block the connection.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Allows requests from ANY website (essential for Streamlit Cloud)
    allow_credentials=True,
    allow_methods=["*"],    # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],    # Allows all HTTP headers
)

# --- LOAD MODELS AT STARTUP ---
# Load the heavy machine learning artifacts into memory ONCE when the server launches.
# We store them in 'app.state' so they are globally accessible and ready for instant use.
# This prevents the app from having to reload the files from disk for every single request (which would be slow).

app.state.model = joblib.load("models/ctp_model.joblib")               # The trained model for prediction
app.state.explainer = joblib.load("models/shap_explainer.joblib")      # The SHAP explainer for interpretability
app.state.taxonomy = joblib.load("models/feature_taxonomy.joblib")     # Feature taxonomy mapping

# Patch for SHAP version mismatch if needed
# This fixes a known compatibility issue where older saved SHAP explainers crash
# because they expect an 'approximate' attribute that might be missing in newer versions.
# We manually add it here to prevent the API from crashing during prediction.

if not hasattr(app.state.explainer, "approximate"):
    app.state.explainer.approximate = False

# --- HELPER FUNCTIONS (Moved from app.py) ---
def sigmoid(x):
    # Converts "Log-Odds" (raw numbers from the model) into a Probability (0 to 1).
    # Essential for interpreting SHAP values as actual percentages of risk.
    # Added np.clip to prevent overflow errors with extreme values
    return 1 / (1 + np.exp(-np.clip(x, -700, 700)))

def map_feature_to_business_pillar(feature_name, taxonomy_dict):

    # Translates technical feature names (e.g., "pca_col_3") into business categories.
    # It first checks our official dictionary (taxonomy).
    # If not found, it guesses based on keywords like "sponsor" or "pca".
    # This is used to group the bars in the "Pillar Impact" chart.

    if feature_name in taxonomy_dict['pillar_map']:
        return taxonomy_dict['pillar_map'][feature_name], taxonomy_dict['subcat_map'][feature_name]

    name = str(feature_name).lower()
    if 'emb_' in name or 'pca' in name:
        return 'Patient & Criteria', 'Criteria Complexity'
    if 'sponsor' in name:
        return 'Sponsor & Operations', 'Sponsor Capability'
    return 'Other', 'Unclassified'



@app.get("/")
def root():
    return {"status": "API is online"}



@app.post("/predict")
async def predict(request: Request):
    """
    Receives a full dictionary of trial data (JSON),
    runs the model + SHAP, and returns the results.
    """
    try:
        # 1. Parse Input Data (Dynamic Schema)
        input_data = await request.json() # Received as a dictionary
        X_new = pd.DataFrame([input_data]) # Convert to DataFrame (1 row)

        model = app.state.model
        explainer = app.state.explainer
        taxonomy = app.state.taxonomy

        # 2. Run Prediction
        # (Fallback if predict_proba is not available)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_new)[0]
            # Assuming class 1 = Fail, class 0 = Success (Adjust based on your model!)
            # Check your original logic: failed=1 -> did NOT complete
            idx_fail = list(model.classes_).index(1)
            p_fail = float(proba[idx_fail])
        else:
            pred = int(model.predict(X_new)[0])
            p_fail = 1.0 if pred == 1 else 0.0

        p_success = 1.0 - p_fail

        # 3. Run SHAP Explanation
        # Transform data using the pipeline's preprocessor
        preprocessor = model.named_steps['preprocessor']
        X_encoded = preprocessor.transform(X_new)

        feature_names = taxonomy['feature_names']
        X_encoded_df = pd.DataFrame(X_encoded, columns=feature_names)

        # Calculate SHAP
        shap_values = explainer(X_encoded_df)
        base_log_odds = float(shap_values.base_values[0]) # Ensure float for safety

        # --- RE-CALCULATION OF PROBABILITY ---
        # We calculate the TRUE total using ALL features first to ensure accuracy
        all_shap_values = shap_values.values[0]
        true_total_log_odds = np.sum(all_shap_values)

        final_log_odds = base_log_odds + true_total_log_odds
        prob_fail_shap = sigmoid(final_log_odds)
        prob_success_shap = 1.0 - prob_fail_shap

        # Calculate scaling factor for % contribution
        # Uses the true total gap so percentages align with the final probability
        total_gap = prob_success_shap - (1 - sigmoid(base_log_odds))
        scale_factor = total_gap / true_total_log_odds if abs(true_total_log_odds) > 1e-9 else 0

        # Process Impacts
        impacts = []
        for i, col in enumerate(feature_names):
            val = all_shap_values[i]
            if abs(val) < 1e-9: continue

            pillar, subcat = map_feature_to_business_pillar(col, taxonomy)
            #if pillar == 'Other': continue

            impacts.append({
                'Pillar': pillar,
                'Subcategory': subcat,
                'Feature': col,
                'Raw_Log_Odds': float(val), # Convert to float for JSON serialization
                'Impact_Pct': float(val * scale_factor) # <--- Wrap in float()
            })

        return {
            "prediction_success": float(prob_success_shap), # <--- Wrap in float()
            "impacts": impacts,
            "status": "success"
        }

    except Exception as e:
        # Check if p_success exists (in case error happened very early)
        safe_prediction = float(p_success) if 'p_success' in locals() else 0.0

        return {
            "prediction_success": safe_prediction, # Fallback to simple prediction
            "impacts": [],
            "status": "partial_error",
            "error_msg": str(e)
        }
