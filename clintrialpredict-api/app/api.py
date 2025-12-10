from fastapi import FastAPI, HTTPException
from app.model import predict, MODEL_FEATURES
import os
import uvicorn

app = FastAPI(title="Clinical Trial Prediction API")

# ------------------------------------------------------------------
# Root / health check
# ------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "model_features": len(MODEL_FEATURES)
    }

# ------------------------------------------------------------------
# Expose expected features (VERY IMPORTANT)
# ------------------------------------------------------------------
@app.get("/features")
def features():
    return {
        "n_features": len(MODEL_FEATURES),
        "features": MODEL_FEATURES
    }

# ------------------------------------------------------------------
# Prediction endpoint
# ------------------------------------------------------------------
@app.post("/predict")
def predict_endpoint(payload: dict):
    """
    Expected payload format:
    {
      "features": {
        "feature_name": value,
        ...
      }
    }
    """
    try:
        if "features" not in payload:
            raise ValueError("Payload must contain a 'features' object")

        prediction = predict(payload["features"])
        return {"prediction": prediction}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------------
# Cloud Run / local entrypoint
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Cloud Run provides PORT as environment variable
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app.api:app", host="0.0.0.0", port=port)
