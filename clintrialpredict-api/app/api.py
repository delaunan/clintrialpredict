from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from app.model import predict, MODEL_FEATURES

app = FastAPI(title="Clinical Trial Prediction API")


class PredictRequest(BaseModel):
    features: Dict[str, Any]


@app.get("/")
def root():
    return {"status": "ok", "message": "Clinical Trial Prediction API running"}


@app.get("/features")
def get_features():
    """
    Returns the list of feature names expected by the model
    """
    return {"features": MODEL_FEATURES}


@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    """
    Make a prediction from named features
    """
    try:
        prediction = predict(request.features)
        return {"prediction": prediction}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
