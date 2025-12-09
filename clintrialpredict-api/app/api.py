from fastapi import FastAPI
from pydantic import BaseModel, conlist
from app.model import predict, MODEL_FEATURES

app = FastAPI(title="Clinical Trial Prediction API")

class PredictRequest(BaseModel):
    # The input list must match the number of features expected
    features: conlist(float, min_length=len(MODEL_FEATURES), max_length=len(MODEL_FEATURES))

@app.post("/predict")
def predict_endpoint(data: PredictRequest):
    try:
        prediction = predict(data.features)
        return {"prediction": prediction}
    except ValueError as e:
        return {"error": str(e)}
