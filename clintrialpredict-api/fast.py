import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

#from taxifare.ml_logic.registry import load_model
#from taxifare.ml_logic.preprocessor import preprocess_features

import mlflow.pyfunc

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Preload the model to accelerate the predictions
# We want to avoid loading the heavy Deep Learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the Uvicorn server starts
# and then store the model in an `app.state.model` global variable, accessible across all routes!
# This will prove very useful for the Demo Day
app.state.model = load_model()

# http://127.0.0.1:8000/predict?pickup_datetime=2014-07-06+19:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2
@app.get("/predict")
def predict(
        #pickup_datetime: str,  # 2014-07-06 19:18:00
        #pickup_longitude: float,    # -73.950655
        #pickup_latitude: float,     # 40.783282
        #dropoff_longitude: float,   # -73.984365
        #dropoff_latitude: float,    # 40.769802
        #passenger_count: int
    ):      # 1
    """
    Make a single course prediction.
    Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
    Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
    """
    # Convert inputs to a DataFrame
    X_pred = pd.DataFrame({
        #"pickup_datetime": pd.Timestamp(pickup_datetime, tz="US/Eastern"),
        #"pickup_longitude": pickup_longitude,
        #"pickup_latitude": pickup_latitude,
        #"dropoff_longitude": dropoff_longitude,
        #"dropoff_latitude": dropoff_latitude,
        #"passenger_count": passenger_count
    }, index=[0])

    #print("load model")
    # Load model
    #model = load_model(stage="production")
    model = app.state.model
    assert model is not None

    #print("preprocess features")
    # Preprocess
    X_processed = preprocess_features(X_pred)


    #print("run inference")
    # Predict
    y_pred = model.predict(X_processed)

    return {"fare": float(y_pred)}


@app.get("/")
def root():
    return {"greeting": "Hello"}
