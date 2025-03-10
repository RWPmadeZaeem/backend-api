from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import xarray as xr
import pandas as pd
import joblib
import pickle
import os
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="Flood Prediction API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your Next.js app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DAYS_INTAKE_LENGTH = 180
FORECAST_DAY = 30
MODEL_PATH = "../ml_models/regressor.pkl"
SCALER_X_PATH = "ml_models/scaler_x.pkl"
SCALER_Y_PATH = "ml_models/scaler_y.pkl"

# Dummy thresholds for risk assessment (can be calibrated with historical data)
THRESHOLDS = {
    "low": 0.3,      # Discharge difference below 0.3 is low risk
    "medium": 0.6,   # Discharge difference between 0.3 and 0.6 is medium risk
    "high": 0.9,     # Discharge difference between 0.6 and 0.9 is high risk
    "severe": 1.0    # Discharge difference above 0.9 is severe risk
}

# Model and scalers
model = None
scaler_x = None
scaler_y = None

class PredictionRequest(BaseModel):
    file_path: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    predicted_discharge: float
    threshold_value: float
    risk_percentage: float
    risk_level: str
    forecast_data: List[float] = []

def shift_input(X_scaled, y_scaled, days_intake_length, forecast_day=0):
    """
    Shifts the datasets in time to be fitted into the LSTM.
    """
    X_train = []
    y_train = []

    for i in range(X_scaled.shape[1]):
        feature_array = []
        for j in range(days_intake_length, len(X_scaled)-forecast_day):
            feature_array.append(X_scaled[j - days_intake_length:j, i])
        X_train.append(feature_array)

    y_feature_array = []
    for i in range(days_intake_length, len(y_scaled)-forecast_day):
        y_feature_array.append(y_scaled[i - days_intake_length:i, 0])
    X_train.append(y_feature_array)

    for i in range(days_intake_length, len(y_scaled)-forecast_day):
        y_train.append(y_scaled[i:i+forecast_day, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[1], X_train.shape[2], X_train.shape[0]))

    return X_train, y_train

def get_risk_level(discharge_value):
    """Determine risk level based on discharge value"""
    if discharge_value < THRESHOLDS["low"]:
        return "Low"
    elif discharge_value < THRESHOLDS["medium"]:
        return "Medium"
    elif discharge_value < THRESHOLDS["high"]:
        return "High"
    else:
        return "Severe"

def calculate_risk_percentage(discharge_value):
    """Calculate risk percentage based on thresholds"""
    risk_percentage = min(100, (discharge_value / THRESHOLDS["severe"]) * 100)
    return round(risk_percentage, 2)

@app.on_event("startup")
async def startup_event():
    """Load model and scalers when the app starts"""
    global model, scaler_x, scaler_y
    
    try:
        # Load the trained model from pickle file
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        # Load the fitted scalers
        scaler_x = joblib.load(SCALER_X_PATH) if os.path.exists(SCALER_X_PATH) else None
        scaler_y = joblib.load(SCALER_Y_PATH) if os.path.exists(SCALER_Y_PATH) else None
        
        print("Model loaded successfully")
        
        # If scalers don't exist, print a warning
        if scaler_x is None or scaler_y is None:
            print("Warning: Scalers not found. Make sure to provide them or handle data normalization manually.")
    except Exception as e:
        print(f"Error loading model or scalers: {e}")

@app.get("/")
async def root():
    return {"message": "Flood Prediction API is running"}



@app.post("/predict", response_model=PredictionResponse)
async def predict():
    """
    Endpoint to make flood predictions using the preloaded dataset features_xy_2.nc.
    """
    global model, scaler_x, scaler_y

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")

    try:
        # Load the dataset from the fixed path
        dataset_path = "../data/features_xy_2.nc"
        ds = xr.open_dataset(dataset_path)
        pd_ds = ds.to_dataframe().dropna()

        # Prepare features and target
        X = pd_ds.drop(columns=['dis_diff']) if 'dis_diff' in pd_ds.columns else pd_ds
        y = pd_ds['dis_diff'] if 'dis_diff' in pd_ds.columns else np.zeros(len(X))

        # Convert to numpy
        X = X.to_numpy()
        y = y.to_numpy().reshape(-1, 1)

        # Scale data if scalers are available
        if scaler_x is not None and scaler_y is not None:
            X_scaled = scaler_x.transform(X)
            y_scaled = scaler_y.transform(y)
        else:
            X_scaled = X
            y_scaled = y

        # Prepare for LSTM
        X_final, _ = shift_input(X_scaled, y_scaled, DAYS_INTAKE_LENGTH, 1)

        # Make prediction
        y_pred_scaled = model.predict(X_final)

        # Inverse transform to get actual values if scalers are available
        if scaler_y is not None:
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
        else:
            y_pred = y_pred_scaled.reshape(-1, 1)

        # Get the latest prediction
        latest_prediction = float(y_pred[-1][0])

        # Calculate risk metrics
        threshold_value = THRESHOLDS["medium"]
        risk_percentage = calculate_risk_percentage(latest_prediction)
        risk_level = get_risk_level(latest_prediction)

        # Prepare forecast data (last 10 predictions for visualization)
        forecast_data = [float(val[0]) for val in y_pred[-10:]]

        return PredictionResponse(
            predicted_discharge=latest_prediction,
            threshold_value=threshold_value,
            risk_percentage=risk_percentage,
            risk_level=risk_level,
            forecast_data=forecast_data
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/thresholds", response_model=Dict[str, float])
async def get_thresholds():
    """Return the current threshold values"""
    return THRESHOLDS