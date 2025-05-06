from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import geojson
from shapely.geometry import Polygon
import numpy as np
from typing import List, Optional
import json
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store data
grid_data = None
predictions = None

def initialize_grid():
    """Initialize the grid with Pakistan coverage"""
    global grid_data
    
    lon_start, lon_end = 60.5, 77.5
    lat_start, lat_end = 37.0, 23.5
    lon_step, lat_step = 0.25, 0.25

    features = []

    lat = lat_start
    cell_index = 0
    while lat > lat_end:
        lon = lon_start
        while lon < lon_end:
            polygon = Polygon([
                [lon, lat],
                [lon + lon_step, lat],
                [lon + lon_step, lat - lat_step],
                [lon, lat - lat_step],
                [lon, lat]
            ])
            feature = geojson.Feature(
                geometry=polygon,
                properties={
                    "id": f"cell-{round(lat,2)}-{round(lon,2)}",
                    "cell_id": cell_index,
                    "risk": "Unknown",
                    "predictions": None,
                    "flood_days": 0,
                    "first_flood_day": None,
                    "last_flood_day": None
                }
            )
            features.append(feature)
            cell_index += 1
            lon += lon_step
        lat -= lat_step

    grid_data = geojson.FeatureCollection(features)

def load_predictions(prediction_data: np.ndarray):
    """Map predictions to grid cells with enhanced flood analysis"""
    global grid_data, predictions
    
    predictions = prediction_data
    
    # Validate predictions shape
    if predictions.shape != (1173, 30, 1):
        raise ValueError(f"Expected predictions shape (1173, 30, 1), got {predictions.shape}")
    
    # Update grid features with predictions and flood analysis
    for feature in grid_data['features']:
        cell_id = feature['properties']['cell_id']
        cell_predictions = predictions[cell_id].flatten().tolist()
        
        # Calculate flood statistics
        flood_days = sum(cell_predictions)
        flood_indices = [i for i, pred in enumerate(cell_predictions) if pred == 1]
        first_flood_day = flood_indices[0] + 1 if flood_indices else None
        last_flood_day = flood_indices[-1] + 1 if flood_indices else None
        
        # Determine risk level
        if flood_days == 0:
            risk = "Low"
        elif flood_days <= 5:
            risk = "Medium"
        elif flood_days <= 15:
            risk = "High"
        else:
            risk = "Extreme"
            
        # Update feature properties
        feature['properties'].update({
            "predictions": cell_predictions,
            "risk": risk,
            "flood_days": flood_days,
            "first_flood_day": first_flood_day,
            "last_flood_day": last_flood_day
        })

def load_predictions_from_file(file_path: str):
    """Load predictions from .npy file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prediction file not found: {file_path}")
    
    try:
        pred_data = np.load(file_path)
        load_predictions(pred_data)
        return True
    except Exception as e:
        raise RuntimeError(f"Error loading predictions: {str(e)}")

# Initialize grid and load predictions on startup
initialize_grid()

# Try to load predictions from file if available
PREDICTION_FILE = "y_pred.npy"
try:
    if os.path.exists(PREDICTION_FILE):
        load_predictions_from_file(PREDICTION_FILE)
        print(f"Successfully loaded predictions from {PREDICTION_FILE}")
    else:
        print(f"No prediction file found at {PREDICTION_FILE}, using dummy data")
        dummy_predictions = np.zeros((1173, 30, 1), dtype=np.int64)
        dummy_predictions[2, -5:] = 1  # Sample flood prediction
        load_predictions(dummy_predictions)
except Exception as e:
    print(f"Error loading predictions: {e}")
    # Fallback to empty predictions
    load_predictions(np.zeros((1173, 30, 1), dtype=np.int64))

@app.get("/grid", response_model=dict)
async def get_grid():
    """Return the complete grid GeoJSON with current predictions"""
    if grid_data is None:
        initialize_grid()
    return grid_data

@app.get("/grid_simplified")
async def get_grid_simplified():
    """Return a simplified version of the grid with only essential data"""
    if grid_data is None:
        initialize_grid()
    
    simplified_features = []
    for feature in grid_data['features']:
        simplified_features.append({
            "type": "Feature",
            "geometry": feature['geometry'],
            "properties": {
                "id": feature['properties']['id'],
                "risk": feature['properties']['risk'],
                "flood_days": feature['properties']['flood_days']
            }
        })
    
    return {
        "type": "FeatureCollection",
        "features": simplified_features
    }

@app.post("/update_predictions")
async def update_predictions(new_predictions: List[List[int]]):
    """Update predictions from external source"""
    try:
        np_predictions = np.array(new_predictions).reshape((1173, 30, 1))
        load_predictions(np_predictions)
        return {
            "message": "Predictions updated successfully",
            "stats": {
                "total_cells": len(grid_data['features']),
                "flood_affected_cells": sum(1 for f in grid_data['features'] if f['properties']['flood_days'] > 0),
                "high_risk_cells": sum(1 for f in grid_data['features'] if f['properties']['risk'] in ["High", "Extreme"])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/update_predictions_file")
async def update_predictions_file():
    """Reload predictions from the file"""
    try:
        success = load_predictions_from_file(PREDICTION_FILE)
        return {"message": "Predictions reloaded from file", "success": success}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/grid_cell/{cell_id}")
async def get_grid_cell(cell_id: int):
    """Get detailed information for a specific grid cell"""
    if grid_data is None:
        initialize_grid()
    
    if cell_id < 0 or cell_id >= len(grid_data['features']):
        raise HTTPException(status_code=404, detail="Invalid cell ID")
    
    feature = grid_data['features'][cell_id]
    return feature['properties']

@app.get("/flood_stats")
async def get_flood_stats():
    """Get summary statistics about flood predictions"""
    if grid_data is None:
        initialize_grid()
    
    total_cells = len(grid_data['features'])
    flood_cells = sum(1 for f in grid_data['features'] if f['properties']['flood_days'] > 0)
    
    return {
        "total_cells": total_cells,
        "flood_affected_cells": flood_cells,
        "percentage_affected": (flood_cells / total_cells) * 100 if total_cells > 0 else 0,
        "risk_distribution": {
            "Extreme": sum(1 for f in grid_data['features'] if f['properties']['risk'] == "Extreme"),
            "High": sum(1 for f in grid_data['features'] if f['properties']['risk'] == "High"),
            "Medium": sum(1 for f in grid_data['features'] if f['properties']['risk'] == "Medium"),
            "Low": sum(1 for f in grid_data['features'] if f['properties']['risk'] == "Low")
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)