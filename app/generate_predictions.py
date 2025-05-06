import numpy as np
import xarray as xr
import geojson
import json
from shapely.geometry import shape
from tensorflow import keras
from keras.models import model_from_json
import joblib
from tqdm import tqdm
import pandas as pd
import os
from pathlib import Path

# 1. Set up all paths
script_dir = Path(__file__).parent
root_dir = script_dir.parent

# Model files
models_dir = root_dir / "ml_models"
config_path = models_dir / "config.json"
weights_path = models_dir / "model.weights.h5"

# Data files
data_dir = root_dir / "data"
grid_path = script_dir / "pakistan-grid.geojson"
ds_path = data_dir / "final_preprocessed_dataset.nc"
output_path = data_dir / "predictions_output.parquet"
cell_scalers_dir = root_dir / "cell_scalers"  # Where per-cell scalers are stored

# 2. Verify all files exist
required_files = {
    "Model config": config_path,
    "Model weights": weights_path,
    "GeoJSON grid": grid_path,
    "Dataset": ds_path
}

print("Verifying files...")
missing = [name for name, path in required_files.items() if not path.exists()]
if missing:
    raise FileNotFoundError(f"Missing files: {', '.join(missing)}")
print("✅ All files found")

# 3. Load resources
with open(config_path, "r") as f:
    model = model_from_json(f.read())
model.load_weights(weights_path)

with open(grid_path, "r") as f:
    grid = geojson.load(f)

ds = xr.open_dataset(ds_path)

# 4. Prediction function with per-cell scaling
def get_cell_scaler(var, lat_idx, lon_idx):
    """Load cell-specific scaler"""
    scaler_path = cell_scalers_dir / f"{var}_scaler_{lat_idx}_{lon_idx}.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    return joblib.load(scaler_path)

# Prepare output storage
predictions = []
time_indices = range(30, len(ds['valid_time']))
chunk_size = 10000

# Process each grid cell
for feature in tqdm(grid['features'], desc="Processing grid cells"):
    centroid = shape(feature['geometry']).centroid
    lon, lat = centroid.x, centroid.y
    
    lat_idx = int((37.0 - lat)/0.25)
    lon_idx = int((lon - 60.5)/0.25)
    
    try:
        # Load all scalers for this cell
        cell_scalers = {
            var: get_cell_scaler(var, lat_idx, lon_idx)
            for var in ['cp', 'lsp', 'ro', 'tcwv', 'swvl1', 'swvl2']
        }
    except FileNotFoundError as e:
        print(f"Skipping cell at lat {lat}, lon {lon}: {str(e)}")
        continue
        
    for time_idx in tqdm(time_indices, desc="Time steps", leave=False):
        try:
            input_data = []
            for var in ['cp', 'lsp', 'ro', 'tcwv', 'swvl1', 'swvl2']:
                var_data = ds[var].isel(
                    latitude=lat_idx,
                    longitude=lon_idx,
                    valid_time=slice(time_idx-30, time_idx)
                ).values
                
                # Scale using cell-specific scaler
                scaled_data = cell_scalers[var].transform(var_data.reshape(-1, 1)).flatten()
                input_data.append(scaled_data)
            
            X = np.stack(input_data, axis=-1).reshape(1, 30, 6)
            pred = model.predict(X, verbose=0)
            
            predictions.append({
                "cell_id": feature['properties']['id'],
                "latitude": lat,
                "longitude": lon,
                "date": pd.to_datetime(str(ds['valid_time'].values[time_idx])).strftime('%Y-%m-%d'),
                "risk_score": float(np.mean(pred > 0.5)),
                **{var: float(ds[var].isel(
                    latitude=lat_idx,
                    longitude=lon_idx,
                    valid_time=time_idx
                ).values) for var in ['cp', 'lsp', 'ro', 'tcwv', 'swvl1', 'swvl2']}
            })
            
            if len(predictions) % chunk_size == 0:
                pd.DataFrame(predictions).to_parquet(data_dir / f"predictions_chunk_{len(predictions)}.parquet")
                predictions = []
                
        except Exception as e:
            print(f"Error at lat {lat}, lon {lon}, time {time_idx}: {str(e)}")
            continue

# Save final chunk
if predictions:
    df = pd.DataFrame(predictions)
    df.to_parquet(data_dir / "all_predictions.parquet")

# Update GeoJSON
latest_date = pd.to_datetime(str(ds['valid_time'].values[-1])).strftime('%Y-%m-%d')
latest_df = pd.read_parquet(data_dir / "all_predictions.parquet")
latest_df = latest_df[latest_df['date'] == latest_date]

for feature in grid['features']:
    cell_id = feature['properties']['id']
    cell_data = latest_df[latest_df['cell_id'] == cell_id]
    
    if not cell_data.empty:
        risk = cell_data.iloc[0]['risk_score']
        feature['properties']['risk'] = "High" if risk > 0.7 else "Medium" if risk > 0.3 else "Low"
        feature['properties']['prediction'] = risk

with open(script_dir / "pakistan-grid-with-predictions.geojson", "w") as f:
    geojson.dump(grid, f)

print("✅ All predictions generated and saved")