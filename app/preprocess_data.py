import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pathlib import Path

def preprocess_dataset(input_file, output_file):
    """Clean and prepare the dataset for temporal sampling"""
    # Load dataset
    ds = xr.open_dataset(input_file)
    
    # Replace _FillValue with NaNs for all variables
    for var in ds.data_vars:
        if '_FillValue' in ds[var].attrs:
            fill_value = ds[var]._FillValue
            ds[var] = ds[var].where(ds[var] != fill_value)

    # Interpolate missing values along time
    ds = ds.interpolate_na(dim='valid_time', method='linear', use_coordinate=True)
    
    # Save cleaned dataset
    ds.to_netcdf(output_file)
    print(f"✅ Preprocessed dataset saved to {output_file}")
    
    return ds

def create_temporal_samples(ds, sequence_length=30, forecast_horizon=30, num_sequences=50000):
    """Create temporal samples with proper per-cell scaling"""
    input_vars = ['cp', 'lsp', 'ro', 'tcwv', 'swvl1', 'swvl2']
    print(f"📊 Using variables: {input_vars}")
    
    # Create directory for per-cell scalers
    os.makedirs("cell_scalers", exist_ok=True)
    
    # Initialize storage
    X_samples = []
    lat_len = len(ds['latitude'])
    lon_len = len(ds['longitude'])
    
    # Track progress
    total_cells = lat_len * lon_len
    processed_cells = 0
    
    # Process each grid cell individually
    for lat_idx in range(lat_len):
        for lon_idx in range(lon_len):
            # Collect all timesteps for this cell
            cell_data = []
            for var in input_vars:
                # Get all timesteps for this cell and variable
                var_data = ds[var].isel(latitude=lat_idx, longitude=lon_idx).values
                
                # Create and fit scaler for this specific cell
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(var_data.reshape(-1, 1)).flatten()
                
                # Save cell-specific scaler
                joblib.dump(scaler, f"cell_scalers/{var}_scaler_{lat_idx}_{lon_idx}.pkl")
                cell_data.append(scaled_data)
            
            # Stack variables for this cell (time, features)
            cell_data = np.stack(cell_data, axis=-1)  # shape: (time, num_vars)
            
            # Create sequences for this cell
            for start_time in range(len(cell_data) - sequence_length - forecast_horizon):
                sequence = cell_data[start_time:start_time+sequence_length]
                X_samples.append(sequence)
                
                if len(X_samples) >= num_sequences:
                    break
            
            # Progress tracking
            processed_cells += 1
            if processed_cells % 10 == 0:
                print(f"Processed {processed_cells}/{total_cells} cells", end='\r')
            
            if len(X_samples) >= num_sequences:
                break
        if len(X_samples) >= num_sequences:
            break
    
    X_samples = np.array(X_samples, dtype=np.float32)
    print(f"\n✅ Created {len(X_samples)} samples. Final shape: {X_samples.shape}")
    
    # Save samples
    np.save("X_samples.npy", X_samples)
    print("💾 Saved samples to X_samples.npy")
    
    return X_samples

if __name__ == "__main__":
    # Configure paths
    data_dir = Path("./data")
    input_file = data_dir / "era5_combined_2025.nc"
    output_file = data_dir / "final_preprocessed_dataset.nc"
    
    try:
        # Step 1: Clean and preprocess
        ds = preprocess_dataset(input_file, output_file)
        
        # Step 2: Create temporal samples with proper scaling
        X_samples = create_temporal_samples(ds, num_sequences=10000)
        
        print("\n🎉 Preprocessing complete!")
        print(f"📁 Output files created:")
        print(f"- {output_file} (cleaned dataset)")
        print("- X_samples.npy (temporal samples)")
        print("- cell_scalers/ (per-cell scalers)")
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        if 'ds' in locals():
            print("Available variables:", list(ds.data_vars.keys()))