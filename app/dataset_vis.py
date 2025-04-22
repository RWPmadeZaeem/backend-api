import xarray as xr

# Open the dataset

ds = xr.open_dataset("c:/Users/786/Desktop/FloodForecast/backend-api/data/processed_data_scaled.nc")

# Print dataset summary
print(ds)

# Correct way to check dimensions
print("\nDimensions:", ds.dims)  # Not ds.dimensions

# Check coordinates
print("\nCoordinates:", ds.coords)

# Check variables
print("\nData variables:", list(ds.data_vars))

# Check attributes
print("\nGlobal attributes:", ds.attrs)

# Close the dataset (good practice)
ds.close()