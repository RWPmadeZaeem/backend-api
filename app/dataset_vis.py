import numpy as np
import os
from pathlib import Path

# 1. Get the correct path to y_pred.npy
# Assuming this structure:
# project/
# ├── app/
# │   └── your_script.py  (this file)
# └── data/
#     └── y_pred.npy

# Method 1: Using pathlib (recommended)
script_dir = Path(__file__).parent  # Gets app/ folder
data_dir = script_dir.parent / "data"  # Goes up to project/ then into data/
pred_path = data_dir / "y_pred.npy"

# Method 2: Using os.path
# script_dir = os.path.dirname(os.path.abspath(__file__))
# data_dir = os.path.join(script_dir, "..", "data")
# pred_path = os.path.join(data_dir, "y_pred.npy")

# 2. Load with verification
try:
    print(f"Loading predictions from: {pred_path}")
    y_pred = np.load(pred_path, allow_pickle=True)
    
    print("\nFile loaded successfully!")
    print("Shape:", y_pred.shape)
    print("Data type:", y_pred.dtype)
    
    # Sample output
    if len(y_pred) > 0:
        print("\nFirst 3 predictions:")
        for i in range(min(3, len(y_pred))):
            print(f"{i}: {y_pred[i]}")
    else:
        print("Warning: Empty prediction file")

except FileNotFoundError:
    print(f"\nError: File not found at {pred_path}")
    print("Current working directory:", os.getcwd())
    print("Contents of data directory:", os.listdir(data_dir))
except Exception as e:
    print(f"\nError loading predictions: {str(e)}")