"""
Batch prediction script for breast cancer classification.

Reads input data from CSV,
applies trained ML pipeline,
and writes predictions to output CSV.
"""
import os
import joblib
import pandas as pd

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "cancer_pipeline.joblib")
INPUT_CSV_PATH = os.path.join(BASE_DIR, "batch", "sample_input.csv")

def main():
    #load input data
    df = pd.read_csv(INPUT_CSV_PATH)
    print("✅ Input CSV loaded")
    print(df.head())

    #load trained model
    model=joblib.load(MODEL_PATH)
    print("✅ ML model loaded successfully")

    # For now, just confirm shapes
    print(f"Input shape: {df.shape}")

if __name__ == "__main__":
    main()