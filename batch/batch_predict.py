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


FEATURE_NAME_MAPPING = {
    "mean_radius": "mean radius",
    "mean_texture": "mean texture",
    "mean_perimeter": "mean perimeter",
    "mean_area": "mean area",
    "mean_smoothness": "mean smoothness",
    "mean_compactness": "mean compactness",
    "mean_concavity": "mean concavity",
    "mean_concave_points": "mean concave points",
    "mean_symmetry": "mean symmetry",
    "mean_fractal_dimension": "mean fractal dimension",

    "radius_error": "radius error",
    "texture_error": "texture error",
    "perimeter_error": "perimeter error",
    "area_error": "area error",
    "smoothness_error": "smoothness error",
    "compactness_error": "compactness error",
    "concavity_error": "concavity error",
    "concave_points_error": "concave points error",
    "symmetry_error": "symmetry error",
    "fractal_dimension_error": "fractal dimension error",

    "worst_radius": "worst radius",
    "worst_texture": "worst texture",
    "worst_perimeter": "worst perimeter",
    "worst_area": "worst area",
    "worst_smoothness": "worst smoothness",
    "worst_compactness": "worst compactness",
    "worst_concavity": "worst concavity",
    "worst_concave_points": "worst concave points",
    "worst_symmetry": "worst symmetry",
    "worst_fractal_dimension": "worst fractal dimension"
}

def main():
    #load input data
    df = pd.read_csv(INPUT_CSV_PATH)
    print("✅ Input CSV loaded")
    print(df.head())

    # Rename columns to match training feature names
    df.rename(columns=FEATURE_NAME_MAPPING, inplace=True)
    print("✅ Feature names mapped to training format")

    #load trained model
    model=joblib.load(MODEL_PATH)
    print("✅ ML model loaded successfully")

    # For now, just confirm shapes
    print(f"Input shape: {df.shape}")

    predictions = model.predict(df)
    probabilities = model.predict_proba(df)

    label_map = {0: "malignant", 1: "benign"}

    df["prediction_code"] = predictions
    df["prediction"] = [label_map[p] for p in predictions]
    df["probability"] = probabilities.max(axis=1)

    output_path = os.path.join(BASE_DIR, "batch", "predictions_output.csv")
    df.to_csv(output_path, index=False)

    print(f"✅ Predictions saved to {output_path}")

if __name__ == "__main__":
    main()