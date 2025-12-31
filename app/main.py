import os

import joblib
from fastapi import FastAPI
from app.schema import CancerInput
import pandas as pd

app = FastAPI(title="Breast Cancer Prediction API")

MODEL_PATH = os.path.join("models", "cancer_pipeline.joblib")

model = None

# Runs once when API starts
# Model is loaded one time
# Stored in memory (model)
# Every request can reuse it
# Avoids loading model on every request


@app.on_event("startup")
def load_model():
    global model
    model = joblib.load(MODEL_PATH)
    print("✅ ML model loaded successfully")


@app.post("/predict")
def predict(data: CancerInput):
    """
        Predict whether a breast tumor is benign or malignant.

        This endpoint accepts breast cancer feature values as JSON input,
        applies the trained machine learning pipeline, and returns:
        - predicted class (benign / malignant)
        - prediction code (0 or 1)
        - probability score for the predicted class

        Input:
            data (CancerInput): Validated input features required by the model

        Returns:
            dict: Prediction result containing label, code, and probability
        """
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

    # convert input data to dataframe
    input_df = pd.DataFrame([data.dict()])
    input_df.rename(columns=FEATURE_NAME_MAPPING, inplace=True)

    #make predictions
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][prediction]

    label_map = {0: "malignant", 1: "benign"}
    return {"prediction_code": int(prediction), "prediction": label_map[prediction],
            "probability": round(float(probability), 4)}


@app.get("/")
def health_check():
    return {"status": "API is running"}
