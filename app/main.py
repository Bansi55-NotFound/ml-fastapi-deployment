import os

import joblib
from fastapi import FastAPI
from app.schema import CancerInput
import pandas as pd

app = FastAPI(title="Breast Cancer Prediction API")

MODEL_PATH = os.path.join("models", "cancer_pipeline.joblib")

model=None


"""@app.on_event("startup")

Runs once when API starts
Model is loaded one time
Stored in memory (model)
Every request can reuse it

📌 This avoids:
Loading model on every API call (very bad practice)"""
@app.on_event("startup")
def load_model():
    global model
    model = joblib.load(MODEL_PATH)
    print("✅ ML model loaded successfully")

@app.get("/")
def health_check():
    return {"status": "API is running"}