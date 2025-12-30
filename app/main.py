from fastapi import FastAPI
app = FastAPI(title="Breast Cancer Prediction API")

@app.get("/")
def health_check():
    return {"status": "API is running"}