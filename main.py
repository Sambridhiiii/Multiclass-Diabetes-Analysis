# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import pandas as pd

# -----------------------------
# Load model
# -----------------------------
models = {}
try:
    model_pipeline = joblib.load("gb_diabetes_model.joblib")  # Make sure this file exists
    models['gb_model'] = model_pipeline
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(title="Diabetes Prediction API")

# -----------------------------
# Define request schema
# -----------------------------
class DiabetesRequest(BaseModel):
    HbA1c: float
    BMI: float
    AGE: int
    Urea: float
    Chol: float
    VLDL: float
    TG: float
    Cr: float
    LDL: float

@app.post("/predict")
async def predict_diabetes(data: DiabetesRequest):
    model = models.get("gb_model")
    if not model:
        return JSONResponse(
            status_code=500,
            content={"status_code": 500, "message": "Model not loaded"}
        )

    # Prepare features as DataFrame (with column names)
    features_df = pd.DataFrame([{
        'HbA1c': data.HbA1c,
        'BMI': data.BMI,
        'AGE': data.AGE,
        'Urea': data.Urea,
        'Chol': data.Chol,
        'VLDL': data.VLDL,
        'TG': data.TG,
        'Cr': data.Cr,
        'LDL': data.LDL
    }])

    # Make prediction
    prediction = model.predict(features_df)[0]

    # Map numeric class to descriptive label
    class_mapping = {
        0: "Non-diabetic ✅ Healthy blood sugar levels",
        1: "Pre-diabetic ⚠️ At risk: monitor diet and lifestyle",
        2: "Diabetic ❌ High risk: consult a doctor immediately"
    }

    return {
        "status_code": 200,
        "predicted_class": class_mapping[prediction]
    }


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
