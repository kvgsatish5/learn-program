from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import numpy as np
from model_train import train

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
MODEL_PATH = "model.pkl"
model = None

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Training now...")
        train()
    model = joblib.load(MODEL_PATH)

class PredictionInput(BaseModel):
    stream: int
    percentage: float
    age: int
    additional_skills_count: int

@app.post("/predict")
async def predict(data: PredictionInput):
    # Prepare input for prediction
    input_features = np.array([[
        data.stream, 
        data.percentage, 
        data.age, 
        data.additional_skills_count
    ]])
    
    # Simple mapping for labels
    labels = ["Software Engineer", "Data Analyst", "Mechanical Engineer", "Business Analyst"]
    
    prediction_idx = model.predict(input_features)[0]
    probabilities = model.predict_proba(input_features)[0]
    confidence = float(np.max(probabilities))
    
    return {
        "prediction": labels[prediction_idx],
        "confidence": confidence
    }

@app.get("/")
def read_root():
    return {"message": "Job Opportunity Prediction API is running"}
