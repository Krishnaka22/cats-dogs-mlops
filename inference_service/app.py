"""
FastAPI inference service for cats vs dogs classification
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
from typing import List, Dict
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Cats vs Dogs Classification API",
    description="MLOps Assignment - Model Packaging & Containerization",
    version="1.0.0"
)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "logistic_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"✅ Model loaded from {MODEL_PATH}")
    logger.info(f"Model type: {type(model).__name__}")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None

# Pydantic models for request/response validation
class ImageData(BaseModel):
    """Input data model for prediction"""
    # Assuming flattened 32x32x3 image (3072 features)
    pixels: List[float]
    
    class Config:
        schema_extra = {
            "example": {
                "pixels": np.random.randn(784).tolist()  # Example input
            }
        }

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    images: List[ImageData]

class PredictionResponse(BaseModel):
    """Prediction response model"""
    prediction: int  # 0=cat, 1=dog
    probability: float
    class_label: str
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_type: str
    timestamp: str
    message: str

# Routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Cats vs Dogs Classification API",
        "endpoints": {
            "health": "/health (GET)",
            "predict": "/predict (POST)",
            "batch_predict": "/batch_predict (POST)",
            "docs": "/docs (Swagger UI)"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    status = "healthy" if model is not None else "degraded"
    message = "Model loaded and ready" if model is not None else "Model not loaded"
    
    return HealthResponse(
        status=status,
        model_loaded=model is not None,
        model_type=type(model).__name__ if model else "None",
        timestamp=datetime.now().isoformat(),
        message=message
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(image_data: ImageData):
    """
    Single prediction endpoint
    
    - **pixels**: Flattened image pixels (3072 values for 32x32x3 image)
    - Returns: prediction (0=cat, 1=dog), probability, and timestamp
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if pixels_array.shape[1] != 784:  # Change from 3072
    raise HTTPException(...)
    
    try:
        # Convert to numpy array
        pixels_array = np.array(image_data.pixels).reshape(1, -1)
        
        # Validate input shape
        if pixels_array.shape[1] != model.n_features_in_:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {model.n_features_in_} features, got {pixels_array.shape[1]}"
            )
        
        # Make prediction
        prediction = model.predict(pixels_array)[0]
        probability = model.predict_proba(pixels_array)[0][int(prediction)]
        
        # Log prediction
        logger.info(f"Prediction: {prediction} (prob: {probability:.3f})")
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            class_label="dog" if prediction == 1 else "cat",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict", tags=["Prediction"])
async def batch_predict(batch_request: BatchPredictionRequest):
    """
    Batch prediction endpoint
    
    - **images**: List of ImageData objects
    - Returns: List of predictions with probabilities
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        
        for i, image_data in enumerate(batch_request.images):
            pixels_array = np.array(image_data.pixels).reshape(1, -1)
            
            if pixels_array.shape[1] != model.n_features_in_:
                continue  # Skip invalid inputs
                
            prediction = model.predict(pixels_array)[0]
            probability = model.predict_proba(pixels_array)[0][int(prediction)]
            
            predictions.append({
                "index": i,
                "prediction": int(prediction),
                "probability": float(probability),
                "class_label": "dog" if prediction == 1 else "cat"
            })
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )