"""
FastAPI inference service for cats vs dogs classification
Includes Monitoring & Model Performance Tracking (M5)
"""

import time
import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import numpy as np
import joblib
import os
from typing import List
from datetime import datetime

# --------------------------------------------------
# Logging Setup
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cats_dogs_api")

# --------------------------------------------------
# FastAPI Initialization
# --------------------------------------------------
app = FastAPI(
    title="Cats vs Dogs Classification API",
    description="MLOps Assignment - Model Packaging, Monitoring & Evaluation",
    version="1.2.0"
)

# --------------------------------------------------
# Monitoring Variables (M5 Part 1)
# --------------------------------------------------
request_count = 0
total_latency = 0.0

# --------------------------------------------------
# Model Performance Tracking (M5 Part 2)
# --------------------------------------------------
evaluation_records = []

# --------------------------------------------------
# Load Model
# --------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "logistic_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# --------------------------------------------------
# Middleware for Logging + Latency Tracking
# --------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    global request_count, total_latency

    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    request_count += 1
    total_latency += process_time

    logger.info(
        f"{request.method} {request.url.path} | "
        f"Status: {response.status_code} | "
        f"Latency: {process_time:.4f}s"
    )

    return response

# --------------------------------------------------
# Pydantic Models
# --------------------------------------------------
class ImageData(BaseModel):
    pixels: List[float]

class BatchPredictionRequest(BaseModel):
    images: List[ImageData]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    class_label: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    timestamp: str
    message: str

class EvaluationRequest(BaseModel):
    pixels: List[float]
    true_label: int  # 0 = cat, 1 = dog

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Cats vs Dogs Classification API",
        "endpoints": {
            "health": "/health (GET)",
            "predict": "/predict (POST)",
            "batch_predict": "/batch_predict (POST)",
            "metrics": "/metrics (GET)",
            "evaluate": "/evaluate (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    status = "healthy" if model is not None else "degraded"
    message = "Model loaded and ready" if model else "Model not loaded"

    return HealthResponse(
        status=status,
        model_loaded=model is not None,
        model_type=type(model).__name__ if model else "None",
        timestamp=datetime.now().isoformat(),
        message=message
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(image_data: ImageData):

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        pixels_array = np.array(image_data.pixels).reshape(1, -1)

        if pixels_array.shape[1] != model.n_features_in_:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {model.n_features_in_} features, got {pixels_array.shape[1]}"
            )

        prediction = model.predict(pixels_array)[0]
        probability = model.predict_proba(pixels_array)[0][int(prediction)]

        logger.info(f"Prediction: {prediction} | Probability: {probability:.4f}")

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

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        predictions = []

        for i, image_data in enumerate(batch_request.images):
            pixels_array = np.array(image_data.pixels).reshape(1, -1)

            if pixels_array.shape[1] != model.n_features_in_:
                continue

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

# --------------------------------------------------
# Metrics Endpoint (Monitoring)
# --------------------------------------------------
@app.get("/metrics")
async def metrics():
    avg_latency = total_latency / request_count if request_count > 0 else 0

    return {
        "request_count": request_count,
        "average_latency_seconds": round(avg_latency, 4),
        "timestamp": datetime.now().isoformat()
    }

# --------------------------------------------------
# Evaluation Endpoint (Model Performance Tracking)
# --------------------------------------------------
@app.post("/evaluate")
async def evaluate_model(data: EvaluationRequest):

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    pixels_array = np.array(data.pixels).reshape(1, -1)

    if pixels_array.shape[1] != model.n_features_in_:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {model.n_features_in_} features"
        )

    prediction = model.predict(pixels_array)[0]

    evaluation_records.append({
        "prediction": int(prediction),
        "true_label": int(data.true_label)
    })

    correct = sum(
        1 for record in evaluation_records
        if record["prediction"] == record["true_label"]
    )

    accuracy = correct / len(evaluation_records)

    return {
        "total_samples": len(evaluation_records),
        "current_accuracy": round(accuracy, 4),
        "timestamp": datetime.now().isoformat()
    }

# --------------------------------------------------
# Run Server
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
