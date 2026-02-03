# Cats vs Dogs Classification - MLOps Assignment

## Project Overview
Implementation of baseline ML models for cats vs dogs classification with MLOps practices including experiment tracking, model serialization, and containerized deployment.

## Assignment Components

### M1: Model Building & Experiment Tracking
- **Baseline Models**: Logistic Regression & CNN (scikit-learn alternative)
- **Model Serialization**: Models saved as `.h5` and `.pkl` files
- **Experiment Tracking**: MLflow integration for parameter/metric logging
- **Artifacts**: Training curves, confusion matrices, model cards

### M2: Model Packaging & Containerization  
- **Inference Service**: FastAPI REST API with health check and prediction endpoints
- **Containerization**: Dockerized service with dependency versioning
- **API Endpoints**: 
  - `GET /health` - Service health check
  - `POST /predict` - Single image prediction
  - `POST /batch_predict` - Batch predictions

## Project Structure
cats_dogs_mlops/
├── train.py # Model training script (M1)
├── models/ # Trained models
├── inference_service/ # FastAPI service (M2)
│ ├── app.py # Main FastAPI application
│ ├── requirements.txt # Dependencies with version pinning
│ ├── Dockerfile # Container configuration
│ ├── quick_test.sh # API test script
│ └── models/ # Model for inference
└── README.md # This file