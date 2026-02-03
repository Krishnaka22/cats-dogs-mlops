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

````markdown
## Project Structure
```text

cats_dogs_mlops/
├── README.md # This documentation
├── requirements.txt # Python dependencies
├── train.py # Model training script (M1)
├── models/
│ └── logistic_model.pkl # Pre-trained baseline model
│ └── mlp_model.pkl # Alternative neural network model
├── experiment_log.json # Experiment tracking results
├── inference_service/ # API & Containerization (M2)
│ ├── app.py # FastAPI inference service
│ ├── requirements.txt # API dependencies (version pinned)
│ ├── Dockerfile # Container configuration
│ ├── quick_test.sh # One-command API test script
│ ├── test_api.py # Comprehensive API tests
│ └── models/
│ └── logistic_model.pkl # Model for inference service
└── .gitignore # Git exclusions
```

##  Quick Start Guide

### Prerequisites
- Python 3.9+
- Docker
- pip (Python package manager)

### Option A: Test Everything (Recommended for Grading)

**1. Clone the repository:**
```bash
git clone <your-repo-url>
cd cats_dogs_mlops


**2. Run the complete test (Builds, runs, and tests everything):**

bash
cd inference_service
bash quick_test.sh

This will automatically build the Docker image, start the API, run tests, and cleanup.


### Option B: Step-by-Step Testing
1. Train the baseline models (M1):
# Install dependencies
pip install -r requirements.txt

# Train models
python train.py

This creates:

models/logistic_model.pkl - Logistic regression baseline

models/mlp_model.pkl - MLP classifier alternative

experiment_log.json - Experiment tracking results


2. Test the inference API (M2):
cd inference_service

# Install API dependencies
pip install -r requirements.txt

# Build Docker image
docker build -t cats-dogs-api:test .

# Run the container
docker run -d --name cats-dogs-api -p 8000:8000 cats-dogs-api:test

# Test the API
bash quick_test.sh