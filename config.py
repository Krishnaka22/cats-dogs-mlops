import os
from pathlib import Path

class Config:
    # Data paths
    DATA_DIR = Path("data")
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    # Model paths
    MODEL_DIR = Path("models")
    MODEL_SAVE_PATH = MODEL_DIR / "baseline_model.h5"
    MODEL_SAVE_PKL = MODEL_DIR / "baseline_model.pkl"
    
    # Training parameters
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 10
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # MLflow tracking
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
    MLFLOW_EXPERIMENT_NAME = "cats_dogs_classification"
    
    # Neptune tracking (optional)
    NEPTUNE_PROJECT = "your-workspace/your-project"
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        dirs = [cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, 
                cls.MODEL_DIR, Path("logs"), Path("artifacts")]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)