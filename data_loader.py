import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

class CatsDogsDataLoader:
    def __init__(self, config):
        self.config = config
        
    def download_kaggle_data(self):
        """Download cats and dogs dataset from Kaggle"""
        try:
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                'chetankv/dogs-cats-images',
                path=self.config.RAW_DATA_DIR,
                unzip=True
            )
            print("Dataset downloaded successfully!")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please download manually from: https://www.kaggle.com/datasets/chetankv/dogs-cats-images")
    
    def load_data(self):
        """Load and preprocess the dataset"""
        # Create data generators
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=self.config.VALIDATION_SPLIT
        )
        
        # Load training data
        train_generator = datagen.flow_from_directory(
            self.config.RAW_DATA_DIR / "dataset" / "training_set",
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        # Load validation data
        val_generator = datagen.flow_from_directory(
            self.config.RAW_DATA_DIR / "dataset" / "training_set",
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )
        
        # Load test data
        test_generator = datagen.flow_from_directory(
            self.config.RAW_DATA_DIR / "dataset" / "test_set",
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )
        
        return train_generator, val_generator, test_generator
    
    def prepare_data_for_logistic(self, generator):
        """Prepare data for logistic regression (flatten pixels)"""
        X, y = [], []
        num_batches = len(generator)
        
        for i in range(num_batches):
            batch_X, batch_y = generator[i]
            batch_X_flat = batch_X.reshape(batch_X.shape[0], -1)
            X.append(batch_X_flat)
            y.append(batch_y)
            
        X = np.vstack(X)
        y = np.hstack(y)
        
        return X, y