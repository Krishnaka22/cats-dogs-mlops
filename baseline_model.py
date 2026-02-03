import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
import mlflow.keras
from datetime import datetime
import json

class BaselineModels:
    def __init__(self, config):
        self.config = config
        self.history = None
        
    def create_simple_cnn(self):
        """Create a simple CNN baseline model"""
        model = keras.Sequential([
            layers.Rescaling(1./255, input_shape=(*self.config.IMG_SIZE, 3)),
            
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC()]
        )
        
        return model
    
    def create_logistic_regression(self, input_shape):
        """Create logistic regression model on flattened pixels"""
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            verbose=1
        )
        return model
    
    def plot_training_history(self, history, run_name=None):
        """Plot and save training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        axes[0].plot(history.history['loss'], label='Training Loss')
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Loss Curves')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy
        axes[1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Accuracy Curves')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"artifacts/training_curves_{run_name or timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_confusion_matrix(self, y_true, y_pred, run_name=None):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred > 0.5)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Cat', 'Dog'],
                   yticklabels=['Cat', 'Dog'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cm_path = f"artifacts/confusion_matrix_{run_name or timestamp}.png"
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return cm_path, cm
    
    def save_model(self, model, model_type='cnn'):
        """Save model in appropriate format"""
        if model_type == 'cnn':
            # Save as Keras .h5 format
            model.save(self.config.MODEL_SAVE_PATH)
            print(f"Model saved as {self.config.MODEL_SAVE_PATH}")
            
            # Also save in TensorFlow SavedModel format
            tf_model_path = self.config.MODEL_DIR / "baseline_tf_model"
            model.save(tf_model_path)
            print(f"Model saved in TF format at {tf_model_path}")
            
        elif model_type == 'logistic':
            # Save as .pkl using joblib
            joblib.dump(model, self.config.MODEL_SAVE_PKL)
            print(f"Model saved as {self.config.MODEL_SAVE_PKL}")