
import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path

class DataPreprocessor:
    def __init__(self, config_path='configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_data_path = Path(self.config['data']['raw_path'])
        self.processed_path = Path(self.config['data']['processed_path'])
        self.img_size = tuple(self.config['data']['image_size'])
        
    def load_and_split_data(self):
        """Load images and create train/val/test splits"""
        cats = list((self.raw_data_path / 'Cat').glob('*.jpg'))
        dogs = list((self.raw_data_path / 'Dog').glob('*.jpg'))
        
        # Create labels: 0 for cats, 1 for dogs
        cat_labels = [0] * len(cats)
        dog_labels = [1] * len(dogs)
        
        all_images = cats + dogs
        all_labels = cat_labels + dog_labels
        
        # Split: 80% train, 10% val, 10% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            all_images, all_labels, 
            test_size=0.2, 
            stratify=all_labels,
            random_state=42
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=42
        )
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def preprocess_image(self, image_path):
        """Preprocess single image"""
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0  # Normalize to [0, 1]
        return img
    
    def save_processed_data(self, data_splits):
        """Save processed data to organized directories"""
        for split_name, (images, labels) in data_splits.items():
            split_path = self.processed_path / split_name
            cat_path = split_path / 'cat'
            dog_path = split_path / 'dog'
            
            cat_path.mkdir(parents=True, exist_ok=True)
            dog_path.mkdir(parents=True, exist_ok=True)
            
            for img_path, label in zip(images, labels):
                img = self.preprocess_image(img_path)
                filename = img_path.name
                
                if label == 0:  # Cat
                    save_path = cat_path / filename
                else:  # Dog
                    save_path = dog_path / filename
                
                # Save as numpy array for faster loading
                np.save(str(save_path).replace('.jpg', '.npy'), img)
                
        print(f"Processed data saved to {self.processed_path}")
    
    def run(self):
        """Main preprocessing pipeline"""
        print("Starting data preprocessing...")
        data_splits = self.load_and_split_data()
        self.save_processed_data(data_splits)
        
        # Save split info
        np.save(self.processed_path / 'split_info.npy', {
            'train_size': len(data_splits['train'][0]),
            'val_size': len(data_splits['val'][0]),
            'test_size': len(data_splits['test'][0])
        })
        
        return data_splits

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run()