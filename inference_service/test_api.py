"""
Test script for the inference service
Run after starting the FastAPI server
"""

import requests
import json
import numpy as np
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\nTesting single prediction...")
    
    # Create random image data (3072 features = 32x32x3)
    random_pixels = np.random.randn(3072).tolist()
    
    payload = {
        "pixels": random_pixels
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error: {response.text}")
            
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\nTesting batch prediction...")
    
    # Create 3 random images
    images = []
    for _ in range(3):
        images.append({
            "pixels": np.random.randn(3072).tolist()
        })
    
    payload = {
        "images": images
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/batch_predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Total predictions: {result['count']}")
            for pred in result['predictions']:
                print(f"  Image {pred['index']}: {pred['class_label']} (prob: {pred['probability']:.3f})")
        else:
            print(f"Error: {response.text}")
            
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_invalid_input():
    """Test with invalid input"""
    print("\nTesting invalid input...")
    
    # Wrong number of features
    payload = {
        "pixels": np.random.randn(100).tolist()  # Should be 3072
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"Expected 400, got: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Cats vs Dogs Inference Service")
    print("=" * 60)
    
    # Wait for server to start
    print("Waiting for server to be ready...")
    time.sleep(3)
    
    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Invalid Input", test_invalid_input)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"TEST: {test_name}")
        print('='*40)
        
        try:
            success = test_func()
            results.append((test_name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results.append((test_name, "ERROR"))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, status in results:
        print(f"{test_name:30} {status}")
    
    print("\nTo test with curl:")
    print("1. Health check: curl http://localhost:8000/health")
    print("2. Single prediction: curl -X POST http://localhost:8000/predict \\")
    print('   -H "Content-Type: application/json" \\')
    print('   -d \'{"pixels": [0.1, -0.2, ...]}\'')

if __name__ == "__main__":
    main()