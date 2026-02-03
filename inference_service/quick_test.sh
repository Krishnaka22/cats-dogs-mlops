#!/bin/bash
# quick_test.sh - Simple one-click test
# Run: bash quick_test.sh

echo "Testing Cats vs Dogs Prediction API..."

# Check if API is running
echo ""
echo "1. Checking API health..."
curl -s http://localhost:8001/health | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'   Status: {data[\"status\"]}')
    print(f'   Model: {data[\"model_type\"]}')
    print(f'   Loaded: {data[\"model_loaded\"]}')
except:
    print('   ❌ Cannot connect to API')
    print('   Make sure Docker container is running:')
    print('   docker run -d -p 8000:8001 cats-dogs-api:test')
    exit(1)
"

# Test prediction
echo ""
echo "2. Testing prediction..."
RANDOM_DATA=$(python3 -c "import numpy as np; import json; print(json.dumps(np.random.randn(784).tolist()))")

curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d "{\"pixels\": $RANDOM_DATA}" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'   Prediction: {data[\"class_label\"]}')
    print(f'   Confidence: {data[\"probability\"]:.2%}')
    print(f'   ✅ Prediction successful!')
except:
    print('   ❌ Prediction failed')
    print(sys.stdin.read())
"


# Batch prediction  
echo ""
curl -X POST http://localhost:8001/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"images": [{"pixels": [...]}, {"pixels": [...]}]}'

echo "✅ Test complete!"
