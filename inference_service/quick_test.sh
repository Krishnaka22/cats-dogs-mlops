#!/bin/bash
# quick_test.sh - Fixed version with batch prediction
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

# Test single prediction
echo ""
echo "2. Testing single prediction..."
RANDOM_DATA=$(python3 -c "import numpy as np; import json; print(json.dumps(np.random.randn(784).tolist()))")

curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d "{\"pixels\": $RANDOM_DATA}" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'   Prediction: {data[\"class_label\"]}')
    print(f'   Confidence: {data[\"probability\"]:.2%}')
    print(f'   ✅ Single prediction successful!')
except Exception as e:
    print(f'   ❌ Single prediction failed: {e}')
    print(sys.stdin.read())
"

# Test batch prediction (FIXED JSON)
echo ""
echo "3. Testing batch prediction..."
# Generate 3 random data arrays
RANDOM_DATA1=$(python3 -c "import numpy as np; import json; print(json.dumps(np.random.randn(784).tolist()))")
RANDOM_DATA2=$(python3 -c "import numpy as np; import json; print(json.dumps(np.random.randn(784).tolist()))")
RANDOM_DATA3=$(python3 -c "import numpy as np; import json; print(json.dumps(np.random.randn(784).tolist()))")

# Create proper JSON using Python
BATCH_JSON=$(python3 -c "
import json
import numpy as np

# Create 3 random datasets
images = []
for _ in range(3):
    pixels = np.random.randn(784).tolist()
    images.append({'pixels': pixels})

print(json.dumps({'images': images}))
")

curl -s -X POST http://localhost:8001/batch_predict \
  -H "Content-Type: application/json" \
  -d "$BATCH_JSON" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'   ✅ Batch prediction successful!')
    print(f'   Processed {data[\"count\"]} images')
    for pred in data['predictions']:
        print(f'   Image {pred[\"index\"]}: {pred[\"class_label\"]} ({pred[\"probability\"]:.2%})')
except Exception as e:
    print(f'   ❌ Batch prediction failed: {e}')
    print('Response:', sys.stdin.read())
"

echo ""
echo "✅ Test complete!"