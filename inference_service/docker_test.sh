#!/bin/bash
# docker_test.sh - Complete Docker build and test script
# Run: bash docker_test.sh

set -e  # Exit on error

echo "========================================="
echo "Building and Testing Docker Container"
echo "========================================="

# 1. Build Docker image
echo ""
echo "1. Building Docker image..."
docker build -t cats-dogs-api:test .

# 2. Stop and remove any existing container
echo ""
echo "2. Cleaning up old containers..."
docker stop cats-dogs-test 2>/dev/null || true
docker rm cats-dogs-test 2>/dev/null || true

# 3. Run container
echo ""
echo "3. Starting container..."
docker run -d --name cats-dogs-test -p 8000:8001 cats-dogs-api:test

# 4. Wait for API to start
echo ""
echo "4. Waiting for API to be ready..."
sleep 5

# 5. Test endpoints
echo ""
echo "5. Testing API endpoints..."

echo "   -> Health check:"
curl -s http://localhost:8001/health | python3 -m json.tool

echo ""
echo "   -> Single prediction:"
RANDOM_DATA=$(python3 -c "import numpy as np; print([float(x) for x in np.random.randn(3072)])")
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d "{\"pixels\": $RANDOM_DATA}" | python3 -m json.tool

echo ""
echo "   -> Batch prediction:"
RANDOM_DATA1=$(python3 -c "import numpy as np; print([float(x) for x in np.random.randn(3072)])")
RANDOM_DATA2=$(python3 -c "import numpy as np; print([float(x) for x in np.random.randn(3072)])")
RANDOM_DATA3=$(python3 -c "import numpy as np; print([float(x) for x in np.random.randn(3072)])")

curl -s -X POST http://localhost:8001/batch_predict \
  -H "Content-Type: application/json" \
  -d "{\"images\": [{\"pixels\": $RANDOM_DATA1}, {\"pixels\": $RANDOM_DATA2}, {\"pixels\": $RANDOM_DATA3}]}" | python3 -m json.tool

# 6. Cleanup
echo ""
echo "6. Cleaning up..."
docker stop cats-dogs-test
docker rm cats-dogs-test

echo ""
echo "========================================="
echo "✅ All tests passed successfully!"
echo "========================================="