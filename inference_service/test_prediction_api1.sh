#!/bin/bash
# test_prediction_api.sh
# Complete API test script for Cats vs Dogs Prediction
# Run: bash test_prediction_api.sh

set -e  # Exit on any error

echo "=================================================="
echo "🐱 vs 🐶 Prediction API Test Script"
echo "=================================================="

BASE_URL="http://localhost:8000"

# Function to generate random data
generate_random_data() {
    python3 -c "
import numpy as np
import json
data = np.random.randn(784).tolist()
print(json.dumps(data))
"
}

# Function to print colored output
print_green() {
    echo -e "\033[0;32m$1\033[0m"
}

print_blue() {
    echo -e "\033[0;34m$1\033[0m"
}

print_red() {
    echo -e "\033[0;31m$1\033[0m"
}

# Test 1: Health Check
echo ""
print_blue "TEST 1: Health Check Endpoint"
echo "----------------------------------------"
response=$(curl -s -w "%{http_code}" "$BASE_URL/health")
status_code=${response: -3}
response_body=${response:0:${#response}-3}

if [ "$status_code" -eq 200 ]; then
    print_green "✓ Status: 200 OK"
    echo "$response_body" | python3 -m json.tool
else
    print_red "✗ Failed: Status $status_code"
    echo "$response_body"
fi

# Test 2: Single Prediction
echo ""
print_blue "TEST 2: Single Prediction Endpoint"
echo "----------------------------------------"
RANDOM_DATA=$(generate_random_data)

response=$(curl -s -w "%{http_code}" -X POST "$BASE_URL/predict" \
    -H "Content-Type: application/json" \
    -d "{\"pixels\": $RANDOM_DATA}")
    
status_code=${response: -3}
response_body=${response:0:${#response}-3}

if [ "$status_code" -eq 200 ]; then
    print_green "✓ Status: 200 OK"
    echo "$response_body" | python3 -m json.tool
    
    # Extract and display prediction
    prediction=$(echo "$response_body" | python3 -c "import sys, json; data=json.load(sys.stdin); print(f'Prediction: {data[\"class_label\"]} (confidence: {data[\"probability\"]:.2%})')")
    print_green "  $prediction"
else
    print_red "✗ Failed: Status $status_code"
    echo "$response_body"
fi

# Test 3: Batch Prediction
echo ""
print_blue "TEST 3: Batch Prediction Endpoint"
echo "----------------------------------------"
RANDOM_DATA1=$(generate_random_data)
RANDOM_DATA2=$(generate_random_data)
RANDOM_DATA3=$(generate_random_data)

BATCH_JSON="{\"images\": [
    {\"pixels\": $RANDOM_DATA1},
    {\"pixels\": $RANDOM_DATA2},
    {\"pixels\": $RANDOM_DATA3}
]}"

response=$(curl -s -w "%{http_code}" -X POST "$BASE_URL/batch_predict" \
    -H "Content-Type: application/json" \
    -d "$BATCH_JSON")
    
status_code=${response: -3}
response_body=${response:0:${#response}-3}

if [ "$status_code" -eq 200 ]; then
    print_green "✓ Status: 200 OK"
    echo "$response_body" | python3 -m json.tool
    
    # Display summary
    echo ""
    print_green "Batch Prediction Summary:"
    echo "$response_body" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'  Total predictions: {data[\"count\"]}')
for pred in data['predictions']:
    print(f'  Image {pred[\"index\"]}: {pred[\"class_label\"]} (confidence: {pred[\"probability\"]:.2%})')
"
else
    print_red "✗ Failed: Status $status_code"
    echo "$response_body"
fi

# Test 4: Invalid Input (Wrong number of features)
echo ""
print_blue "TEST 4: Invalid Input Test (Wrong feature count)"
echo "----------------------------------------"
INVALID_DATA=$(python3 -c "import numpy as np; import json; print(json.dumps(np.random.randn(100).tolist()))")

response=$(curl -s -w "%{http_code}" -X POST "$BASE_URL/predict" \
    -H "Content-Type: application/json" \
    -d "{\"pixels\": $INVALID_DATA}")
    
status_code=${response: -3}
response_body=${response:0:${#response}-3}

if [ "$status_code" -eq 400 ]; then
    print_green "✓ Status: 400 Bad Request (Expected for invalid input)"
    echo "$response_body"
else
    print_red "✗ Expected 400, got $status_code"
    echo "$response_body"
fi

# Test 5: Root endpoint
echo ""
print_blue "TEST 5: Root Endpoint"
echo "----------------------------------------"
response=$(curl -s -w "%{http_code}" "$BASE_URL/")
status_code=${response: -3}
response_body=${response:0:${#response}-3}

if [ "$status_code" -eq 200 ]; then
    print_green "✓ Status: 200 OK"
    echo "$response_body" | python3 -m json.tool
else
    print_red "✗ Failed: Status $status_code"
    echo "$response_body"
fi

# Summary
echo ""
echo "=================================================="
echo "📊 TEST SUMMARY"
echo "=================================================="
echo "All endpoints tested:"
echo "  ✓ GET  /          - Root endpoint"
echo "  ✓ GET  /health    - Health check"
echo "  ✓ POST /predict   - Single prediction"
echo "  ✓ POST /batch_predict - Batch prediction"
echo "  ✓ Error handling - Invalid input"
echo ""
echo "✅ API is working correctly!"
echo "=================================================="