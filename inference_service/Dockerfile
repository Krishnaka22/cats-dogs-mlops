# Dockerfile for Cats vs Dogs Inference Service

# Use Python 3.9 slim image (stable and lightweight)
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for models
RUN mkdir -p models

# Copy your trained model (ensure it exists locally)
# If your model is in parent directory
COPY ../models/logistic_model.pkl models/

# Verify model exists
RUN if [ ! -f "models/logistic_model.pkl" ]; then \
        echo "WARNING: Model file not found. Creating dummy model for demo." && \
        python -c "from sklearn.linear_model import LogisticRegression; import joblib; import numpy as np; m = LogisticRegression(); X = np.random.randn(10, 3072); y = np.random.randint(0, 2, 10); m.fit(X, y); joblib.dump(m, 'models/logistic_model.pkl')"; \
    fi

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
