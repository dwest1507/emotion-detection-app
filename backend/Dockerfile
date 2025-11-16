# Use Python 3.12.3 slim base image
FROM python:3.12.3-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/app/ ./app/

# Copy model files
# Note: If using Git LFS, ensure Railway is configured to pull LFS files during build
# The .onnx.data file is optional (ONNX Runtime cache) and not required
# Option 1: Copy entire models directory (preferred - works with Git LFS)
COPY backend/models/ ./models/

# Option 2: If model file is not available (e.g., Git LFS not configured in Railway),
# uncomment the following lines and provide MODEL_URL as a build argument:
# ARG MODEL_URL
# RUN if [ -z "$MODEL_URL" ]; then \
#     echo "Error: Model file not found and MODEL_URL not provided. Please either:" && \
#     echo "  1. Configure Git LFS in Railway, or" && \
#     echo "  2. Provide MODEL_URL build argument pointing to model file URL" && \
#     exit 1; \
#     fi
# RUN mkdir -p ./models && \
#     wget -O ./models/emotion_classifier.onnx "$MODEL_URL" || \
#     curl -o ./models/emotion_classifier.onnx "$MODEL_URL"

# Expose port 8000 (default, Railway will override with PORT env var)
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the application
# Use shell form to support PORT environment variable (Railway provides this)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}

