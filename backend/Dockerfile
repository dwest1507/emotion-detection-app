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
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY models/emotion_classifier.onnx ./models/emotion_classifier.onnx
COPY models/emotion_classifier.onnx.data ./models/emotion_classifier.onnx.data

# Expose port 8000 (default, Railway will override with PORT env var)
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the application
# Use shell form to support PORT environment variable (Railway provides this)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}

