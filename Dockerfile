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

# Copy entrypoint script
COPY scripts/docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x ./docker-entrypoint.sh

# Download model from Hugging Face Hub
# Use build argument for model repository ID (default: dwest1507/emotion-detection-model)
ARG HF_MODEL_ID=dwest1507/emotion-detection-model
ARG MODEL_FILENAME=emotion_classifier.onnx

# Create models directory and download model files using huggingface_hub
# Download both .onnx and .onnx.data files (if .data exists)
RUN mkdir -p ./models && \
    python3 <<EOF
from huggingface_hub import hf_hub_download
import os

repo_id = '${HF_MODEL_ID}'
model_file = '${MODEL_FILENAME}'
data_file = '${MODEL_FILENAME}.data'

# Download the main model file
hf_hub_download(repo_id=repo_id, filename=model_file, local_dir='./models', local_dir_use_symlinks=False)
print(f'Downloaded {model_file}')

# Try to download the .data file (optional)
try:
    hf_hub_download(repo_id=repo_id, filename=data_file, local_dir='./models', local_dir_use_symlinks=False)
    print(f'Downloaded {data_file}')
except Exception:
    print(f'Note: {data_file} not found (optional optimization cache)')
EOF
RUN echo "Model downloaded from Hugging Face Hub: ${HF_MODEL_ID}"

# Verify the model file is valid (exists and has reasonable size)
RUN if [ ! -f ./models/emotion_classifier.onnx ]; then \
    echo "ERROR: Model file not found after download!" && exit 1; \
    elif [ $(stat -c%s ./models/emotion_classifier.onnx 2>/dev/null || echo 0) -lt 10000 ]; then \
    echo "ERROR: Model file is too small ($(stat -c%s ./models/emotion_classifier.onnx 2>/dev/null || echo 0) bytes), likely corrupted or incomplete." && exit 1; \
    else \
    echo "Model file validation passed. Size: $(stat -c%s ./models/emotion_classifier.onnx) bytes"; \
    fi

# Expose port 8000 (default, Railway will override with PORT env var)
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the application
# Use entrypoint script for proper signal handling while supporting PORT env var
ENTRYPOINT ["./docker-entrypoint.sh"]

