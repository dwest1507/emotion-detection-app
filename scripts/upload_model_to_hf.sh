#!/bin/bash
# Script to upload emotion_classifier.onnx, .data file, and model card to Hugging Face Hub
# Usage: ./scripts/upload_model_to_hf.sh [REPO_ID] [MODEL_DIR]
# Example: ./scripts/upload_model_to_hf.sh dwest1507/emotion-detection-model models

set -e

# Default values
REPO_ID="${1:-dwest1507/emotion-detection-model}"
MODEL_DIR="${2:-models}"

MODEL_FILE="${MODEL_DIR}/emotion_classifier.onnx"
MODEL_DATA_FILE="${MODEL_DIR}/emotion_classifier.onnx.data"
MODEL_CARD="${MODEL_DIR}/README.md"

# Check if model file exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found: $MODEL_FILE"
    exit 1
fi

# Check if .data file exists (optional but recommended)
if [ ! -f "$MODEL_DATA_FILE" ]; then
    echo "Warning: Model .data file not found: $MODEL_DATA_FILE"
    echo "The .data file is an optimization cache and is optional."
    echo "Continuing without .data file..."
    MODEL_DATA_FILE=""
fi

# Check if model card exists
if [ ! -f "$MODEL_CARD" ]; then
    echo "Warning: Model card not found: $MODEL_CARD"
    echo "Continuing without model card..."
    MODEL_CARD=""
fi

# Check if huggingface_hub is installed
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "Error: huggingface_hub not installed. Install it with: pip install huggingface-hub"
    exit 1
fi

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. You may need to login first."
    echo "Run: huggingface-cli login"
    echo "Or set HF_TOKEN environment variable"
fi

echo "Uploading model files to Hugging Face Hub..."
echo "Repository: $REPO_ID"
echo "Model file: $MODEL_FILE"
if [ -n "$MODEL_DATA_FILE" ]; then
    echo "Model data file: $MODEL_DATA_FILE"
fi
if [ -n "$MODEL_CARD" ]; then
    echo "Model card: $MODEL_CARD"
fi

# Upload using huggingface_hub Python library
python3 << EOF
from huggingface_hub import HfApi, create_repo
import os
from pathlib import Path

repo_id = "$REPO_ID"
model_file = Path("$MODEL_FILE")
model_data_file = Path("$MODEL_DATA_FILE") if "$MODEL_DATA_FILE" else None
model_card = Path("$MODEL_CARD") if "$MODEL_CARD" else None

# Create repository if it doesn't exist (will fail silently if it exists)
try:
    create_repo(repo_id, exist_ok=True, repo_type="model")
    print(f"✓ Repository {repo_id} is ready")
except Exception as e:
    print(f"Note: {e}")

api = HfApi()

# Upload the model file
print(f"Uploading model file: {model_file.name}...")
api.upload_file(
    path_or_fileobj=str(model_file),
    path_in_repo=model_file.name,
    repo_id=repo_id,
    repo_type="model",
)
print(f"✓ Successfully uploaded {model_file.name}")

# Upload the .data file if it exists
if model_data_file and model_data_file.exists():
    print(f"Uploading model data file: {model_data_file.name}...")
    api.upload_file(
        path_or_fileobj=str(model_data_file),
        path_in_repo=model_data_file.name,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"✓ Successfully uploaded {model_data_file.name}")
else:
    print("⚠ Skipping .data file upload (file not found)")

# Upload the model card (README.md) if it exists
if model_card and model_card.exists():
    print(f"Uploading model card: {model_card.name}...")
    api.upload_file(
        path_or_fileobj=str(model_card),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"✓ Successfully uploaded model card (README.md)")
else:
    print("⚠ Skipping model card upload (file not found)")

print(f"\n✓ Upload complete!")
print(f"  Model available at: https://huggingface.co/{repo_id}")
EOF

echo ""
echo "Upload complete!"
echo "Model URL: https://huggingface.co/$REPO_ID"
echo ""
echo "You can now view your model at: https://huggingface.co/$REPO_ID"

