"""Configuration settings for the emotion detection API."""

from pathlib import Path

# Model configuration
MODEL_PATH = Path(__file__).parent.parent / "models" / "emotion_classifier.onnx"
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
NUM_CLASSES = len(EMOTION_CLASSES)

# Image preprocessing
INPUT_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.6

# File upload limits
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# MediaPipe configuration
MEDIAPIPE_MODEL_SELECTION = 1  # 1 = full range, 0 = short range
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
FACE_CROP_PADDING = 0.2  # 20% padding around detected face

# Model metadata
MODEL_VERSION = "1.0.0"
MODEL_ARCHITECTURE = "EfficientNet-B0"
MODEL_DATASET = "FER-2013 Dataset"

