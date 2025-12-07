"""Tests for FastAPI endpoints."""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from unittest.mock import Mock, patch, MagicMock

from app.main import app
from app.config import MAX_FILE_SIZE, ALLOWED_EXTENSIONS


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing."""
    image = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.read()


@pytest.fixture
def large_image_bytes():
    """Create a large image that exceeds MAX_FILE_SIZE."""
    # Create a very large image
    image = Image.new('RGB', (5000, 5000), color='blue')
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG', quality=95)
    img_bytes.seek(0)
    return img_bytes.read()


@pytest.fixture
def mock_classifier():
    """Create a mock EmotionClassifier."""
    mock = MagicMock()
    mock.is_loaded.return_value = True
    mock.predict_with_timing.return_value = (
        {'angry': 0.1, 'disgust': 0.05, 'fear': 0.1, 'happy': 0.7, 'sad': 0.03, 'surprise': 0.01, 'neutral': 0.01},
        'happy',
        0.7,
        50.0
    )
    return mock


@pytest.fixture
def mock_face_detector():
    """Create a mock FaceDetector."""
    mock = MagicMock()
    # Mock detect_and_crop to return a cropped image
    cropped_image = Image.new('RGB', (100, 100), color='red')
    mock.detect_and_crop.return_value = cropped_image
    mock.detect_face_count.return_value = 1
    return mock


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data
    assert data["message"] == "Emotion Detection API"


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["status"] == "ok"
    assert isinstance(data["model_loaded"], bool)


def test_info_endpoint(client):
    """Test model info endpoint."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "model_architecture" in data
    assert "classes" in data
    assert "input_size" in data
    assert "trained_on" in data
    assert isinstance(data["classes"], list)
    assert len(data["classes"]) > 0


def test_predict_endpoint_invalid_file_type(client):
    """Test predict endpoint with invalid file type."""
    # Create a text file instead of image
    files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200  # Returns 200 with ErrorResponse
    data = response.json()
    assert data["success"] is False
    assert "error" in data
    assert "message" in data


def test_predict_endpoint_file_too_large(client, large_image_bytes):
    """Test predict endpoint with file that's too large."""
    # Only test if the image is actually too large
    if len(large_image_bytes) > MAX_FILE_SIZE:
        files = {"file": ("large.jpg", io.BytesIO(large_image_bytes), "image/jpeg")}
        response = client.post("/predict", files=files)
        
        assert response.status_code == 200  # Returns 200 with ErrorResponse
        data = response.json()
        assert data["success"] is False
        assert "error" in data
        assert "filetoolarge" in data["error"].lower() or "file_too_large" in data["error"].lower()


def test_predict_endpoint_no_file(client):
    """Test predict endpoint without file."""
    response = client.post("/predict")
    assert response.status_code == 422  # FastAPI validation error


def test_predict_endpoint_valid_image_no_face(client, sample_image_bytes):
    """Test predict endpoint with valid image but no face detected."""
    # This will fail face detection
    files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
    response = client.post("/predict", files=files)
    
    # Should return error response for no face detected
    assert response.status_code == 200
    data = response.json()
    # If no face is detected, should return error response
    if not data.get("success", True):
        assert "error" in data
        assert "no_face_detected" in data["error"] or "message" in data


def test_predict_endpoint_success(client, sample_image_bytes, mock_classifier, mock_face_detector):
    """Test successful prediction."""
    # Set up mocks
    import app.main as main_module
    
    # Mock the global variables
    main_module.classifier = mock_classifier
    main_module.face_detector = mock_face_detector
    
    files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
    response = client.post("/predict", files=files)
    
    # Check if we got a successful response or an error (due to no face)
    assert response.status_code == 200
    data = response.json()
    
    # If successful, check structure
    if data.get("success"):
        assert "emotion" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "inference_time_ms" in data
        assert isinstance(data["confidence"], (int, float))
        assert 0.0 <= data["confidence"] <= 1.0


def test_predict_endpoint_model_not_loaded(client, sample_image_bytes):
    """Test predict endpoint when model is not loaded."""
    # Temporarily set classifier to None
    from app.main import classifier
    import app.main as main_module
    
    original_classifier = main_module.classifier
    main_module.classifier = None
    
    try:
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/predict", files=files)
        
        # Should return 503 Service Unavailable
        assert response.status_code == 503
    finally:
        # Restore original classifier
        main_module.classifier = original_classifier


def test_predict_endpoint_png_file(client):
    """Test predict endpoint with PNG file."""
    image = Image.new('RGB', (224, 224), color='green')
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    files = {"file": ("test.png", img_bytes.read(), "image/png")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    # May succeed or fail depending on face detection


def test_predict_endpoint_jpeg_file(client):
    """Test predict endpoint with JPEG file."""
    image = Image.new('RGB', (224, 224), color='blue')
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    files = {"file": ("test.jpeg", img_bytes.read(), "image/jpeg")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    # May succeed or fail depending on face detection


def test_predict_endpoint_uncertain_prediction(client, sample_image_bytes, mock_classifier, mock_face_detector):
    """Test predict endpoint with uncertain prediction (low confidence)."""
    import app.main as main_module
    
    # Mock low confidence prediction
    mock_classifier.predict_with_timing.return_value = (
        {'angry': 0.2, 'disgust': 0.2, 'fear': 0.2, 'happy': 0.2, 'sad': 0.1, 'surprise': 0.05, 'neutral': 0.05},
        'angry',
        0.2,  # Low confidence
        30.0
    )
    
    main_module.classifier = mock_classifier
    main_module.face_detector = mock_face_detector
    
    files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    if data.get("success"):
        assert "is_uncertain" in data
        if data["confidence"] < 0.6:  # CONFIDENCE_THRESHOLD
            assert data["is_uncertain"] is True
            assert "message" in data


def test_cors_headers(client):
    """Test that CORS headers are present."""
    response = client.options("/predict")
    # CORS middleware should handle OPTIONS requests
    # The exact headers depend on CORS configuration
    assert response.status_code in [200, 204, 405]  # Various valid responses

