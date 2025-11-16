"""Tests for ONNX inference."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from app.inference import EmotionClassifier
from app.exceptions import ModelNotLoadedError
from app.config import MODEL_PATH, EMOTION_CLASSES, NUM_CLASSES


@pytest.fixture
def mock_onnx_session():
    """Create a mock ONNX Runtime session."""
    mock_session = MagicMock()
    
    # Mock input/output names
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_input.shape = [1, 3, 224, 224]
    
    mock_output = MagicMock()
    mock_output.name = "output"
    
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.get_outputs.return_value = [mock_output]
    
    # Mock run method to return fake logits
    # Create fake logits (7 classes)
    fake_logits = np.array([[2.0, 0.5, 0.3, 0.1, 0.05, 0.02, 0.01]], dtype=np.float32)
    mock_session.run.return_value = [fake_logits]
    
    return mock_session


@pytest.fixture
def sample_preprocessed_image():
    """Create a sample preprocessed image array."""
    return np.random.randn(1, 3, 224, 224).astype(np.float32)


def test_emotion_classifier_initialization_with_mock(mock_onnx_session):
    """Test EmotionClassifier initialization with mocked ONNX session."""
    with patch('app.inference.ort.InferenceSession', return_value=mock_onnx_session):
        with patch('pathlib.Path.exists', return_value=True):
            classifier = EmotionClassifier()
            
            assert classifier.session is not None
            assert classifier.input_name == "input"
            assert classifier.output_name == "output"


def test_emotion_classifier_model_not_found():
    """Test that FileNotFoundError is raised when model file doesn't exist."""
    with patch('pathlib.Path.exists', return_value=False):
        with pytest.raises(FileNotFoundError):
            EmotionClassifier()


def test_emotion_classifier_predict_shape(mock_onnx_session, sample_preprocessed_image):
    """Test that predict returns correct shapes."""
    with patch('app.inference.ort.InferenceSession', return_value=mock_onnx_session):
        with patch('pathlib.Path.exists', return_value=True):
            classifier = EmotionClassifier()
            
            probabilities, logits, predicted_emotion, confidence = classifier.predict(sample_preprocessed_image)
            
            assert probabilities.shape == (NUM_CLASSES,), f"Expected shape ({NUM_CLASSES},), got {probabilities.shape}"
            assert logits.shape == (NUM_CLASSES,), f"Expected shape ({NUM_CLASSES},), got {logits.shape}"
            assert isinstance(predicted_emotion, str)
            assert predicted_emotion in EMOTION_CLASSES
            assert 0.0 <= confidence <= 1.0


def test_emotion_classifier_predict_probabilities_sum_to_one(mock_onnx_session, sample_preprocessed_image):
    """Test that probabilities sum to approximately 1.0."""
    with patch('app.inference.ort.InferenceSession', return_value=mock_onnx_session):
        with patch('pathlib.Path.exists', return_value=True):
            classifier = EmotionClassifier()
            
            probabilities, _, _, _ = classifier.predict(sample_preprocessed_image)
            
            prob_sum = np.sum(probabilities)
            assert np.isclose(prob_sum, 1.0, atol=1e-5), f"Probabilities should sum to 1.0, got {prob_sum}"


def test_emotion_classifier_predict_confidence_matches_max_probability(mock_onnx_session, sample_preprocessed_image):
    """Test that confidence equals the maximum probability."""
    with patch('app.inference.ort.InferenceSession', return_value=mock_onnx_session):
        with patch('pathlib.Path.exists', return_value=True):
            classifier = EmotionClassifier()
            
            probabilities, _, predicted_emotion, confidence = classifier.predict(sample_preprocessed_image)
            
            max_prob = np.max(probabilities)
            assert np.isclose(confidence, max_prob, atol=1e-5), \
                f"Confidence should equal max probability, got {confidence} vs {max_prob}"


def test_emotion_classifier_predict_with_timing(mock_onnx_session, sample_preprocessed_image):
    """Test predict_with_timing returns timing information."""
    with patch('app.inference.ort.InferenceSession', return_value=mock_onnx_session):
        with patch('pathlib.Path.exists', return_value=True):
            classifier = EmotionClassifier()
            
            probabilities_dict, predicted_emotion, confidence, inference_time_ms = \
                classifier.predict_with_timing(sample_preprocessed_image)
            
            assert isinstance(probabilities_dict, dict)
            assert len(probabilities_dict) == NUM_CLASSES
            assert all(emotion in EMOTION_CLASSES for emotion in probabilities_dict.keys())
            assert all(0.0 <= prob <= 1.0 for prob in probabilities_dict.values())
            assert isinstance(predicted_emotion, str)
            assert predicted_emotion in EMOTION_CLASSES
            assert 0.0 <= confidence <= 1.0
            assert inference_time_ms >= 0.0


def test_emotion_classifier_predict_model_not_loaded(sample_preprocessed_image):
    """Test that ModelNotLoadedError is raised when model is not loaded."""
    classifier = EmotionClassifier.__new__(EmotionClassifier)
    classifier.session = None
    
    with pytest.raises(ModelNotLoadedError):
        classifier.predict(sample_preprocessed_image)


def test_emotion_classifier_is_loaded(mock_onnx_session):
    """Test is_loaded method."""
    with patch('app.inference.ort.InferenceSession', return_value=mock_onnx_session):
        with patch('pathlib.Path.exists', return_value=True):
            classifier = EmotionClassifier()
            
            assert classifier.is_loaded() is True
            
            classifier.session = None
            assert classifier.is_loaded() is False


def test_emotion_classifier_predict_probabilities_dict_structure(mock_onnx_session, sample_preprocessed_image):
    """Test that probabilities_dict has correct structure."""
    with patch('app.inference.ort.InferenceSession', return_value=mock_onnx_session):
        with patch('pathlib.Path.exists', return_value=True):
            classifier = EmotionClassifier()
            
            probabilities_dict, _, _, _ = classifier.predict_with_timing(sample_preprocessed_image)
            
            # Check all emotion classes are present
            for emotion in EMOTION_CLASSES:
                assert emotion in probabilities_dict, f"Missing emotion class: {emotion}"
            
            # Check values are floats
            for prob in probabilities_dict.values():
                assert isinstance(prob, float)


def test_emotion_classifier_invalid_input_shape(mock_onnx_session):
    """Test that invalid input shape is detected."""
    # Create a mock session with wrong input shape
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_input.shape = [1, 3, 128, 128]  # Wrong size
    
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.get_outputs.return_value = [MagicMock()]
    
    with patch('app.inference.ort.InferenceSession', return_value=mock_session):
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(ValueError, match="Unexpected input shape"):
                EmotionClassifier()

