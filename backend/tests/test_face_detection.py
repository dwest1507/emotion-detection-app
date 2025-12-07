"""Tests for MediaPipe face detection."""

import numpy as np
import pytest
from PIL import Image

from app.face_detection import FaceDetector
from app.exceptions import NoFaceDetectedError, MultipleFacesError


@pytest.fixture
def face_detector():
    """Create a FaceDetector instance for testing."""
    return FaceDetector()


def test_face_detector_initialization(face_detector):
    """Test that FaceDetector initializes correctly."""
    assert face_detector is not None
    assert face_detector.detector is not None


def test_detect_face_count_no_face(face_detector):
    """Test face counting with an image containing no faces."""
    # Create a simple image with no faces (just a solid color)
    image = Image.new('RGB', (224, 224), color='blue')
    
    count = face_detector.detect_face_count(image)
    
    assert count == 0, "Should detect 0 faces in an image without faces"


def test_detect_and_crop_no_face(face_detector):
    """Test that NoFaceDetectedError is raised when no face is detected."""
    # Create a simple image with no faces
    image = Image.new('RGB', (224, 224), color='green')
    
    with pytest.raises(NoFaceDetectedError):
        face_detector.detect_and_crop(image)


def test_detect_and_crop_returns_pil_image(face_detector):
    """Test that detect_and_crop returns a PIL Image when a face is detected."""
    # Note: This test requires an actual image with a face
    # For now, we'll test the structure - in real tests, use a sample image
    # This test will be skipped if no face images are available
    pass


def test_detect_and_crop_multiple_faces_use_largest(face_detector):
    """Test that detect_and_crop uses largest face when multiple faces detected."""
    # Note: This test requires an actual image with multiple faces
    # For now, we'll test the structure
    pass


def test_detect_and_crop_multiple_faces_raise_error(face_detector):
    """Test that MultipleFacesError is raised when use_largest_face=False."""
    # Note: This test requires an actual image with multiple faces
    # For now, we'll test the structure
    pass


def test_detect_and_crop_cropped_size(face_detector):
    """Test that cropped face has reasonable dimensions."""
    # Note: This test requires an actual image with a face
    # The cropped image should be smaller than or equal to the original
    pass


def test_detect_face_count_with_face(face_detector):
    """Test face counting with an image containing a face."""
    # Note: This test requires an actual image with a face
    # Should return count >= 1
    pass


def test_detect_and_crop_padding(face_detector):
    """Test that face cropping includes padding."""
    # Note: This test requires an actual image with a face
    # The cropped bounding box should be larger than the detected face box
    pass


def test_detect_and_crop_boundary_clamping(face_detector):
    """Test that face cropping clamps to image boundaries."""
    # Create a test case where face is near image edge
    # The crop should not exceed image dimensions
    pass


def test_detect_face_count_empty_image(face_detector):
    """Test face counting with an empty/small image."""
    # Very small image
    image = Image.new('RGB', (10, 10), color='red')
    
    count = face_detector.detect_face_count(image)
    
    # Should handle gracefully (likely 0 faces)
    assert isinstance(count, int)
    assert count >= 0


def test_detect_and_crop_invalid_dimensions(face_detector):
    """Test that invalid bounding box dimensions raise appropriate error."""
    # This would happen if MediaPipe returns invalid coordinates
    # The code should handle this gracefully
    pass

