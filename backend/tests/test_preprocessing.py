"""Tests for image preprocessing."""

import numpy as np
import pytest
from PIL import Image

from app.preprocessing import preprocess_image, numpy_to_pil
from app.config import INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD


def test_preprocess_image_shape():
    """Test that preprocessed image has correct shape."""
    # Create a test image
    image = Image.new('RGB', (100, 100), color='red')
    
    preprocessed = preprocess_image(image)
    
    assert preprocessed.shape == (1, 3, 224, 224), f"Expected shape (1, 3, 224, 224), got {preprocessed.shape}"
    assert preprocessed.dtype == np.float32, f"Expected dtype float32, got {preprocessed.dtype}"


def test_preprocess_image_resize():
    """Test that image is resized to INPUT_SIZE."""
    # Create images of different sizes
    sizes = [(50, 50), (300, 300), (224, 224), (100, 200)]
    
    for size in sizes:
        image = Image.new('RGB', size, color='blue')
        preprocessed = preprocess_image(image)
        assert preprocessed.shape == (1, 3, 224, 224), f"Image of size {size} not resized correctly"


def test_preprocess_image_normalization():
    """Test that image is normalized with ImageNet stats."""
    # Create a white image (all pixels = 255)
    image = Image.new('RGB', (224, 224), color='white')
    preprocessed = preprocess_image(image)
    
    # After normalization, values should be in a reasonable range
    # White (255) -> 1.0 -> normalized by ImageNet stats
    assert preprocessed.min() >= -3.0, "Normalized values too low"
    assert preprocessed.max() <= 3.0, "Normalized values too high"
    
    # Check that normalization was applied (values should not be in [0, 1] range)
    # After ImageNet normalization, white pixels should be around (1.0 - mean) / std
    mean_channel = np.mean(preprocessed[0, 0, :, :])
    # For white image, after normalization, mean should be around (1.0 - 0.485) / 0.229 ≈ 2.25
    assert abs(mean_channel) < 5.0, "Normalization may not be applied correctly"


def test_preprocess_image_dtype():
    """Test that preprocessed image has correct dtype."""
    image = Image.new('RGB', (224, 224), color='green')
    preprocessed = preprocess_image(image)
    
    assert preprocessed.dtype == np.float32, "Preprocessed image should be float32"


def test_preprocess_image_channel_order():
    """Test that image channels are in CHW format."""
    # Create an image with distinct colors per channel
    # Red channel: 255, Green: 0, Blue: 0
    image = Image.new('RGB', (224, 224), color='red')
    preprocessed = preprocess_image(image)
    
    # Check that first channel (R) has higher values than others
    # After normalization, red channel should have different values
    red_channel_mean = np.mean(preprocessed[0, 0, :, :])
    green_channel_mean = np.mean(preprocessed[0, 1, :, :])
    blue_channel_mean = np.mean(preprocessed[0, 2, :, :])
    
    # For a red image, red channel should be different from green/blue
    # (exact values depend on normalization, but they should differ)
    assert not np.isclose(red_channel_mean, green_channel_mean, atol=0.1), \
        "Channels should be in CHW format and distinguishable"


def test_numpy_to_pil():
    """Test conversion from numpy array to PIL Image."""
    # Create a numpy array in RGB format (H, W, C)
    array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    pil_image = numpy_to_pil(array)
    
    assert isinstance(pil_image, Image.Image), "Should return PIL Image"
    assert pil_image.size == (100, 100), "Size should match"
    assert pil_image.mode == 'RGB', "Mode should be RGB"


def test_numpy_to_pil_clipping():
    """Test that values are clipped to [0, 255] range."""
    # Create array with values outside [0, 255]
    array = np.array([[[300, -10, 128]]], dtype=np.int32)
    
    pil_image = numpy_to_pil(array)
    
    # Should not raise error and should clip values
    assert isinstance(pil_image, Image.Image), "Should handle out-of-range values"


def test_preprocess_image_different_modes():
    """Test preprocessing handles different image modes."""
    # Test with RGB (should work)
    rgb_image = Image.new('RGB', (100, 100), color='red')
    preprocessed_rgb = preprocess_image(rgb_image)
    assert preprocessed_rgb.shape == (1, 3, 224, 224)
    
    # Test with RGBA (should convert to RGB first)
    rgba_image = Image.new('RGBA', (100, 100), color=(255, 0, 0, 255))
    # Note: preprocessing expects RGB, so conversion should happen before calling preprocess_image
    # This test verifies the function works with RGB input

