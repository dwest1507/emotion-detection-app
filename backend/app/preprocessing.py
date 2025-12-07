"""Image preprocessing pipeline for emotion detection."""

import numpy as np
from PIL import Image
from typing import Tuple

from app.config import INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess an image for emotion classification.
    
    Args:
        image: PIL Image object (RGB)
        
    Returns:
        Preprocessed image array with shape (1, 3, 224, 224) and dtype float32
    """
    # Resize to model input size
    image = image.resize(INPUT_SIZE, Image.Resampling.BILINEAR)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Convert from HWC to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Normalize with ImageNet statistics
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    img_array = (img_array - mean) / std
    
    # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def numpy_to_pil(image_array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.
    
    Args:
        image_array: numpy array in RGB format (H, W, C)
        
    Returns:
        PIL Image object
    """
    # Ensure values are in [0, 255] range
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    return Image.fromarray(image_array)

