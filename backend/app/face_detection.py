"""MediaPipe face detection and cropping."""

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from typing import Tuple

from app.config import (
    MEDIAPIPE_MODEL_SELECTION,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    FACE_CROP_PADDING
)
from app.exceptions import NoFaceDetectedError, MultipleFacesError


class FaceDetector:
    """MediaPipe-based face detector."""
    
    def __init__(self):
        """Initialize the MediaPipe face detector."""
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=MEDIAPIPE_MODEL_SELECTION,  # 1 = full range, 0 = short range
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE
        )
    
    def detect_and_crop(self, image: Image.Image, use_largest_face: bool = True) -> Image.Image:
        """
        Detect face in image and return cropped face.
        
        Args:
            image: PIL Image object (RGB)
            use_largest_face: If True, use largest face when multiple faces detected.
                            If False, raise MultipleFacesError.
        
        Returns:
            Cropped face image as PIL Image (RGB)
        
        Raises:
            NoFaceDetectedError: If no face is detected
            MultipleFacesError: If multiple faces detected and use_largest_face=False
        """
        # Convert PIL Image to numpy array (RGB)
        image_np = np.array(image)
        
        # MediaPipe expects RGB format
        image_rgb = image_np if len(image_np.shape) == 3 and image_np.shape[2] == 3 else cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.detector.process(image_rgb)
        
        if not results.detections:
            raise NoFaceDetectedError("No face detected in the image")
        
        # Handle multiple faces
        if len(results.detections) > 1:
            if not use_largest_face:
                raise MultipleFacesError("Multiple faces detected in the image")
            
            # Use the largest face (by bounding box area)
            detection = max(
                results.detections,
                key=lambda d: (
                    d.location_data.relative_bounding_box.width *
                    d.location_data.relative_bounding_box.height
                )
            )
        else:
            detection = results.detections[0]
        
        # Extract bounding box
        bbox = detection.location_data.relative_bounding_box
        h, w = image_rgb.shape[:2]
        
        # Calculate bounding box coordinates with padding
        x_min = bbox.xmin * w
        y_min = bbox.ymin * h
        box_width = bbox.width * w
        box_height = bbox.height * h
        
        # Add padding
        padding_x = box_width * FACE_CROP_PADDING
        padding_y = box_height * FACE_CROP_PADDING
        
        x = int(x_min - padding_x)
        y = int(y_min - padding_y)
        width = int(box_width + 2 * padding_x)
        height = int(box_height + 2 * padding_y)
        
        # Clamp to image boundaries
        x = max(0, x)
        y = max(0, y)
        width = min(width, w - x)
        height = min(height, h - y)
        
        # Ensure valid dimensions
        if width <= 0 or height <= 0:
            raise NoFaceDetectedError("Invalid face bounding box dimensions")
        
        # Crop face
        cropped = image_rgb[y:y+height, x:x+width]
        
        # Convert back to PIL Image
        cropped_image = Image.fromarray(cropped)
        
        return cropped_image
    
    def detect_face_count(self, image: Image.Image) -> int:
        """
        Count the number of faces in the image.
        
        Args:
            image: PIL Image object (RGB)
            
        Returns:
            Number of faces detected
        """
        # Convert PIL Image to numpy array (RGB)
        image_np = np.array(image)
        image_rgb = image_np if len(image_np.shape) == 3 and image_np.shape[2] == 3 else cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.detector.process(image_rgb)
        
        return len(results.detections) if results.detections else 0

