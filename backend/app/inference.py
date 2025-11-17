"""ONNX model inference for emotion classification."""

import time
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Tuple, Dict

from app.config import MODEL_PATH, EMOTION_CLASSES, NUM_CLASSES
from app.exceptions import ModelNotLoadedError


class EmotionClassifier:
    """ONNX-based emotion classifier."""
    
    def __init__(self, model_path: Path = MODEL_PATH):
        """
        Initialize the emotion classifier.
        
        Args:
            model_path: Path to the ONNX model file
        """
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        self._load_model()
    
    def _load_model(self):
        """Load the ONNX model and create inference session."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Configure session options for optimal performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        
        # Create ONNX Runtime session with CPU provider
        # The .data file (if present) will be automatically used for faster loading
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        
        # Get input and output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Verify input shape
        input_shape = self.session.get_inputs()[0].shape
        if input_shape[1] != 3 or input_shape[2] != 224 or input_shape[3] != 224:
            raise ValueError(
                f"Unexpected input shape: {input_shape}. Expected (batch, 3, 224, 224)"
            )
    
    def predict(self, image_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str, float]:
        """
        Predict emotion from preprocessed image array.
        
        Args:
            image_array: Preprocessed image array with shape (1, 3, 224, 224) and dtype float32
            
        Returns:
            Tuple of (probabilities, logits, predicted_emotion, confidence)
            - probabilities: numpy array of shape (NUM_CLASSES,) with probability for each class
            - logits: raw model output
            - predicted_emotion: string name of predicted emotion
            - confidence: confidence score (max probability)
        """
        if self.session is None:
            raise ModelNotLoadedError("Model session is not initialized")
        
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: image_array}
        )
        
        logits = outputs[0][0]  # Remove batch dimension: (1, 7) -> (7,)
        
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Get predicted class
        predicted_idx = np.argmax(probabilities)
        predicted_emotion = EMOTION_CLASSES[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        return probabilities, logits, predicted_emotion, confidence
    
    def predict_with_timing(self, image_array: np.ndarray) -> Tuple[Dict[str, float], str, float, float]:
        """
        Predict emotion with timing information.
        
        Args:
            image_array: Preprocessed image array with shape (1, 3, 224, 224) and dtype float32
            
        Returns:
            Tuple of (probabilities_dict, predicted_emotion, confidence, inference_time_ms)
            - probabilities_dict: dictionary mapping emotion names to probabilities
            - predicted_emotion: string name of predicted emotion
            - confidence: confidence score (max probability)
            - inference_time_ms: inference time in milliseconds
        """
        start_time = time.time()
        probabilities, _, predicted_emotion, confidence = self.predict(image_array)
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Convert probabilities array to dictionary
        probabilities_dict = {
            emotion: float(prob) 
            for emotion, prob in zip(EMOTION_CLASSES, probabilities)
        }
        
        return probabilities_dict, predicted_emotion, confidence, inference_time_ms
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.session is not None

