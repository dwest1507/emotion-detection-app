"""Pydantic models for request/response validation."""

from typing import Dict, Optional
from pydantic import BaseModel, Field, ConfigDict


class PredictionResponse(BaseModel):
    """Response model for successful predictions."""
    success: bool = True
    emotion: str = Field(..., description="Predicted emotion class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the prediction")
    is_uncertain: bool = Field(..., description="Whether the prediction confidence is below threshold")
    probabilities: Dict[str, float] = Field(..., description="Probability distribution over all emotion classes")
    inference_time_ms: float = Field(..., ge=0, description="Inference time in milliseconds")
    message: Optional[str] = Field(None, description="Optional message, e.g., uncertainty warning")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    success: bool = False
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")


class HealthResponse(BaseModel):
    """Response model for health check."""
    model_config = ConfigDict(protected_namespaces=())
    
    status: str = "ok"
    model_loaded: bool = Field(..., description="Whether the ONNX model is loaded")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_version: str
    model_architecture: str
    classes: list[str]
    input_size: list[int]
    trained_on: str

