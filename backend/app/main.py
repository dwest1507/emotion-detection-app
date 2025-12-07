"""FastAPI application for emotion detection."""

import io
import uuid
import logging
import time
from pathlib import Path
from typing import Union, Callable
from contextvars import ContextVar

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from PIL import Image

# Request ID context variable for tracking requests (defined early for use in logging)
request_id_var: ContextVar[str] = ContextVar('request_id', default='')


class RequestIDFilter(logging.Filter):
    """Logging filter to add request ID to log records."""
    
    def filter(self, record):
        # Always set request_id, defaulting to empty string if not set
        record.request_id = request_id_var.get('') if request_id_var else ''
        return True


class SafeRequestIDFormatter(logging.Formatter):
    """Custom formatter that safely handles missing request_id."""
    
    def format(self, record):
        # Ensure request_id exists on the record
        if not hasattr(record, 'request_id'):
            record.request_id = request_id_var.get('') if request_id_var else ''
        return super().format(record)


# Configure logging with safe request_id handling
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Apply the filter and formatter to root logger to catch all log messages
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.addFilter(RequestIDFilter())
    handler.setFormatter(SafeRequestIDFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

# Create logger
logger = logging.getLogger(__name__)

from app.config import (
    MAX_FILE_SIZE,
    ALLOWED_EXTENSIONS,
    CONFIDENCE_THRESHOLD,
    EMOTION_CLASSES,
    INPUT_SIZE,
    MODEL_VERSION,
    MODEL_ARCHITECTURE,
    MODEL_DATASET
)
from app.exceptions import (
    NoFaceDetectedError,
    MultipleFacesError,
    InvalidFileTypeError,
    FileTooLargeError,
    ModelNotLoadedError
)
from app.models import (
    PredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse
)
from app.inference import EmotionClassifier
from app.face_detection import FaceDetector
from app.preprocessing import preprocess_image

# Initialize FastAPI app
app = FastAPI(
    title="Emotion Detection API",
    description="API for detecting emotions in facial images using deep learning",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next: Callable):
    """Add request ID to each request for tracking and log request/response."""
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    
    # Log request
    start_time = time.time()
    logger.info(
        f"Request started: {request.method} {request.url.path} - "
        f"Client: {request.client.host if request.client else 'unknown'}"
    )
    
    try:
        response = await call_next(request)
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Request completed: {request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Request failed: {request.method} {request.url.path} - "
            f"Error: {str(e)} - "
            f"Time: {process_time:.3f}s",
            exc_info=True
        )
        raise


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with request ID."""
    request_id = request_id_var.get('')
    logger.warning(
        f"HTTP exception: {exc.status_code} - {exc.detail} - "
        f"Path: {request.url.path}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": "http_error",
            "message": exc.detail,
            "request_id": request_id,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    request_id = request_id_var.get('')
    logger.warning(
        f"Validation error: {exc.errors()} - "
        f"Path: {request.url.path}"
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "validation_error",
            "message": "Invalid request data. Please check your input.",
            "details": exc.errors(),
            "request_id": request_id
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    request_id = request_id_var.get('')
    logger.error(
        f"Unhandled exception: {type(exc).__name__} - {str(exc)} - "
        f"Path: {request.url.path}",
        exc_info=True
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again later.",
            "request_id": request_id
        }
    )

# Global model and detector instances (loaded at startup)
classifier: Union[EmotionClassifier, None] = None
face_detector: Union[FaceDetector, None] = None


@app.on_event("startup")
async def startup_event():
    """Load model and initialize face detector at startup."""
    global classifier, face_detector
    logger.info("Starting up application...")
    try:
        logger.info("Loading emotion classifier model...")
        classifier = EmotionClassifier()
        logger.info("Model loaded successfully")
        
        logger.info("Initializing face detector...")
        face_detector = FaceDetector()
        logger.info("Face detector initialized successfully")
        
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global classifier, face_detector
    logger.info("Shutting down application...")
    classifier = None
    face_detector = None
    logger.info("Application shutdown complete")


def validate_file(file: UploadFile) -> None:
    """
    Validate uploaded file.
    
    Raises:
        InvalidFileTypeError: If file type is not allowed
        FileTooLargeError: If file size exceeds limit
    """
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise InvalidFileTypeError(
            f"File type '{file_ext}' not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Note: File size check should be done by reading the file
    # FastAPI doesn't provide file size before reading


@app.post("/predict", response_model=Union[PredictionResponse, ErrorResponse])
async def predict_emotion(file: UploadFile = File(...)):
    """
    Predict emotion from uploaded facial image.
    
    Args:
        file: Image file (JPEG or PNG)
        
    Returns:
        PredictionResponse with emotion, confidence, and probabilities
        or ErrorResponse if an error occurs
    """
    global classifier, face_detector
    request_id = request_id_var.get('')
    
    if classifier is None or face_detector is None:
        logger.error("Prediction attempted but model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        logger.info(f"Processing prediction request - File: {file.filename}, Size: {file.size if hasattr(file, 'size') else 'unknown'}")
        
        # Validate file
        validate_file(file)
        
        # Read file content
        contents = await file.read()
        file_size_mb = len(contents) / 1024 / 1024
        logger.debug(f"File read successfully - Size: {file_size_mb:.2f} MB")
        
        # Check file size
        if len(contents) > MAX_FILE_SIZE:
            logger.warning(f"File too large: {file_size_mb:.2f} MB (max: {MAX_FILE_SIZE / 1024 / 1024} MB)")
            raise FileTooLargeError(
                f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size "
                f"({MAX_FILE_SIZE / 1024 / 1024} MB)"
            )
        
        # Load image with PIL
        try:
            image = Image.open(io.BytesIO(contents))
            # Convert to RGB if necessary (handles RGBA, L, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            logger.debug(f"Image loaded - Mode: {image.mode}, Size: {image.size}")
        except Exception as e:
            logger.warning(f"Invalid image file: {str(e)}")
            raise InvalidFileTypeError(f"Invalid image file: {str(e)}")
        
        # Face detection and cropping
        face_detection_start = time.time()
        try:
            cropped_face = face_detector.detect_and_crop(image, use_largest_face=True)
            face_detection_time = (time.time() - face_detection_start) * 1000
            logger.info(f"Face detected and cropped - Time: {face_detection_time:.2f}ms")
        except NoFaceDetectedError:
            logger.warning("No face detected in image")
            return ErrorResponse(
                success=False,
                error="no_face_detected",
                message="No face found in the image. Please upload a clear photo with a visible face."
            )
        except MultipleFacesError:
            logger.warning("Multiple faces detected in image")
            return ErrorResponse(
                success=False,
                error="multiple_faces",
                message="Multiple faces detected. Please upload a photo with a single person."
            )
        
        # Preprocess image
        preprocessed = preprocess_image(cropped_face)
        
        # Run inference
        try:
            probabilities_dict, predicted_emotion, confidence, inference_time_ms = \
                classifier.predict_with_timing(preprocessed)
            logger.info(
                f"Inference completed - Emotion: {predicted_emotion}, "
                f"Confidence: {confidence:.3f}, "
                f"Time: {inference_time_ms:.2f}ms"
            )
        except Exception as e:
            logger.error(f"Inference error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference error: {str(e)}"
            )
        
        # Check if uncertain
        is_uncertain = confidence < CONFIDENCE_THRESHOLD
        message = None
        if is_uncertain:
            message = "The model is uncertain about this prediction."
            logger.warning(f"Low confidence prediction: {confidence:.3f} < {CONFIDENCE_THRESHOLD}")
        
        # Return response
        logger.info(f"Prediction successful - Emotion: {predicted_emotion}, Confidence: {confidence:.3f}")
        return PredictionResponse(
            success=True,
            emotion=predicted_emotion,
            confidence=confidence,
            is_uncertain=is_uncertain,
            probabilities=probabilities_dict,
            inference_time_ms=inference_time_ms,
            message=message
        )
    
    except (InvalidFileTypeError, FileTooLargeError) as e:
        request_id = request_id_var.get('')
        logger.warning(f"Validation error: {type(e).__name__} - {str(e)}")
        return ErrorResponse(
            success=False,
            error=type(e).__name__.lower().replace("error", ""),
            message=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        request_id = request_id_var.get('')
        logger.error(f"Unexpected error in predict endpoint: {type(e).__name__} - {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error. Request ID: {request_id}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with status and model_loaded flag
    """
    global classifier
    return HealthResponse(
        status="ok",
        model_loaded=classifier is not None and classifier.is_loaded()
    )


@app.get("/info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get model information.
    
    Returns:
        ModelInfoResponse with model metadata
    """
    return ModelInfoResponse(
        model_version=MODEL_VERSION,
        model_architecture=MODEL_ARCHITECTURE,
        classes=EMOTION_CLASSES,
        input_size=list(INPUT_SIZE),
        trained_on=MODEL_DATASET
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Emotion Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "info": "/info",
            "docs": "/docs"
        }
    }

