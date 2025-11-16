"""Custom exceptions for the emotion detection API."""


class EmotionDetectionError(Exception):
    """Base exception for emotion detection errors."""
    pass


class NoFaceDetectedError(EmotionDetectionError):
    """Raised when no face is detected in the image."""
    pass


class MultipleFacesError(EmotionDetectionError):
    """Raised when multiple faces are detected in the image."""
    pass


class InvalidFileTypeError(EmotionDetectionError):
    """Raised when the uploaded file type is not supported."""
    pass


class FileTooLargeError(EmotionDetectionError):
    """Raised when the uploaded file exceeds the size limit."""
    pass


class ModelNotLoadedError(EmotionDetectionError):
    """Raised when the model is not loaded."""
    pass

