# API Documentation

## Base URL
- **Local**: `http://localhost:8000`
- **Production**: `https://[your-railway-app].up.railway.app`

## Endpoints

### 1. Predict Emotion
Upload an image to get an emotion prediction.

- **URL**: `/predict`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`

#### Parameters
| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | File | Yes | Image file (JPEG/PNG). Max 5MB. |

#### Success Response (200 OK)
```json
{
  "success": true,
  "emotion": "happy",
  "confidence": 0.95,
  "is_uncertain": false,
  "probabilities": {
    "angry": 0.01,
    "disgust": 0.00,
    "fear": 0.00,
    "happy": 0.95,
    "sad": 0.02,
    "surprise": 0.01,
    "neutral": 0.01
  },
  "inference_time_ms": 125.5
}
```

#### Error Response (400 Bad Request / 422 Unprocessable Entity)
```json
{
  "success": false,
  "error": "no_face_detected",
  "message": "No face found in the image. Please upload a clear photo with a visible face."
}
```

---

### 2. Health Check
Check operational status of the API.

- **URL**: `/health`
- **Method**: `GET`

#### Success Response (200 OK)
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### 3. Model Info
Get metadata about the currently loaded model.

- **URL**: `/info`
- **Method**: `GET`

#### Success Response (200 OK)
```json
{
  "model_version": "1.0.0",
  "model_architecture": "EfficientNet-B0",
  "classes": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
  "input_size": [224, 224],
  "trained_on": "FER-2013"
}
```
