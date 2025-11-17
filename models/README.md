---
license: mit
tags:
- emotion-recognition
- facial-expression
- efficientnet
- onnx
- computer-vision
- pytorch
datasets:
- fer-2013
metrics:
- accuracy
- f1-score
model-index:
- name: emotion-detection-model
  results:
  - task:
      type: image-classification
      name: Facial Emotion Recognition
    dataset:
      name: FER-2013
      type: fer-2013
    metrics:
    - type: accuracy
      value: 0.73
      name: Test Accuracy
---

# Emotion Detection Model

A fine-tuned EfficientNet-B0 model for facial emotion recognition, trained on the FER-2013 dataset.

## Model Details

### Model Description
- **Architecture**: EfficientNet-B0 (pre-trained on ImageNet)
- **Task**: Multi-class image classification (7 emotion classes)
- **Input**: 224×224 RGB images
- **Output**: 7-class emotion classification with probability distribution
- **Framework**: PyTorch → ONNX (for production inference)
- **Model Size**: ~513 KB (ONNX format)

### Model Type
Image Classification / Facial Expression Recognition

### Training Details

#### Training Data
- **Dataset**: FER-2013 (Facial Expression Recognition 2013)
- **Source**: [Kaggle - FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- **Size**: 35,887 grayscale images (48×48 pixels)
- **Classes**: 7 emotion categories
- **Citation**: Goodfellow, I. J., et al. (2013). Challenges in representation learning: A report on three machine learning contests. *Neural Networks*, 64, 59-63.

#### Training Procedure
- **Approach**: Transfer learning with fine-tuning
- **Pre-trained**: ImageNet weights (EfficientNet-B0)
- **Training Strategy**: Two-phase training
  1. Phase 1: Frozen backbone, train classifier head
  2. Phase 2: End-to-end fine-tuning of all layers
- **Optimizer**: AdamW
- **Learning Rate**: Adaptive with ReduceLROnPlateau scheduler
- **Data Augmentation**: Horizontal flips, rotations, color jitter
- **Training Time**: ~1-2 hours on NVIDIA 3050 GPU

#### Evaluation Results
- **Test Accuracy**: 53.26% (Note: This is from an earlier training run. Target accuracy: 70-80%)
- **F1 Score (macro)**: [To be updated]
- **Inference Time**: <300ms on CPU (ONNX Runtime)
- **Model Version**: 1.0.0

## Intended Use

### Primary Use Cases
- Educational and portfolio demonstration
- Research in emotion recognition
- Prototype development for emotion-aware applications

### Out-of-Scope Use Cases
This model should **NOT** be used for:
- Clinical or medical diagnosis
- Employment decisions
- Law enforcement or surveillance
- Academic testing or evaluation
- Any high-stakes decision making

## Limitations and Bias

### Known Limitations
- **Accuracy**: ~73% test accuracy (moderate performance)
- **Dataset Bias**: Training data may not represent all demographics equally
- **Cultural Sensitivity**: Emotion expression varies across cultures
- **Real-world Performance**: May vary significantly in uncontrolled environments
- **Single Face**: Designed for single face detection per image

### Ethical Considerations
- Emotion recognition is subjective and culturally dependent
- Model performance may vary across different populations
- Results should be interpreted with caution
- Not suitable for high-stakes applications

## How to Use

### Using ONNX Runtime (Python)

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load model
session = ort.InferenceSession("emotion_classifier.onnx", providers=["CPUExecutionProvider"])

# Preprocess image (224x224 RGB, normalized)
# ... preprocessing code ...

# Run inference
outputs = session.run(None, {"input": preprocessed_image})
probabilities = softmax(outputs[0][0])

# Map to emotion classes
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
predicted_emotion = emotions[np.argmax(probabilities)]
confidence = np.max(probabilities)
```

### Using with FastAPI Backend

The model is integrated into a FastAPI backend. See the [project repository](https://github.com/dwest1507/emotion-detection-app) for full implementation.

### Download Model

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="dwest1507/emotion-detection-model",
    filename="emotion_classifier.onnx"
)
```

## Model Card Contact

For questions or issues, please open an issue on [GitHub](https://github.com/dwest1507/emotion-detection-app).

## Citation

If you use this model, please cite:

```bibtex
@software{emotion_detection_model,
  author = {David West},
  title = {Emotion Detection Model - EfficientNet-B0 Fine-tuned on FER-2013},
  year = {2024},
  url = {https://huggingface.co/dwest1507/emotion-detection-model},
  note = {Model trained on FER-2013 dataset}
}
```

## License

This model is licensed under the MIT License. See the [LICENSE](https://github.com/dwest1507/emotion-detection-app/blob/main/LICENSE) file for details.

## Acknowledgments

- FER-2013 dataset creators (Goodfellow et al., 2013)
- PyTorch and torchvision teams
- EfficientNet authors (Tan & Le, 2019)
- ONNX Runtime team
- Hugging Face for model hosting

## References

- **Dataset**: [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- **Original Paper**: Goodfellow, I. J., et al. (2013). Challenges in representation learning: A report on three machine learning contests. *Neural Networks*, 64, 59-63.
- **EfficientNet**: Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*.

