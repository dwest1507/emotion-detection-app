# Emotion Classifier Model Card

## Model Details
- **Architecture**: EfficientNet-B0 (transfer learning from ImageNet)
- **Input**: 224×224 RGB images
- **Output**: 7 classes (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Framework**: PyTorch → ONNX
- **Parameters**: ~4M (Fine-tuned)
- **Model Size**: ~20MB (ONNX Quantized)
- **License**: MIT
- **Date**: 2025-12-13

## Training Data
- **Dataset**: FER-2013 (Facial Expression Recognition 2013)
- **Source**: Kaggle / ICML 2013 Workshop
- **Size**: 35,887 images (48x48 pixel grayscale, converted to RGB)
- **Split**:
  - Training: 28,709 images
  - Public Test: 3,589 images
  - Private Test: 3,589 images
- **Class Distribution**:
  - Happy: 8989
  - Neutral: 6198
  - Sad: 6077
  - Fear: 5121
  - Angry: 4953
  - Surprise: 4002
  - Disgust: 547

## Performance
> [!NOTE]
> Metrics below are targets/estimates based on partial training logs.

- **Test Accuracy**: ~73% (Target)
- **F1 Score (Macro)**: TBD
- **Inference Time (CPU)**: <300ms (ONNX Runtime)

## Limitations
- **Lighting**: Performance degrades in low-light or extreme lighting conditions.
- **Pose**: Best results with frontal face views; profile views may have lower accuracy.
- **Demographics**: Training data (FER-2013) has known biases; may not generalize equally across all demographics.
- **Context**: Recognizes facial expressions, which may not always align with internal emotional state.

## Intended Use
- **Primary Use**: Educational portfolio project demonstrating ML engineering pipeline.
- **Secondary Use**: Experimental UI/UX for emotion-based interaction.
- **Out of Scope**: Medical diagnosis, surveillance, high-stakes decision making.

## Ethical Considerations
- **Privacy**: No images are stored. Processing is done in memory and data is discarded immediately.
- **Bias**: Users should be aware of potential biases in the FER-2013 dataset. 
