# Training Report

## Experiment Overview
This report documents the training process for the emotion classification model.

- **Model**: EfficientNet-B0 (Transfer Learning)
- **Dataset**: FER-2013
- **Objective**: 7-class classification (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)

## Exploratory Data Analysis (EDA)

Key findings from initial data analysis:
- **Class Imbalance**: 'Disgust' class is significantly underrepresented (~547 samples) compared to 'Happy' (~9000 samples).
- **Resolution**: Images are low resolution (48x48), requiring upscaling to 224x224 for EfficientNet.
- **Grayscale**: Source data is single-channel grayscale, necessitating conversion to 3-channel RGB for pre-trained weights.

## Training Configuration

### Hyperparameters
- **Batch Size**: 32
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Weight Decay**: 1e-4
- **Loss Function**: CrossEntropyLoss (with Class Weights)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=2)
- **Mixed Precision**: Enabled (fp16)

### Training Strategy
1. **Phase 1 (Frozen Backbone)**:
   - Train only the custom classifier head.
   - 3 Epochs.
   - Purpose: Stabilize random weights in the new head before modifying the pretrained backbone.
   
2. **Phase 2 (Fine-tuning)**:
   - Unfreeze all layers.
   - Train end-to-end.
   - Purpose: Adapt feature extraction features to specific domain (facial expressions).

## Preliminary Results (Phase 1)

Early training logs (Epoch 1-3) show:
- **Validation Accuracy**: ~35% (Initial frozen state).
- **Loss**: Decreasing steadily.
- **Note**: Significant improvement is expected in Phase 2 (Fine-tuning) as the backbone adapts. Accuracy targets for this architecture are typically 65-73% on FER-2013.

## Confusion Matrix & Analysis
*To be generated after full training.*

### Observations
- Initial confusion likely high between subtle expressions (e.g., Sad vs Neutral).
- 'Disgust' class performance likely lower due to sample scarcity, partially mitigated by class weighting.

## Conclusion
The transfer learning approach provides a strong foundation. Using EfficientNet-B0 strikes a balance between performance (good feature extraction) and efficiency (fast CPU inference for deployment).
