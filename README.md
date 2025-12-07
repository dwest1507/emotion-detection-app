# 🎭 Emotion Detection Web App

> AI-powered facial emotion recognition using deep learning - A full-stack machine learning portfolio project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)

A production-ready web application that detects human emotions from facial images using a fine-tuned EfficientNet-B0 model. Built with modern full-stack technologies and deployed on Railway (backend) and Vercel (frontend).

## 🎯 Project Overview

This project demonstrates end-to-end machine learning engineering skills, from model training to production deployment. Users can upload facial images and receive real-time emotion predictions with confidence scores and probability distributions.

### Key Features

- 🧠 **Deep Learning Model**: Fine-tuned EfficientNet-B0 using transfer learning on FER-2013 dataset
- 🎨 **Modern Frontend**: Next.js 14 with TypeScript, TailwindCSS, and shadcn/ui
- ⚡ **Fast API**: FastAPI backend with ONNX Runtime for CPU-optimized inference
- 🔍 **Face Detection**: MediaPipe integration for robust face detection and cropping
- 🐳 **Dockerized**: Containerized backend for easy deployment
- 📊 **Real-time Predictions**: <500ms inference time with detailed probability breakdowns
- 🔒 **Privacy-First**: Zero data retention - images processed and immediately deleted
- 📱 **Responsive Design**: Mobile-friendly UI with smooth animations

## 🚀 Live Demo

- **Frontend**: [Coming Soon - Deploy to Vercel]
- **Backend API**: [Coming Soon - Deploy to Railway]
- **API Documentation**: [Swagger UI - Coming Soon]

## 📸 Screenshots

*Screenshots will be added after deployment*

## 🧠 Machine Learning Pipeline

### Dataset
- **FER-2013** (Facial Expression Recognition 2013)
- 35,887 grayscale images (48×48 pixels)
- 7 emotion classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Citation: Goodfellow et al. (2013)

### Model Architecture
- **Base Model**: EfficientNet-B0 (pre-trained on ImageNet)
- **Approach**: Transfer learning with fine-tuning
- **Input**: 224×224 RGB images
- **Output**: 7-class emotion classification
- **Framework**: PyTorch → ONNX export for production

### Training Details
- Transfer learning from ImageNet weights
- Two-phase training: frozen backbone → end-to-end fine-tuning
- Data augmentation: horizontal flips, rotations, color jitter
- Optimizer: AdamW with ReduceLROnPlateau scheduler
- Training time: ~1-2 hours on NVIDIA 3050 GPU

### Performance Metrics
- **Test Accuracy**: ~73% (target: 70-80%)
- **F1 Score (macro)**: [To be updated after training]
- **Inference Time**: <300ms on CPU
- **Model Size**: ~20MB

*Note: Metrics will be updated after model training*

## 🏗️ Architecture

### System Overview

```
User → Next.js Frontend → FastAPI Backend → MediaPipe → ONNX Model → Prediction
```

### Tech Stack

**Frontend:**
- Next.js 14 (App Router)
- TypeScript
- TailwindCSS
- shadcn/ui components
- Framer Motion (animations)
- Recharts (probability visualization)

**Backend:**
- FastAPI
- ONNX Runtime (CPU inference)
- MediaPipe (face detection)
- Python 3.11
- Docker

**Deployment:**
- Railway (backend)
- Vercel (frontend)
- Hugging Face Hub (model hosting)

### Data Flow

1. User uploads image via frontend
2. Image validated (type, size)
3. Face detection via MediaPipe
4. Face cropped and preprocessed (224×224, normalized)
5. ONNX model inference
6. Results returned with probabilities
7. Image immediately deleted (zero retention)

## 🛠️ Technical Highlights

- **Transfer Learning**: Leveraged pre-trained ImageNet weights for faster training and better accuracy
- **ONNX Optimization**: Model exported to ONNX for efficient CPU inference
- **Hugging Face Integration**: Model hosted on Hugging Face Hub for reliable deployment
- **Face Detection Pipeline**: Automatic face detection and cropping before emotion classification
- **Cold Start Handling**: Graceful handling of Railway's auto-sleep feature
- **Error Handling**: Comprehensive error states (no face, multiple faces, low confidence)
- **Privacy-First Design**: Zero data retention, GDPR-friendly
- **Production-Ready**: Docker containerization, health checks, logging, monitoring

## 📊 Results

*Results section will be updated after model training and evaluation*

- Test accuracy: [TBD]
- Confusion matrix: [See docs/TRAINING_REPORT.md]
- Per-class performance: [TBD]
- Sample predictions: [TBD]

## 🔧 Local Development

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (optional, for backend)
- CUDA-capable GPU (for model training, optional for inference)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

### Docker (Backend)

```bash
# Build image from repository root
docker build -t emotion-api .

# Run container
docker run -p 8000:8000 emotion-api

# Or with custom Hugging Face model repository
docker build --build-arg HF_MODEL_ID=your-username/your-model -t emotion-api .
```

## 📁 Project Structure

```
emotion-detection-app/
├── backend/              # FastAPI backend
│   ├── app/              # Application code
│   └── tests/            # Test suite
├── frontend/             # Next.js frontend
│   ├── app/              # Next.js app directory
│   ├── components/       # React components
│   └── lib/              # Utilities
├── models/               # ONNX model files (hosted on Hugging Face)
│   └── emotion_classifier.onnx
├── notebooks/            # Jupyter notebooks
│   ├── 01_eda.ipynb      # Exploratory data analysis
│   ├── 02_training.ipynb  # Model training
│   └── 03_onnx_export.ipynb
├── scripts/              # Utility scripts
│   └── upload_model_to_hf.sh  # Upload model to Hugging Face
├── docs/                 # Documentation
│   ├── MODEL_CARD.md
│   ├── ARCHITECTURE.md
│   ├── TRAINING_REPORT.md
│   └── API_DOCUMENTATION.md
├── Dockerfile            # Docker configuration (root)
├── railway.json          # Railway deployment config
└── README.md
```

## 🎓 Learning Outcomes

This project demonstrates:

- **ML Engineering**: End-to-end pipeline from data to deployment
- **Model Optimization**: ONNX export, CPU inference optimization
- **Full-Stack Development**: Backend API + frontend integration
- **Production Concerns**: Error handling, monitoring, budget management
- **DevOps**: Docker, CI/CD, cloud deployment
- **Communication**: Technical documentation, presenting complex projects

## ⚠️ Disclaimers

**Educational Purpose:**
This project is created for educational and portfolio demonstration purposes.

**Accuracy Limitations:**
- Model accuracy: ~73% on test set (target)
- Emotion recognition is subjective and culturally dependent
- Performance may vary significantly in real-world conditions

**Not for High-Stakes Use:**
This system should NOT be used for:
- Clinical or medical diagnosis
- Employment decisions
- Law enforcement or surveillance
- Academic testing or evaluation
- Any other high-stakes decision making

**Privacy:**
- Images are processed in real-time and immediately deleted
- No personal data is stored
- See Privacy Policy for details

**Bias Considerations:**
- Training data may not represent all demographics equally
- Model performance may vary across different populations
- Results should be interpreted with caution

## 📚 References

- **Dataset**: [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- **Original Paper**: Goodfellow, I. J., et al. (2013). Challenges in representation learning: A report on three machine learning contests. *Neural Networks*, 64, 59-63.
- **EfficientNet**: Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FER-2013 dataset creators
- PyTorch and torchvision teams
- FastAPI and Next.js communities
- All open-source contributors whose libraries made this possible

See [ATTRIBUTION.md](ATTRIBUTION.md) for full attributions.

## 📧 Contact

For questions or feedback, please open an issue on [GitHub](https://github.com/dwest1507/emotion-detection-app).

---

**Built with ❤️ for learning and portfolio demonstration**

