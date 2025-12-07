# Attributions

This document provides proper attribution to all datasets, pre-trained models, libraries, and resources used in this project.

## Dataset

### FER-2013 (Facial Expression Recognition 2013)

- **Source**: [Kaggle - FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- **Original Competition**: [Kaggle Challenges in Representation Learning](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
- **Citation**: 
  ```
  Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., 
  Hamner, B., ... & Bengio, Y. (2013). Challenges in representation learning: 
  A report on three machine learning contests. Neural Networks, 64, 59-63.
  ```
- **License**: Public domain / Research use (commercial use generally permitted)
- **Dataset Details**: 35,887 grayscale images (48×48 pixels), 7 emotion classes

## Pre-trained Models

### EfficientNet-B0

- **Source**: PyTorch torchvision.models
- **Pre-trained on**: ImageNet (ILSVRC 2012)
- **License**: Apache 2.0
- **Repository**: [PyTorch Vision](https://github.com/pytorch/vision)
- **Original Paper**: 
  ```
  Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for 
  convolutional neural networks. ICML.
  ```

### ImageNet Dataset

- **Dataset**: ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012
- **License**: Research and educational use
- **Note**: Pre-trained weights are used for transfer learning. The model is fine-tuned on FER-2013. Commercial use of transfer-learned models is generally permitted.

## Backend Libraries

| Library | Version | License | Purpose |
|---------|---------|---------|---------|
| [FastAPI](https://fastapi.tiangolo.com/) | 0.104.1 | MIT | Web framework |
| [Uvicorn](https://www.uvicorn.org/) | 0.24.0 | BSD | ASGI server |
| [ONNX Runtime](https://onnxruntime.ai/) | 1.16.3 | MIT | Model inference |
| [MediaPipe](https://mediapipe.dev/) | 0.10.8 | Apache 2.0 | Face detection |
| [Pillow](https://python-pillow.org/) | 10.1.0 | HPND | Image processing |
| [NumPy](https://numpy.org/) | 1.24.3 | BSD | Numerical computing |
| [PyTorch](https://pytorch.org/) | Latest | BSD 3-Clause | Model training |
| [Python-multipart](https://github.com/andrew-d/python-multipart) | 0.0.6 | Apache 2.0 | File uploads |
| [Pydantic](https://pydantic.dev/) | 2.5.0 | MIT | Data validation |
| [Pytest](https://pytest.org/) | 7.4.3 | MIT | Testing |
| [Pytest-asyncio](https://pytest-asyncio.readthedocs.io/) | 0.21.1 | Apache 2.0 | Async testing |
| [httpx](https://www.python-httpx.org/) | 0.25.2 | BSD | HTTP client |

## Frontend Libraries

| Library | Version | License | Purpose |
|---------|---------|---------|---------|
| [Next.js](https://nextjs.org/) | 14 | MIT | React framework |
| [React](https://react.dev/) | Latest | MIT | UI library |
| [TypeScript](https://www.typescriptlang.org/) | Latest | Apache 2.0 | Type safety |
| [TailwindCSS](https://tailwindcss.com/) | Latest | MIT | CSS framework |
| [shadcn/ui](https://ui.shadcn.com/) | Latest | MIT | UI components |
| [Framer Motion](https://www.framer.com/motion/) | Latest | MIT | Animations |
| [Recharts](https://recharts.org/) | Latest | MIT | Data visualization |

## Development Tools

| Tool | License | Purpose |
|------|---------|---------|
| [Docker](https://www.docker.com/) | Apache 2.0 | Containerization |
| [Git](https://git-scm.com/) | GPL-2.0 | Version control |
| [GitHub Actions](https://github.com/features/actions) | - | CI/CD |

## Deployment Platforms

| Platform | Purpose | License/Terms |
|----------|---------|---------------|
| [Railway](https://railway.app/) | Backend hosting | Free tier available |
| [Vercel](https://vercel.com/) | Frontend hosting | Free tier available |

## Additional Resources

- **ONNX Format**: [ONNX Specification](https://onnx.ai/) - Apache 2.0
- **CUDA**: NVIDIA CUDA Toolkit - Proprietary (for GPU training)

## License Summary

All major dependencies use permissive open-source licenses (MIT, Apache 2.0, BSD) that allow:
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Notes

- This project uses transfer learning with pre-trained ImageNet weights. The fine-tuned model is trained on FER-2013 dataset.
- All images processed by the application are immediately deleted after inference (zero data retention).
- This project is for educational and portfolio demonstration purposes.

---

**Last Updated**: 2025

