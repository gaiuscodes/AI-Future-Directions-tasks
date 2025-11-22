# Edge AI Prototype: Recyclable Item Classification

A lightweight image classification system for recognizing recyclable items (paper, plastic, glass, metal) optimized for edge devices like Raspberry Pi.

## 🎯 Overview

This project demonstrates how to:
- Train a lightweight CNN model for image classification
- Convert models to TensorFlow Lite for edge deployment
- Achieve real-time inference on resource-constrained devices
- Understand the benefits of Edge AI for real-time applications

## 📋 Features

- **Lightweight Model**: ~500K parameters, optimized for edge devices
- **TensorFlow Lite**: Converted models with 70-80% size reduction
- **Real-time Inference**: 5-50 ms per image on edge devices
- **Quantization**: 8-bit quantization for maximum efficiency
- **Comprehensive Testing**: Accuracy metrics and performance benchmarks

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.13+ (for training)
- NumPy, Matplotlib, scikit-learn
- For edge devices: TensorFlow Lite runtime

### Installation

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

For Raspberry Pi (use TensorFlow Lite runtime instead):
```bash
pip install tflite-runtime numpy pillow
```

### Usage

#### 1. Train the Model

```bash
python train_model.py
```

This will:
- Create a lightweight CNN model
- Train on your dataset (or generate synthetic data for demo)
- Save the model to `models/recyclable_classifier.h5`
- Generate training curves and metrics

**Note**: For production, replace synthetic data with real images organized in:
```
data/
  ├── paper/
  ├── plastic/
  ├── glass/
  └── metal/
```

#### 2. Convert to TensorFlow Lite

```bash
python convert_to_tflite.py
```

This creates:
- `models/recyclable_classifier.tflite` (float32)
- `models/recyclable_classifier_quantized.tflite` (quantized, recommended)

#### 3. Test the Models

```bash
python test_tflite.py
```

This will:
- Compare Keras vs TFLite models
- Benchmark inference speed
- Validate accuracy and predictions
- Generate performance reports

#### 4. Run Edge Inference Demo

```bash
python inference_edge.py
```

Demonstrates real-time inference suitable for edge devices.

## 📁 Project Structure

```
edge_ai/
├── train_model.py              # Model training script
├── convert_to_tflite.py        # TFLite conversion
├── test_tflite.py              # Model testing & benchmarking
├── inference_edge.py           # Edge device inference
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── EDGE_AI_REPORT.md           # Comprehensive report
└── models/                     # Generated models (created after training)
    ├── recyclable_classifier.h5
    ├── recyclable_classifier.tflite
    ├── recyclable_classifier_quantized.tflite
    ├── training_history.json
    ├── conversion_report.json
    └── test_report.json
```

## 🔧 Configuration

### Model Parameters

Edit the configuration in `train_model.py`:

```python
IMG_SIZE = 128          # Input image size
BATCH_SIZE = 32         # Training batch size
EPOCHS = 30             # Number of training epochs
NUM_CLASSES = 4         # Number of classes
LEARNING_RATE = 0.001   # Learning rate
```

### Class Names

Modify `CLASS_NAMES` in the scripts to match your dataset:

```python
CLASS_NAMES = ['paper', 'plastic', 'glass', 'metal']
```

## 📊 Expected Results

### Model Performance

- **Model Size**: 
  - Keras: ~2-3 MB
  - TFLite (float32): ~1.5-2 MB
  - TFLite (quantized): ~0.5-0.8 MB

- **Accuracy**: 80-90% (depends on dataset quality)

- **Inference Speed**:
  - CPU (laptop): 5-15 ms per image
  - Raspberry Pi 4: 20-50 ms per image
  - With Coral USB Accelerator: 2-5 ms per image

### Performance Comparison

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| Keras | 2-3 MB | Baseline | 100% |
| TFLite (float32) | 1.5-2 MB | 1.5-2x faster | ~99% |
| TFLite (quantized) | 0.5-0.8 MB | 2-4x faster | ~98% |

## 🎯 Edge AI Benefits

### 1. **Low Latency**
- Edge: 5-50 ms per inference
- Cloud: 100-500 ms (with network)
- **10-100x faster** for real-time applications

### 2. **Privacy**
- Data processed locally
- No transmission to cloud servers
- GDPR/privacy compliant

### 3. **Offline Operation**
- Works without internet
- No dependency on cloud services
- Reliable in any environment

### 4. **Cost Efficiency**
- No per-request API fees
- One-time deployment cost
- Scales to thousands of devices

## 🖥️ Deployment on Raspberry Pi

### Step 1: Install TensorFlow Lite Runtime

```bash
sudo apt update
sudo apt install python3-pip -y
pip3 install tflite-runtime numpy pillow
```

### Step 2: Transfer Model

```bash
# From your development machine
scp models/recyclable_classifier_quantized.tflite pi@raspberrypi:/home/pi/models/
```

### Step 3: Run Inference

```python
from inference_edge import RecyclableItemClassifier
import cv2

# Initialize
classifier = RecyclableItemClassifier('models/recyclable_classifier_quantized.tflite')

# With camera
camera = cv2.VideoCapture(0)
while True:
    ret, frame = camera.read()
    if ret:
        class_name, confidence = classifier.predict(frame)
        print(f"{class_name}: {confidence:.2%}")
```

## 📈 Real-World Applications

1. **Smart Recycling Bins**: Automatic item classification
2. **Mobile Apps**: Instant recycling guidance
3. **AR Applications**: Overlay classification in real-time
4. **Industrial Sorting**: Automated recycling facilities
5. **Education**: Teaching recycling awareness

## 🔍 Troubleshooting

### Issue: Model not found
**Solution**: Run `train_model.py` first, then `convert_to_tflite.py`

### Issue: Low accuracy
**Solution**: 
- Use a larger, more diverse dataset
- Increase training epochs
- Try transfer learning with MobileNet/EfficientNet

### Issue: Slow inference on Raspberry Pi
**Solution**:
- Use quantized model
- Consider Coral USB Accelerator
- Reduce input image size
- Use batch processing

### Issue: Out of memory
**Solution**:
- Use quantized model
- Reduce batch size
- Process images one at a time

## 📚 Additional Resources

- [TensorFlow Lite Documentation](https://www.tensorflow.org/lite)
- [Edge AI Best Practices](https://www.tensorflow.org/lite/performance)
- [Raspberry Pi Setup Guide](https://www.raspberrypi.org/documentation/)
- [Model Optimization Guide](https://www.tensorflow.org/model_optimization)

## 📄 Report

See `EDGE_AI_REPORT.md` for:
- Detailed methodology
- Complete results and metrics
- Deployment instructions
- Future improvements

## 🤝 Contributing

Feel free to:
- Add support for more classes
- Improve model architecture
- Add data augmentation techniques
- Optimize for specific edge devices

## 📝 License

This project is provided as-is for educational and demonstration purposes.

## 🙏 Acknowledgments

- TensorFlow team for TensorFlow Lite
- Raspberry Pi Foundation for edge computing platform
- Open source community for tools and libraries

---

**Version**: 1.0  
**Last Updated**: 2024

For questions or issues, please refer to the comprehensive report in `EDGE_AI_REPORT.md`.

