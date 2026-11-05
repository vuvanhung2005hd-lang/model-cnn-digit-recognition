# ✍️ Handwritten Digit Recognition

A CNN built with PyTorch to recognize handwritten digits, plus an OpenCV script for real-time webcam testing.
Personal project to learn the fundamentals of Deep Learning and Computer Vision.

## 📈 Results

| Set | Accuracy (final epoch) |
|-----|------------------------|
| Training | 99.22% |
| Test (EMNIST Digits, 40,000 images) | **99.61%** |

Training accuracy is measured with augmentation and Dropout enabled, which is why it stays slightly below test accuracy.

### Training curves (20 epochs)

<img width="1000" height="500" alt="loss_curve" src="https://github.com/user-attachments/assets/5674ff47-1896-4e15-8fd7-64f2e7822d94" />

The training loss drops quickly in the first epochs and then decreases smoothly, showing a stable learning process.

<img width="1000" height="500" alt="accuracy_curve" src="https://github.com/user-attachments/assets/fcb788df-0b15-4b12-b131-ecef4374dbdd" />

Test accuracy stays consistently high from the first epochs and keeps improving slowly, without signs of overfitting.

> Webcam performance can still be affected by complex backgrounds or poor lighting.

## 📷 Webcam Demo

`camera.py` runs the trained model on a live webcam feed:

1. Crops a Region of Interest (ROI) from each frame.
2. Converts it to grayscale and applies thresholding.
3. Feeds the processed ROI to the CNN and shows the predicted digit on screen.

<!-- Drag and drop your demo GIF or screenshot here -->

**Tips for better results**
- Write the digit **thick and dark** on white paper.
- Use a plain background and good lighting.
- Keep the digit inside the ROI box.

## 🎯 Overview

- **Basic CNN architecture**: based on the classic LeNet-5, with Batch Normalization and Dropout for more stable training.
- **Webcam integration**: `camera.py` uses OpenCV to crop a Region of Interest (ROI), threshold the image and pass it to the trained model for real-time prediction.
- **Optimization**: Adam optimizer with a Cosine Annealing learning rate scheduler.
- **Data augmentation**: slight rotation, scaling and Gaussian blur on the training data, so the model performs better on real webcam feeds.

## 🧠 Model

- Conv block (x2): `Conv2d (5x5)` → `BatchNorm2d` → `ReLU` → `MaxPool2d (2x2)`
- Classifier: `Flatten (256)` → `Linear (120)` → `ReLU` → `Linear (84)` → `ReLU` → `Dropout (0.5)` → `Linear (10)`

## 📂 Dataset

EMNIST (Extended MNIST), `digits` split: 240,000 training images and 40,000 test images.

## ⚙️ Training setup

| Setting | Value |
|---------|-------|
| Optimizer | Adam (weight decay `<value>`) |
| Learning rate | `<value>`, Cosine Annealing scheduler |
| Epochs | 20 |
| Batch size | `<value>` |
| Augmentation | Slight rotation, scaling, Gaussian blur |

## 🗂️ Project structure

```
.
├── <train notebook / train.py>
├── camera.py
├── <model.pth>
├── requirements.txt
└── README.md
```

## 🚀 How to run

```bash
git clone https://github.com/vuvanhung2005hd-lang/model-cnn-digit-recognition.git
cd model-cnn-digit-recognition
pip install torch torchvision opencv-python numpy matplotlib
python camera.py
```
