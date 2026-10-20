# Handwritten Digit Recognition 📝

This is a personal project I built to learn the fundamentals of Deep Learning and Computer Vision. It uses a Convolutional Neural Network (CNN) built with PyTorch to recognize handwritten digits, along with a simple OpenCV script for real-time webcam testing.

## 🎯 Project Overview
**Basic CNN Architecture:** The model is based on the classic LeNet-5 structure, with a few modern additions like Batch Normalization and Dropout to improve training stability.
<img width="1398" height="312" alt="image" src="https://github.com/user-attachments/assets/35ea155c-139a-40ab-b021-b20047e5e22a" />
**Webcam Integration:** A simple script (`camera.py`) that uses OpenCV to crop a Region of Interest (ROI), threshold the image, and pass it to the trained model for prediction.
**Optimization:** The model is trained using the Adam optimizer and a Cosine Annealing learning rate scheduler.
**Data Augmentation:** I applied basic transformations (slight rotation, scaling, and Gaussian blur) to the training data to help the model perform better when tested on real webcam feeds.

## 🧠 Model & Dataset

- **Dataset:** EMNIST (Extended MNIST) - 'digits' split (240,000 training images, 40,000 test images).
- **Structure:** 
  - **Convolutional blocks:** Conv2d (5x5) ➔ BatchNorm2d ➔ ReLU ➔ MaxPool2d (2x2) *(applied twice)*.
  - **Fully Connected layers:** Flatten (256) ➔ Linear (120) ➔ ReLU ➔ Linear (84) ➔ ReLU ➔ Dropout (p=0.5) ➔ Linear (10) ➔ Softmax.

## 📊 Results
- **Training Accuracy:** 99.22%
- **Testing Accuracy:** 99.61%
- Here is how the model performed during the 20 epochs of training:
*The training loss decreased smoothly, showing a stable learning process.*
<img width="1000" height="500" alt="loss_curve" src="https://github.com/user-attachments/assets/5674ff47-1896-4e15-8fd7-64f2e7822d94" />
*The test accuracy stayed consistently high without significant overfitting.*
<img width="1000" height="500" alt="accuracy_curve" src="https://github.com/user-attachments/assets/fcb788df-0b15-4b12-b131-ecef4374dbdd" />
- *Note: While the model achieves high accuracy on the dataset, its real-world webcam performance can still be affected by complex backgrounds or poor lighting conditions.*
