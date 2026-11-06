# ✍️ Handwritten Digit Recognition

## 📌 Project Overview

This project focuses on developing a Convolutional Neural Network (CNN) using PyTorch to recognize handwritten digits (0–9) from image data.

The model is trained on the EMNIST Digits dataset and integrated with OpenCV to perform real-time digit recognition through a webcam.

This personal project aims to explore the fundamentals of Deep Learning and Computer Vision, covering CNN architecture design, data augmentation, model optimization, performance evaluation, and real-time image processing.

---

## 📊 Results & Evaluation

| Dataset | Accuracy |
| :--- | :---: |
| Training Set | 99.22% |
| Test Set (40,000 images) | **99.61%** |

**Evaluation Metric: Accuracy**

The model achieved 99.61% accuracy on the EMNIST Digits test set after 20 training epochs.

The training accuracy is slightly lower than the test accuracy because data augmentation and Dropout are enabled during training, introducing additional regularization.

### Training Performance

The following curves illustrate the model's training progress over 20 epochs.

**Training Loss**

<img width="1000" height="500" alt="loss_curve" src="https://github.com/user-attachments/assets/5674ff47-1896-4e15-8fd7-64f2e7822d94" />)

The training loss decreases rapidly during the initial epochs and continues to decline smoothly, indicating stable optimization.

**Model Accuracy**

<img width="1000" height="500" alt="accuracy_curve" src="https://github.com/user-attachments/assets/fcb788df-0b15-4b12-b131-ecef4374dbdd" />

The model maintains high test accuracy throughout training, with gradual improvements over subsequent epochs.

> Note: Webcam prediction performance may vary depending on handwriting style, image quality, lighting conditions, and background complexity.

---

## 🛠️ Methodology & Implementation

### 1. Dataset Preparation

Used the EMNIST (Extended MNIST) dataset, specifically the `digits` split.

| Dataset | Number of Images |
| :--- | ---: |
| Training Set | 240,000 |
| Test Set | 40,000 |
| Number of Classes | 10 |
| Image Type | Grayscale |
| Prediction Classes | 0–9 |

The dataset contains handwritten digit images labeled from 0 to 9.

### 2. Data Augmentation

Applied data augmentation techniques to improve the model's ability to recognize variations in handwritten digits.

Techniques include:

- **Random Rotation:** Introduces slight rotational variations in handwritten digits.
- **Scaling:** Simulates differences in digit size.
- **Gaussian Blur:** Introduces slight image blur to improve robustness to image quality variations.

These transformations help the model generalize beyond the original training images and improve its ability to handle real-world webcam input.

### 3. CNN Architecture

Designed a CNN architecture based on the classic LeNet-5, incorporating Batch Normalization and Dropout for improved training stability and regularization.

**Network Architecture:**

**Convolutional Feature Extraction:**

Two convolutional blocks, each consisting of:

```text
Conv2d (5x5)
    ↓
BatchNorm2d
    ↓
ReLU
    ↓
MaxPool2d (2x2)
```

**Classification Head:**

```text
Flatten (256)
    ↓
Linear (120)
    ↓
ReLU
    ↓
Linear (84)
    ↓
ReLU
    ↓
Dropout (0.5)
    ↓
Linear (10)
```

The convolutional layers extract spatial features from handwritten digit images, while the fully connected layers perform the final classification into 10 digit classes.

### 4. Model Training & Optimization

Implemented the training process using PyTorch.

**Training Configuration:**

| Setting | Value |
| :--- | :--- |
| Framework | PyTorch |
| Architecture | LeNet-5-inspired CNN |
| Optimizer | Adam |
| Learning Rate Scheduler | CosineAnnealingLR |
| Epochs | 20 |
| Batch Size | See training configuration |
| Weight Decay | See training configuration |
| Dropout | 0.5 |
| Data Augmentation | Rotation, Scaling, Gaussian Blur |

**Optimization Techniques:**

- **Adam Optimizer:** Updates model parameters using adaptive learning rates.
- **Cosine Annealing Scheduler:** Gradually adjusts the learning rate following a cosine schedule.
- **Batch Normalization:** Normalizes intermediate feature activations to support stable training.
- **Dropout:** Randomly disables neurons during training to reduce overfitting.

### 5. Model Evaluation

Evaluated the trained CNN using the EMNIST Digits test set.

The evaluation process includes:

- Calculating training accuracy after each epoch.
- Measuring test accuracy on 40,000 images.
- Monitoring training loss and accuracy curves.
- Comparing training and test performance to examine model generalization.

The final model achieved **99.61% test accuracy** after 20 epochs.

---

## 📷 Real-Time Webcam Recognition

Integrated the trained PyTorch model with OpenCV to perform real-time handwritten digit recognition.

The `camera.py` script captures webcam frames and processes handwritten digits through the trained CNN.

**Image Processing Pipeline:**

1. **Frame Capture:** Captures live frames from the webcam.
2. **Region of Interest (ROI):** Extracts the selected region from each frame.
3. **Grayscale Conversion:** Converts the ROI into a single-channel grayscale image.
4. **Thresholding:** Separates the handwritten digit from the background.
5. **Model Inference:** Passes the processed image to the trained CNN.
6. **Prediction Display:** Displays the predicted digit on the webcam feed.

**Tips for Better Recognition:**

- Write digits using a thick, dark marker on white paper.
- Use a plain background with sufficient lighting.
- Keep the handwritten digit inside the ROI box.
- Avoid excessive rotation or overlapping strokes.

---

## 💻 Tech Stack

| Category | Technologies |
| :--- | :--- |
| Programming Language | Python |
| Deep Learning Framework | PyTorch, Torchvision |
| Computer Vision | OpenCV |
| Numerical Processing | NumPy |
| Data Visualization | Matplotlib |
| Dataset | EMNIST Digits |

---

## 📁 Project Structure

```text
model-cnn-digit-recognition/
│
├── weights/             # Saved model weights
│
├── LICENSE              # Project license
├── README.md            # Project documentation
├── camera.py            # Real-time digit recognition using webcam
├── dataset.py           # Dataset loading and preparation
├── model.py             # CNN model architecture
├── test_overall.py      # Model testing and evaluation
├── train.py             # Model training script
└── utils.py             # Utility functions
```


## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/vuvanhung2005hd-lang/model-cnn-digit-recognition.git

cd model-cnn-digit-recognition
```

### 2. Install Dependencies

Ensure that Python 3.8 or later is installed.

```bash
pip install -r requirements.txt
```

Alternatively, install the dependencies manually:

```bash
pip install torch torchvision opencv-python numpy matplotlib
```

### 3. Run the Webcam Demo

Make sure that your webcam is connected and the trained model weights are available.

```bash
python camera.py
```

The webcam window will display the live camera feed and the predicted digit.

---

## 🎯 Key Learnings

Through this project, I gained practical experience in:

- Understanding CNN architecture and convolutional feature extraction.
- Implementing Deep Learning models using PyTorch.
- Applying data augmentation to improve model generalization.
- Using Batch Normalization and Dropout for regularization.
- Optimizing model training with Adam and Cosine Annealing.
- Evaluating classification performance using accuracy and training curves.
- Integrating trained PyTorch models with OpenCV.
- Building a real-time Computer Vision application using webcam input.

---























