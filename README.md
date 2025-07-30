# Pneumonia Detection with CNN (PyTorch)

This project implements a Convolutional Neural Network (CNN) in PyTorch to classify chest X-ray images as either **Normal** or **Pneumonia**. It trains a custom model from scratch on a dataset of X-ray images to learn and detect signs of pneumonia.

---

## 📁 Dataset

The dataset is structured as follows:

data/
├── train/
│ ├── NORMAL/
│ └── PNEUMONIA/
└── test/
├── NORMAL/
└── PNEUMONIA/


- All images are grayscale chest X-rays.
- Labels are assigned automatically using `torchvision.datasets.ImageFolder`.

---

Input (1 x 128 x 128)
↓
Conv2d → ReLU → MaxPool
↓
Conv2d → ReLU → MaxPool
↓
Conv2d → ReLU → MaxPool
↓
Flatten
↓
Linear → ReLU → Dropout
↓
Linear → Output (2 classes: Normal / Pneumonia)
