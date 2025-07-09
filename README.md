# 😷 Face Mask Detection with Lightweight Custom CNN (98.6% Accuracy)

This project presents an ultra-efficient **Custom Convolutional Neural Network (CNN)** architecture for classifying whether individuals are wearing a face mask or not. Despite containing only **~2 million trainable parameters**, the model achieves performance **comparable to heavier architectures like VGG16**, making it ideal for real-time or resource-constrained environments.

---

## 🧠 Key Features

- 🪶 **Lightweight Design**: ~2M parameters—perfect for deployment on edge devices.
- 🧱 **Deep but Efficient**: Uses `SeparableConv2D`, `BatchNormalization`, and `Dropout` for high efficiency and generalization.
- 🔁 **Data Augmentation**: Boosts model robustness using rotation, shear, zoom, flipping, and more.
- 📊 **98.6% Accuracy**: Achieved on real-world test data with impressive generalization.
- 📈 **Training Visuals**: Loss/Accuracy curves, classification report, and confusion matrix included.
- 🛡️ **Regularized and Stable**: L2 regularization + callbacks like `EarlyStopping` and `ReduceLROnPlateau`.

---

## 🗂️ Project Structure

Face-Mask-CNN/ 
├── notebooks/ 
│   └── fmd_cnn_notebook.ipynb 
├── dataset/ 
│   ├── with_mask/ 
│   └── without_mask/ 
├── models/ 
│   └── best_model.h5 
├── outputs/ 
│   └── confusion_matrix.png
│   └── Train & Validation Accuracy.png 
│   └── Train & Validation Loss.png  
├── requirements.txt └── README.md


---

## 📁 Dataset

### Face Mask Detection Dataset

#### About Dataset
Face Mask Detection Data set
In recent trend in world wide Lockdowns due to COVID19 outbreak, as Face Mask is became mandatory for everyone while roaming outside, approach of Deep Learning for Detecting Faces With and Without mask were a good trendy practice. Here I have created a model that detects face mask trained on 7553 images with 3 color channels (RGB).
On Custom CNN architecture Model training accuracy reached 94% and Validation accuracy 96%.

- **Classes**:
  - `With Mask` (Label: 1)
  - `Without Mask` (Label: 0)

#### Content
Data set consists of 7553 RGB images in 2 folders as with_mask and without_mask. Images are named as label with_mask and without_mask. Images of faces with mask are 3725 and images of faces without mask are 3828.

- **Source**: [Face Mask Dataset on Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- **Classes**:
  - `With Mask` (Label: 1)
  - `Without Mask` (Label: 0)

Images are resized to `128x128x3` RGB format during preprocessing.

---

## 🔬 Model Architecture Overview

- 5× Convolutional blocks (Conv2D + SeparableConv2D)
- BatchNormalization & Dropout in every major block
- Dense classification head with 3 hidden layers
- Output: Softmax layer with 2 neurons for binary classification

> 🧮 **Total Parameters**: ~2.1 million  
> 🚀 **Framework**: TensorFlow / Keras (GPU optimized)

---

## 🧪 Results

| Metric           | Value      |
|------------------|------------|
| Test Accuracy    | 98.61%     |
| Val Accuracy     | ~98%       |
| Loss Curve       | Stable     |
| Confusion Matrix | ✅ High Precision |

<p align="center">
  <img src="outputs/confusion_matrix.png" alt="Confusion Matrix" width="450"/>
</p>

<p align="center">
  <img src="outputs/Train & Validation Accuracy.png" alt="Train & Validation Accuracy" width="450"/>
</p>

<p align="center">
  <img src="outputs/Train & Validation Loss.png" alt="Train & Validation Loss" width="450"/>
</p>
---

## 📦 Installation

```bash
pip install -r requirements.txt


📜 License
This project is licensed under the Apache 2.0 License—free for personal and commercial use

🙌 Acknowledgments
Dataset curated by Omkar Gurav on Kaggle

CNN and training framework built using TensorFlow 2.x

