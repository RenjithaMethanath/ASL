# 🤟 ASL Sign Language Recognition using Deep Learning

A deep learning-based image classification project that recognizes American Sign Language (ASL) hand gestures using Convolutional Neural Networks and transfer learning with VGG16.

## 📌 Project Summary

This project focuses on interpreting ASL hand gestures through an image classifier built with Keras and TensorFlow. It uses a pre-trained VGG16 model as the base and fine-tunes it to detect 29 distinct ASL symbols (A-Z + del, nothing, space).

## 🚀 Key Features

- ✅ Trained on 87,000+ labeled ASL images
- 🔁 Data augmentation and class balancing
- 🧠 Transfer Learning with VGG16
- 🎯 82.3% accuracy on validation set
- 🎥 Real-time gesture recognition with webcam
- 📊 Confusion matrix, misclassification report, and class-wise accuracy
- 🔡 Text-to-sign and sign-to-text conversion functionality

## 🧰 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Scikit-learn
- Seaborn / Matplotlib
- Imbalanced-learn

## 🗂️ Dataset

- Source: [ASL Alphabet Dataset (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- Contains images for 29 categories with ~3,000 samples per class

## 🛠️ Model Architecture

- Base Model: VGG16 (pre-trained on ImageNet)
- Custom Top Layers: Flatten + Dense(29, softmax)
- Loss: Categorical Crossentropy
- Optimizer: Adam
- Metrics: Accuracy

## 📈 Results

- Final Validation Accuracy: **82.3%**
- Most Misclassified: `U` ↔ `R`, `N` ↔ `M`

## 🖥️ Live Demo

- Open your webcam and start signing! The model predicts letters in real-time.
- Example command to run:
  ```bash
  python asl_webcam_demo.py
