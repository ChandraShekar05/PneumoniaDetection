# Pneumonia Detection Model using CNN

This project aims to detect pneumonia in chest X-ray images using a Convolutional Neural Network (CNN). The model is trained on a dataset of labeled chest X-ray images and can classify images as either having pneumonia or being normal.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

Pneumonia is a serious lung infection that can be life-threatening, especially for children and the elderly. Early detection and diagnosis are crucial for effective treatment. This project leverages deep learning techniques to build a CNN model capable of classifying chest X-ray images as either 'Pneumonia' or 'Normal'.

## Dataset

The dataset used in this project is the **Chest X-Ray Images (Pneumonia)** dataset, which can be downloaded from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). It consists of:

- **Normal**: Chest X-ray images that are normal.
- **Pneumonia**: Chest X-ray images showing signs of pneumonia.

## Model Architecture

The CNN model used in this project consists of several convolutional layers followed by pooling layers and fully connected layers. The architecture is designed to extract and learn relevant features from the input images to effectively classify them.

- **Convolutional Layers**: Extract features from the input images.
- **Pooling Layers**: Reduce the spatial dimensions of the feature maps.
- **Fully Connected Layers**: Perform the final classification.

## Installation

To run this project, you need to have Python and the following libraries installed:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- OpenCV
- Pandas
- Streamlit

You can install the required libraries using the following command:

```bash
pip install tensorflow keras numpy matplotlib scikit-learn opencv-python pandas streamlit
```
## Usage

To run this run the following command in the terminal:

```bash
streamlit run main.py
