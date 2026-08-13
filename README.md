# Handwritten Digit Classification Using CNN

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge\&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge\&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge\&logo=keras)
![CNN](https://img.shields.io/badge/Model-CNN-green?style=for-the-badge)

## 📌 Project Overview

This project implements a **Convolutional Neural Network (CNN)** for handwritten digit recognition using the **MNIST dataset** available through TensorFlow/Keras.

The MNIST dataset contains **60,000 training images** and **10,000 testing images**. Each image is a **28 × 28 grayscale image** representing a handwritten digit from **0 to 9**.

The objective is to build a deep learning model that can automatically learn visual patterns such as edges, curves, and shapes and accurately classify handwritten digits.

---

## 🎯 Objectives

The project covers:

* Loading the MNIST dataset
* Exploring handwritten digit images
* Data preprocessing
* Pixel normalization
* Image reshaping for CNN input
* Building a CNN architecture
* Applying Max Pooling
* Applying Dropout regularization
* Training the CNN model
* Implementing Early Stopping
* Evaluating model performance
* Predicting handwritten digits

---

## 📊 Dataset

The project uses the **MNIST handwritten digit dataset** provided by TensorFlow/Keras.

```python
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

### Dataset Details

| Dataset  | Images | Image Size | Classes |
| -------- | -----: | ---------: | ------: |
| Training | 60,000 |    28 × 28 |      10 |
| Testing  | 10,000 |    28 × 28 |      10 |

### Classes

The model classifies handwritten digits into:

```text
0 1 2 3 4 5 6 7 8 9
```

### Input

Each image contains:

```text
28 × 28 = 784 pixels
```

Since the images are grayscale, the CNN input shape is:

```text
28 × 28 × 1
```

where `1` represents the grayscale channel.

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook / Google Colab

---

## 🧹 Data Preprocessing

### 1. Normalization

Pixel values in MNIST range from:

```text
0 to 255
```

They are normalized to:

```text
0 to 1
```

using:

```python
x_train = x_train / 255.0
x_test = x_test / 255.0
```

This helps the neural network train more efficiently.

### 2. Reshaping

For CNN input, a channel dimension is added:

```python
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
```

The resulting shapes are:

```text
x_train → (60000, 28, 28, 1)
x_test  → (10000, 28, 28, 1)
```

---

# 🧠 CNN Architecture

The model consists of convolutional layers, pooling layers, fully connected layers, and dropout regularization.

```text
Input Image
28 × 28 × 1
      ↓
Conv2D
32 Filters, 3×3, ReLU
      ↓
MaxPooling2D
2×2
      ↓
Conv2D
64 Filters, 3×3, ReLU
      ↓
MaxPooling2D
2×2
      ↓
Flatten
      ↓
Dense
128 Neurons, ReLU
      ↓
Dropout
0.5
      ↓
Dense
32 Neurons, ReLU
      ↓
Dropout
0.5
      ↓
Dense
10 Neurons, Softmax
      ↓
Prediction
0–9
```

---

## 🔍 CNN Components

### 1. Conv2D

Convolutional layers extract visual features from the images.

```python
Conv2D(32, kernel_size=(3,3), activation='relu')
Conv2D(64, kernel_size=(3,3), activation='relu')
```

The first convolutional layer learns basic features such as:

* Edges
* Lines
* Curves

The deeper convolutional layer learns more complex patterns.

---

### 2. MaxPooling2D

Max Pooling reduces the spatial dimensions of feature maps while retaining important information.

```python
MaxPooling2D(pool_size=(2,2))
```

Benefits:

* Reduces computation
* Reduces feature-map size
* Helps retain important features
* Provides some resistance to small positional changes

---

### 3. Flatten

After convolution and pooling, the feature maps are converted into a one-dimensional vector.

```python
Flatten()
```

For example:

```text
Feature Maps
     ↓
2D/3D representation
     ↓
Flatten
     ↓
1D vector
```

This allows the extracted features to be passed into the Dense layers.

---

### 4. Dense Layers

The model uses:

```python
Dense(128, activation='relu')
Dense(32, activation='relu')
```

These layers learn higher-level representations from the features extracted by the CNN.

---

### 5. Dropout

Dropout is used to reduce overfitting.

```python
Dropout(0.5)
```

During training, a percentage of neurons are randomly ignored.

The model uses two Dropout layers with a rate of `0.5`.

---

### 6. Output Layer

The final layer contains 10 neurons:

```python
Dense(10, activation='softmax')
```

Each neuron corresponds to one digit:

```text
Neuron 0 → Digit 0
Neuron 1 → Digit 1
...
Neuron 9 → Digit 9
```

Softmax produces a probability for each class.

Example:

```text
0 → 0.01
1 → 0.02
2 → 0.01
3 → 0.03
4 → 0.01
5 → 0.87  ← highest probability
6 → 0.01
7 → 0.02
8 → 0.01
9 → 0.01
```

Prediction:

```text
5
```

---

## ⏹️ Early Stopping

The model uses Keras `EarlyStopping` to prevent unnecessary training and reduce overfitting.

```python
EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

### Benefits

* Monitors validation loss
* Stops training when validation performance stops improving
* Prevents unnecessary epochs
* Restores the best model weights

---

## 📈 Model Training

During training, the following metrics are monitored:

* Training Loss
* Validation Loss
* Training Accuracy
* Validation Accuracy

Training and validation curves can be visualized using Matplotlib.

This helps identify:

* Overfitting
* Underfitting
* Training progress
* Model convergence

---

## 📊 Model Evaluation

The trained CNN is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

Example:

```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
```

The confusion matrix helps identify which digits are being confused by the model.

---

## 🔄 Project Workflow

```text
MNIST Dataset
      ↓
Load Dataset
      ↓
Data Exploration
      ↓
Normalize Pixel Values
      ↓
Reshape Images
      ↓
Build CNN
      ↓
Compile Model
      ↓
Train Model
      ↓
Early Stopping
      ↓
Evaluate Model
      ↓
Confusion Matrix
      ↓
Predict Handwritten Digits
```

---

## 💡 Key Learning Outcomes

Through this project, I practiced:

* Deep Learning Fundamentals
* Computer Vision
* Convolutional Neural Networks
* Image Classification
* Feature Extraction
* Convolution Operations
* Max Pooling
* Flattening
* Dense Neural Networks
* Dropout Regularization
* Softmax Classification
* Multi-class Classification
* Model Evaluation
* Early Stopping
* TensorFlow/Keras

---

## 📁 Project Structure

```text
MNIST-Digit-Classification-CNN/
│
├── MNIST_Digit_Classification_CNN.ipynb
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Dhruv-develop/mnist-digit-classification-cnn.git
```

### 2. Navigate to the Project

```bash
cd mnist-digit-classification-cnn
```

### 3. Install Required Libraries

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

### 4. Open the Notebook

Open:

```text
MNIST_Digit_Classification_CNN.ipynb
```

using **Jupyter Notebook** or **Google Colab**.

### 5. Run the Notebook

Run the cells sequentially to:

1. Load the MNIST dataset
2. Explore the images
3. Normalize pixel values
4. Reshape images
5. Build the CNN model
6. Train the model
7. Evaluate performance
8. Predict handwritten digits

---

## 📌 Conclusion

This project demonstrates the practical implementation of a **Convolutional Neural Network (CNN)** for handwritten digit classification using the MNIST dataset.

The CNN learns visual features through convolution and pooling operations and uses Dense layers for final classification. Dropout and Early Stopping are used to improve generalization and reduce overfitting.

The project provides hands-on experience with **Deep Learning, Computer Vision, CNN architecture, image preprocessing, model training, and multi-class image classification using TensorFlow/Keras**.

---

## 👨‍💻 Author

**Dhruv Rapariya**

M.Sc. CA & IT | Data Science Enthusiast

### Skills Demonstrated

`Python` `TensorFlow` `Keras` `CNN` `Deep Learning` `Computer Vision` `MNIST` `Image Classification` `NumPy` `Scikit-learn`
