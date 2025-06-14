# 🧠 GDnet — Custom Neural Network Framework (NumPy/CuPy)

Welcome to **GDnet** — a self-built, NumPy and CuPy-powered deep learning framework. It was created as a learning journey into the internal workings of neural networks. GDnet supports both **CPU and GPU** training, and includes layers like `Dense`, `Conv2D`, `MaxPool2D`, `Flatten`, and more.

> ⚠️ Note: This is not meant to compete with PyTorch or TensorFlow — it's a ** framework** for learning, experimentation, and understanding how things work under the hood & not the best for production 😭.

---

## ✨ Features

* ✅ Fully object-oriented design
* ✅ CuPy (GPU) support with NumPy fallback
* ✅ Custom activation functions (`ReLU`, `Sigmoid`, `Softmax`, etc.)
* ✅ Built-in loss functions (`CrossEntropy`, `MSE`)
* ✅ Layer types: `Dense`, `Conv2D`, `MaxPool2D`, `Flatten`, `Dropout`
* ✅ Debug logger with file-based logging
* ✅ Early stopping, warmup epochs, L2/L1 regularization
* ✅ Custom `TextManager` for text preprocessing and vectorization
* ✅ Training with progress tracking and confusion matrix reporting
* ✅ Built-in model save/load system with GPU-safe pickling

---

## 📦 Installation

GDnet is a single-file framework. Just copy `GDnet.py` to your project and:

```python
import GDnet as gd
```

Or package it and install locally:

```bash
pip install -e .
```

---

## 🔍 Sample Usage

```python
import GDnet as gd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
y = gd.utils.one_hot(y, num_classes=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

layerdefs = [
    gd.LayerConfig(gd.DenseLayer, activation=gd.RELU(), output_size=64),
    gd.LayerConfig(gd.DenseLayer, activation=gd.Softmax(), output_size=10),
]

model = gd.Model(layerdefs, input_size=X.shape[1])
loss_fn = gd.CrossEntropy()

model.train(X_train, y_train, X_test, y_test,
            learning_rate=0.01, epochs=100, batch_size=32,
            loss_fn=loss_fn, early_stopping=True)

model.save("model.pkl")
```

---

## 🧐 Learn How Neural Networks Work

This project would **not have been possible** without these two **incredible resources** that explain complex concepts clearly:

* 📘 [Victor Zhou's Blog on Neural Networks](https://victorzhou.com/blog/intro-to-neural-networks/)
* 🎓 [StatQuest YouTube Channel](https://www.youtube.com/@statquest) — perfect for breaking down the math and intuition

If you’re trying to build something like this, start there! I did 🥹.

---

## 📁 Structure

| Component             | Description                                  |
| --------------------- | -------------------------------------------- |
| `Model`               | Core model logic and training loop           |
| `DenseLayer`          | Fully connected neural layer                 |
| `Conv2DLayer`         | Basic 2D convolution layer (with Numba/cuPy) |
| `MaxPool2D`           | Max pooling for image downsampling           |
| `Flatten`             | Reshape tensor for Dense input               |
| `Dropout`             | Regularization by randomly disabling units   |
| `TextManager`         | Basic TF-IDF + stemming based vectorization  |
| `CrossEntropy`, `MSE` | Loss functions                               |
| `DebugLogger`         | File logger to debug model internals         |

---

## 🚪 Requirements

* Python 3.11+
* `numpy`, `scikit-learn`, `nltk`
* Optional (for GPU): `cupy`, `numba`, `torch` (minimal fallback use)

---

## ❤️ Contributing

Feel free to fork, tweak, or use this as a base for your own learning journey. PRs that improve performance, readability, or structure are welcome!

---
