# 🧠 GDnet — Custom Neural Network Framework (NumPy/CuPy)

Welcome to **GDnet** — a self-built, NumPy and CuPy-powered deep learning framework. It was created as a learning journey into the internal workings of neural networks. GDnet supports both **CPU and GPU** training, and includes layers like `Dense`, `Conv2D`, `MaxPool2D`, `Flatten`, and more.

> ⚠️ Note: This is not meant to compete with PyTorch or TensorFlow — it's a ** Learning framework** for learning, experimentation, and understanding how things work under the hood & not the best for production 😭.
![logo](gdnet.png)
---

## ✨ Features

* ✅ Fully object-oriented design
* ✅ CuPy (GPU) support with NumPy fallback
* ✅ Custom activation functions (`ReLU`, `Sigmoid`, `Softmax`, etc.)
* ✅ Built-in loss functions (`CrossEntropy`, `MSE`)
* ✅ Layer types: `Dense`, `Conv2D`, `MaxPool2D`, `Flatten`, `Dropout`, `Embedding`,`DebugLayer`
* ✅ Transformer layers: `FeedForward`, `TransformerBlock`
* ✅ Debug logger with file-based logging
* ✅ Early stopping, warmup epochs, L2/L1 regularization
* ✅ Custom `TextManager` for text preprocessing and vectorization
* ✅ Training with progress tracking and confusion matrix reporting
* ✅ Built-in model save/load system with GPU-safe pickling
* ✅ install directly using pip install gdnet
---

## 📦 Installation
you can clone the repo folder in your project and import it like this:

```python
from gdnet.layers import DenseLayer, Conv2DLayer, MaxPool2D, Flatten, Dropout
from gdnet.lossfunctions import CrossEntropy, MSE
from gdnet.core import Model
from gdnet.utils import TextManager
```


```bash
pip install gdnet
```

---

## 🔍 Sample Usage

```python
from gdnet.core import Model,LayerConfig
from gdnet.layers import  Conv2DLayer, DenseLayer, Flatten
from gdnet.activations import RELU, Softmax
from gdnet.lossfunctions import CrossEntropy
from gdnet.utils import AugmentImage
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import time
# === Load and preprocess MNIST ===
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
print("Successfully fetched DataSet")
X = mnist.data.astype(np.float32) / 255.0
X = X.reshape(-1, 1, 28, 28)
print(f"Input shape: {X.shape}")
y = mnist.target.astype(int)
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.1, random_state=42)
del mnist, X, y, y_onehot
Augmentor =  AugmentImage(rotate="random", shift="random",zoom="random")
X_train = Augmentor.augment_library(X_train)
layer_defs = [
     LayerConfig( Conv2DLayer,  RELU(), input_shape=(1, 28, 28), num_filters=8, filter_size=5, stride=2, padding=2),
     LayerConfig( Conv2DLayer,  RELU(), num_filters=16, filter_size=3, stride=2, padding=1),
     LayerConfig( Flatten, activation=None),
     LayerConfig( DenseLayer,  RELU(), output_size=64),
     LayerConfig( DenseLayer,  Softmax(), output_size=10)
]
model =  Model(layer_defs, input_size=(1, 28, 28), regularization='l2')
print("Starting Training")
try:
    model.train(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        learning_rate=0.01,
        epochs=100, 
        batch_size=128,
        verbose=True,
        loss_fn= CrossEntropy(),
        early_stopping=True,
        lambda_=0.0001, 
        patience=5,
        warmup_epochs=2
    )
    print("\nSaving model...")
    model.save("mnist-gdnet-conv.pkl")
except Exception as e:
    print(f"Training failed: {e}")
```

---
## Results of the training
Example Results:
    * [MNIST](https://github.com/ghua8088/gdnet-mnist)
---
## 🧐 Learn How Neural Networks Work

This project would **not have been possible** without these two **incredible resources** that explain complex concepts clearly:

* 📘 [Victor Zhou's Blog on Neural Networks](https://victorzhou.com/blog/intro-to-neural-networks/)
* 🎓 [StatQuest YouTube Channel](https://www.youtube.com/@statquest) — perfect for breaking down the math and intuition

If you’re trying to build something like this, start there! I did 🥹.

---

## 📁 Structure
________________________________________________________________________
| Component             | Description                                  |
| --------------------- | -------------------------------------------- |
| `core.py`             | Core model logic and training loop           |
| `layers/`             | All neural network layers                    |
| `activations/`        | All activation functions                     |
| `lossfunctions/`      | All loss functions                           |
| `utils/`              | All utility functions                        |
| `optimizers/`         | All optimizers                               |
| `transformers/`       | All transformers                             |
| `gpu.py`              | GPU-Helper                                   |
------------------------------------------------------------------------

## 🚪 Requirements

* Python 3.11+
* `numpy`, `scikit-learn`, `nltk`
* Optional (for GPU): `cupy`, `numba`, `torch` (minimal fallback use)

---

## ❤️ Contributing

Feel free to fork, tweak, or use this as a base for your own learning journey. PRs that improve performance, readability, or structure are welcome!
Hoping to push this to the next level, I'll be adding more layers and optimizers soon.
and I'll be adding more examples soon.
good luck! 😊
Have fun learning!
---
