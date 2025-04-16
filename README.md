# 🧠 NumPy Digit & Fashion Classifier

> A neural network framework built **entirely from scratch using NumPy** — no deep learning libraries, just math, logic, and code.
> Trains on **MNIST** or **Fashion-MNIST**, supports many optimizers and activations, and includes a GUI to classify real-time drawings.
> It supports training on **MNIST** or **Fashion-MNIST** datasets with a wide variety of custom activation functions, learning rate schedules, and optimizers. It also includes a simple Tkinter GUI where you can draw digits or fashion items and have the model predict your drawing. ✨

---

## 🚀 Features

- ✅ **Built entirely with NumPy – a true from-scratch deep learning model.**
- ✅ Supports **MNIST** and **Fashion-MNIST** datasets.
- ✅ Training, validation, and test split functionality.
- ✅ Custom neural network engine with:
  - Weight initializations: **He**, **Xavier**
  - Optimizers: **SGD**, **Adam**, **RMSprop**, **Adamax** – all manually implemented.
  - Activation functions: **ReLU**, **GELU**, **Swish** and more.. including brand new **custom** activation functions!
  - LR scheduling: **Step** and **Exponential decay**
  - **Dropout** & **L2 regularization**
- ✅ Training visualizations: loss & accuracy plots, confusion matrix.
- ✅Custom logging for tracking experiments.
- ✅ Easy CLI with [`tyro`](https://github.com/brentyi/tyro)
- ✅ **Tkinter GUI** for drawing digits/clothing and seeing model predictions
- ✅ Easy model saving/loading for testing or deployment.

---

## 💡 Custom Activation Functions

One of the unique aspects of this project is the design and testing of completely new activation functions — including experimental ones like:

- `reverse_relu`, `chaotic_relu`, `oscillating_relu`, `sin_exp_decay`
- `gravity`, `gravity_x`, and **`gravity_x_swish`** — a novel function that **outperformed ReLU and GELU when paired with Adamax**

> 💥 In benchmark testing, `gravity_x` + **Adamax** achieved **the highest test accuracy** outperforming all standard activations.

![Custom Activation Functions by: Gabriel Souza](https://i.imgur.com/SG3X9z4.png)

---

## ⭐ Architecture Overview

- Feedforward fully-connected neural network
- Manual forward & backward propagation
- Cross-entropy loss with softmax output
- Accurate gradient computations per activation

---

## 📁 Project Structure

```bash
├─data/                          # MNIST and/or Fashion MNIST are automatically saved here
├─images/                        # Project showcase images
├──model/
│   ├── neural_net.py            # Neural network implementation
│   ├── optimizers.py            # Manual optimizers (Adam, SGD, etc.)
│   ├── activation_functions.py  # Activation functions & derivatives
├─runs/                          # Experiment results will output here by default
├──utils/
│   ├── data_loader.py           # MNIST & Fashion-MNIST loaders
│   ├── helpers.py               # Logging, plotting, saving utils
│   ├── logger.py                # Logging, plotting, saving utils
│   ├── schedules.py             # Learning rate schedules
├──gui.py                        # Interactive drawing classification tool
├──train.py                      # Training entry point
├──test.py                       # Evaluation-only entry point
```

---

## 📦 Setup

```bash
pip install -r requirements.txt
```

Requires only:

- numpy
- matplotlib
- scikit-learn
- pillow
- tyro
- tkinter (standard with Python on most systems)

---

## 🖥️ GUI Demo (Tkinter)

![GUI Classification Example](https://i.imgur.com/fqWbzYo.gif)

Using the included `gui.py`, you can load any trained model and draw directly in a canvas interface:

```bash
python gui.py --weights runs/checkpoint_20250414-130219/model_weights.npz --dataset mnist
```

-Draw any digit (0–9) or fashion item (if using Fashion-MNIST).

-See the predicted class and probability bar chart.

-Swap models or datasets with ease.

---

## 🧪 Training

```bash
python train.py --activation relu --optimizer adam --epochs 20 --batch_size 64 --learning_rate 0.01
```

All training configs are available via CLI thanks to tyro. You can view a full list of available args within the train.py file.

---

## 📊 Testing

You can download the model weights with the best results from here:

- [MNIST checkpoint] (https://drive.google.com/uc?export=download&id=1FHzvdBak7tN1FAq-Kgqkn1G8RDTDH18g)
- [Fashion MNIST checkpoint] (https://drive.google.com/uc?export=download&id=1id3XKeb6jMsyNB9zIrj29hKWpbHmCbFh)

```bash
python test.py --load_path path/to/model_weights.npz --dataset mnist
```

---

## 📏 Sample Accuracy (MNIST, 200 Epochs)

| Activation      | Optimizer | Test Accuracy |
| --------------- | --------- | ------------- |
| ReLU            | Adamax    | 98.33%        |
| Swish           | Adamax    | 98.30%        |
| Sigmoid         | Adamax    | 96.86%        |
| Gravity_x_swish | Adamax    | 99.95%        |

> Full results and visualizations are saved under `/runs/checkpoint_*/`. Further testing is required with Fashion MNIST.

---

## 🧠 Author Notes

This project was a deep dive into the core mechanics of neural networks. Everything — from weight initialization to backpropagation, gradient descent to activation design — was written line-by-line in NumPy.

It’s meant not only as a showcase of skills but as an educational tool and playground for experimentation.

---

## 📣 Contact

If you're a recruiter or collaborator interested in AI, deep learning, or research-driven work — feel free to connect!

> gabesouza004@gmail.com
