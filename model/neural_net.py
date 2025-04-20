""""
* File: neural_net.py
* Project: MNIST Digit Classifier
* Author: Gabriel Souza
* Description: Defines the NeuralNetwork class, including forward and backward passes, dropout, and accuracy evaluation.
* Published: 2025-04-15
"""

from utils import backend
from .activation_functions import set_activation, activation_derivative
import cupy as cp
import numpy as np

class NeuralNetwork:
    """
    A simple fully connected feedforward neural network implementation using NumPy.
    Supports various activation functions, dropout, multiple weight initializations, and L2 regularization.
    """

    def __init__(self, layer_sizes, activation='relu', optimizer="sgd", init_type="he_scaling", dropout_rate=0.0, dropout=True):
        """
        Initializes the network structure and weights.

        Args:
            layer_sizes (tuple): Sizes of each layer (e.g., (784, 128, 64, 10)).
            activation (str): Activation function to use for hidden layers.
            optimizer (str): Optimizer type (e.g., 'sgd', 'adam').
            init_type (str): Weight initialization type ('he_scaling', 'xavier', 'default').
            dropout_rate (float): Dropout rate for hidden layers.
            dropout (bool): Whether to enable dropout during training.
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        self.dropout = dropout
        self.dropout_masks = [] # Store dropout masks for each layer
        self.init_type = init_type
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            if self.init_type == "he_scaling":
                # He initialization: good for ReLU and similar activations
                w = backend.np.random.randn(layer_sizes[i], layer_sizes[i+1]) * backend.np.sqrt(2. / layer_sizes[i]) # He initialization
            elif self.init_type == "xavier":
                # Xavier initialization: best for tanh/sigmoid activations.
                w = backend.np.random.randn(layer_sizes[i], layer_sizes[i+1]) * backend.np.sqrt(1. / layer_sizes[i])
            else:
                w = backend.np.random.randn(layer_sizes[i], layer_sizes[i+1]) # Default initialization
            b = backend.np.zeros((1, layer_sizes[i+1])) # Biases initialized to zero
            # Append weights and biases to the lists
            self.weights.append(w)
            self.biases.append(b)

    def load_checkpoint(self, path):
        """
        Loads model weights and biases from a .npz file.

        Args:
            path (str): Path to saved weights.
        """
        data = np.load(path)
        available_keys = data.files

        # Detect format
        if f'weight_0' in available_keys:
            self.weights = [data[f'weight_{i}'] for i in range(len(self.layer_sizes) - 1)]
            self.biases = [data[f'bias_{i}'] for i in range(len(self.layer_sizes) - 1)]
        elif f'W0' in available_keys:
            self.weights = [data[f'W{i}'] for i in range(len(self.layer_sizes) - 1)]
            self.biases = [data[f'b{i}'] for i in range(len(self.layer_sizes) - 1)]
        else:
            raise KeyError(f"Couldn't find expected weight/bias keys in file. Found: {available_keys}")

        print(f"[✅] Loaded model weights from: {path}")
    
    def enable_dropout(self):
        """Enables dropout for training."""
        self.dropout = True

    def disable_dropout(self):
        """Disables dropout for evaluation."""
        self.dropout = False

    # Forward pass
    def forward(self, x):
        """
        Performs a forward pass through the network.

        Args:
            x (ndarray): Input data.

        Returns:
            tuple: Activations and pre-activations at each layer.
        """
        activations = [x] # Stores activations layer by layer
        pre_activations = [] # Stores raw outputs before activation (z)
        self.dropout_masks = [] # Reset dropout masks for each forward pass
        # For every weight matrix
        for i in range(len(self.weights)):
            a_prev = activations[-1]               # shape: (batch_size, layer_sizes[i])
            W = self.weights[i]                    # shape: (layer_sizes[i], layer_sizes[i+1])
            b = self.biases[i]                     # shape: (1, layer_sizes[i+1])

            # 1) Linear step
            z = a_prev @ W + b                     # shape: (batch_size, layer_sizes[i+1])
            pre_activations.append(z)

            # 2) Nonlinear/dropout OR straight-through for last layer
            if i < len(self.weights) - 1:
                # hidden layer — apply activation
                a = set_activation(z, self.activation)
                # optional dropout
                if self.dropout and self.dropout_rate > 0:
                    mask = (backend.np.random.rand(*a.shape) > self.dropout_rate).astype(float)
                    a = (a * mask) / (1 - self.dropout_rate)
                    self.dropout_masks.append(mask)
                else:
                    self.dropout_masks.append(None)
            else:
                # output layer — no activation
                a = z
                self.dropout_masks.append(None)

            activations.append(a)

        return activations, pre_activations
    
    def backward(self, activations, pre_activations, dL_dz_output):
        """
        Performs backpropagation and computes gradients.

        Args:
            activations (list): Output of each layer.
            pre_activations (list): Raw activations before applying activation function.
            y_true (ndarray): True labels.

        Returns:
            tuple: Gradients for weights and biases.
        """
        # Create empty gradients lists
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        # Initial delta is the derivative from the output layer (dL/dz from softmax+CE)
        delta = dL_dz_output
        grads_w[-1] = activations[-2].T @ delta
        grads_b[-1] = backend.np.sum(delta, axis=0, keepdims=True)

        # Backpropagation error to hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            dropout_mask = self.dropout_masks[i]
            # Backpropagate delta through weights and activation
            # Defensive shape print (just while debugging)
            #print(f"[backward] layer {i}: delta.shape={delta.shape}, weights[{i+1}].T.shape={self.weights[i+1].T.shape}")

            # Matrix mult then activation derivative
            delta = delta @ self.weights[i + 1].T

            delta *= activation_derivative(pre_activations[i], self.activation) # core backprop formula where delta @ weights[i+1].T propagates the error backwards
            # apply the dropout_mask from forward pass if used
            if dropout_mask is not None:
                delta *= dropout_mask
                delta /= (1 - self.dropout_rate)

            # Compute gradients for each layer
            grads_w[i] = activations[i].T @ delta
            grads_b[i] = backend.np.sum(delta, axis=0, keepdims=True)

        for i, grad in enumerate(grads_w):
            if grad is None:
                print(f"[X] grads_w[{i}] is None!")

        for i, grad in enumerate(grads_b):
            if grad is None:
                print(f"[X] grads_b[{i}] is None!")

        return grads_w, grads_b
    

    # Predict class labels for input samples
    def predict(self, x):
        """
        Predicts class labels for input samples.

        Args:
            x (ndarray): Input data.

        Returns:
            ndarray: Predicted class indices.
        """
        activations, pre_activations = self.forward(x)
        # Apply softmax manually to output layer
        z = pre_activations[-1]
        exp_scores = backend.np.exp(z - backend.np.max(z, axis=1, keepdims=True))
        probs = exp_scores / backend.np.sum(exp_scores, axis=1, keepdims=True)
        return backend.np.argmax(probs, axis=1)
    
    # Compute accuracy on given data
    def accuracy(self, x, y):
        """
        Calculates classification accuracy.

        Args:
            x (ndarray): Input data.
            y (ndarray): True labels.

        Returns:
            float: Accuracy score.
        """
        predictions = self.predict(x)
        print("backend.isgpu", backend.IS_GPU)
        if backend.IS_GPU:
            y = cp.asarray(y)

        return backend.np.mean(predictions == y)