""""
* File: neural_net.py
* Project: MNIST Digit Classifier
* Author: Gabriel Souza
* Description: Defines the NeuralNetwork class, including forward and backward passes, dropout, and accuracy evaluation.
* Published: 2025-04-15
"""

import numpy as np
from .activation_functions import set_activation, activation_derivative

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
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i]) # He initialization
            elif self.init_type == "xavier":
                # Xavier initialization: best for tanh/sigmoid activations.
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1. / layer_sizes[i])
            else:
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) # Default initialization
            b = np.zeros((1, layer_sizes[i+1])) # Biases initialized to zero
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
        try:
            self.weights = [data[f'weight_{i}'] for i in range(len(self.layer_sizes) - 1)]
            self.biases = [data[f'bias_{i}'] for i in range(len(self.layer_sizes) - 1)]
        except KeyError:
            self.weights = [data[f'W{i}'] for i in range(len(self.layer_sizes) - 1)]
            self.biases = [data[f'b{i}'] for i in range(len(self.layer_sizes) - 1)]

        print(f"[✅] Loaded model weights from: {path}")

    # Cross-entropy loss function
    def cross_entropy_loss(self, y_pred, y_true, lambda_l2=0.0):
        """
        Computes cross-entropy loss with optional L2 regularization.

        Args:
            y_pred (ndarray): Predicted probabilities.
            y_true (ndarray): True class labels.
            lambda_l2 (float): L2 regularization parameter.

        Returns:
            float: Total loss value.
        """
        n_samples = y_pred.shape[0]
        clipped_preds = np.clip(y_pred, 1e-15, 1 - 1e-15) # Avoid log(0)
        log_probs = -np.log(clipped_preds[range(n_samples), y_true])
        data_loss = np.mean(log_probs)

        # L2 penalty
        l2_penalty = 0.0
        if lambda_l2 > 0:
            for w in self.weights:
                l2_penalty += np.sum(np.square(w))
            l2_penalty *= (lambda_l2 / 2)
        return data_loss + l2_penalty
    
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
        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            a = set_activation(z, self.activation)

            # Apply dropout if training (skip output layer)
            if self.dropout and self.dropout_rate > 0 and i < len(self.weights) - 2:
                dropout_mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(float)
                a *= dropout_mask
                a /= (1 - self.dropout_rate) # Scale to keep expectation the same
                self.dropout_masks.append(dropout_mask)
            else:
                self.dropout_masks.append(None) # No dropout mask for this layer

            activations.append(a)

        # Output layer
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        pre_activations.append(z)
        a = set_activation(z, 'softmax')
        activations.append(a)

        return activations, pre_activations
    
    def backward(self, activations, pre_activations, y_true):
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
        # batch size for computing gradients
        n = y_true.shape[0]

        # Convert labels to one-hot encoding
        y_onehot = np.zeros_like(activations[-1])
        y_onehot[np.arange(n), y_true] = 1

        # Compute the gradient of the loss with respect to the output layer
        delta = (activations[-1] - y_onehot) / n # Gradient of softmax + cross-entropy loss
        grads_w[-1] = activations[-2].T @ delta
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True)

        # Backpropagation error to hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            dropout_mask = self.dropout_masks[i]
            delta = delta @ self.weights[i+1].T * activation_derivative(pre_activations[i], self.activation)
            if dropout_mask is not None:
                delta *= dropout_mask
                delta /= (1 - self.dropout_rate)

            grads_w[i] = activations[i].T @ delta
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)

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
        activations, _ = self.forward(x)
        return np.argmax(activations[-1], axis=1)
    
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
        return np.mean(predictions == y)