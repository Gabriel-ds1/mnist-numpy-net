""""
* File: optimizers.py
* Project: MNIST Digit Classifier
* Author: Gabriel Souza
* Description: Implements various optimization algorithms (SGD, Adam, RMSprop, Adamax) with learning rate scheduling support.
* Published: 2025-04-15
"""

import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    """
    Abstract base class for all optimizers.
    Each optimizer must implement the `update` method.
    """
    @abstractmethod
    def update(self, model, grads_w, grads_b, lambda_l2):
        """
        Update model weights and biases.

        Args:
            model: Neural network model.
            grads_w: List of weight gradients for each layer.
            grads_b: List of bias gradients for each layer.
            lambda_l2: L2 regularization parameter.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def update_lr(self, epoch):
        """Optionally update learning rate (for scheduling)"""
        pass

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    """
    def __init__(self, learning_rate, schedule=None):
        self.lr = learning_rate
        self.schedule = schedule
    
    def update_lr(self, epoch):
        if self.schedule:
            self.lr = self.schedule(epoch)

    def update(self, model, grads_w, grads_b, lambda_l2):
        for i in range(len(model.weights)):
            grads_w[i] += lambda_l2 * model.weights[i]
            model.weights[i] -= self.lr * grads_w[i]
            model.biases[i] -= self.lr * grads_b[i]

class Adam(Optimizer):
    """
    Adam optimizer with bias correction.
    Combines momentum (first moment) and RMSProp (second moment).
    """
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, schedule=None):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.schedule = schedule
        self.t = 0

    def initialize(self, model):
        self.m_w = [np.zeros_like(w) for w in model.weights]
        self.v_w = [np.zeros_like(w) for w in model.weights]
        self.m_b = [np.zeros_like(b) for b in model.biases]
        self.v_b = [np.zeros_like(b) for b in model.biases]

    def update_lr(self, epoch):
        if self.schedule:
            self.lr = self.schedule(epoch)

    def update(self, model, grads_w, grads_b, lambda_l2):
        self.t += 1
        for i in range(len(model.weights)):
            grads_w[i] += lambda_l2 * model.weights[i]

            # First moment
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grads_w[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]

            # Second moment
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grads_w[i] ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grads_b[i] ** 2)

            # Bias correction
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            # Update
            model.weights[i] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            model.biases[i]  -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

class RMSprop(Optimizer):
    """
    RMSprop optimizer with adaptive learning rate.
    """
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-8, schedule=None):
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.schedule = schedule

    def initialize(self, model):
        self.v_w = [np.zeros_like(w) for w in model.weights]
        self.v_b = [np.zeros_like(b) for b in model.biases]

    def update_lr(self, epoch):
        if self.schedule:
            self.lr = self.schedule(epoch)

    def update(self, model, grads_w, grads_b, lambda_l2):
        for i in range(len(model.weights)):
            grads_w[i] += lambda_l2 * model.weights[i]

            self.v_w[i] = self.beta * self.v_w[i] + (1 - self.beta) * (grads_w[i] ** 2)
            self.v_b[i] = self.beta * self.v_b[i] + (1 - self.beta) * (grads_b[i] ** 2)

            model.weights[i] -= self.lr * grads_w[i] / (np.sqrt(self.v_w[i]) + self.epsilon)
            model.biases[i]  -= self.lr * grads_b[i] / (np.sqrt(self.v_b[i]) + self.epsilon)


class Adamax(Optimizer):
    """
    Adamax optimizer - a variant of Adam using the infinity norm.
    More stable under some conditions than standard Adam.
    """
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, schedule=None):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.schedule = schedule
        self.t = 0

    def initialize(self, model):
        self.m_w = [np.zeros_like(w) for w in model.weights]
        self.u_w = [np.zeros_like(w) for w in model.weights]
        self.m_b = [np.zeros_like(b) for b in model.biases]
        self.u_b = [np.zeros_like(b) for b in model.biases]

    def update_lr(self, epoch):
        if self.schedule:
            self.lr = self.schedule(epoch)

    def update(self, model, grads_w, grads_b, lambda_l2):
        self.t += 1
        for i in range(len(model.weights)):
            grads_w[i] += lambda_l2 * model.weights[i]

            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grads_w[i]
            self.u_w[i] = np.maximum(self.beta2 * self.u_w[i], np.abs(grads_w[i]))

            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]
            self.u_b[i] = np.maximum(self.beta2 * self.u_b[i], np.abs(grads_b[i]))

            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)

            model.weights[i] -= self.lr * m_w_hat / (self.u_w[i] + self.epsilon)
            model.biases[i]  -= self.lr * m_b_hat / (self.u_b[i] + self.epsilon)