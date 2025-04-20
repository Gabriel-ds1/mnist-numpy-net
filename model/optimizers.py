""""
* File: optimizers.py
* Project: MNIST Digit Classifier
* Author: Gabriel Souza
* Description: Implements various optimization algorithms (SGD, Adam, RMSprop, Adamax) with learning rate scheduling support.
* Published: 2025-04-15
"""

from utils import backend
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

class SGD_l2(Optimizer):
    """
    Stochastic Gradient Descent optimizer with l2 regularization
    """
    def __init__(self, learning_rate, schedule=None):
        self.lr = learning_rate
        self.schedule = schedule
    
    def update_lr(self, epoch):
        if self.schedule:
            self.lr = self.schedule(epoch)

    def update(self, model, grads_w, grads_b, lambda_l2):
        for i in range(len(model.weights)):
            # l2 regularization (prevent overfitting; keeps weights small and not rely too heavily on any one feature)
            grads_w[i] += lambda_l2 * model.weights[i]
            model.weights[i] -= self.lr * grads_w[i]
            model.biases[i] -= self.lr * grads_b[i]

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    grads_w[i]: the slope (gradient) of the loss with respect to the weights
    self.lr: the learning rate -- how big your step should be
    -= : subtract because we want to go downhill (i.e. minimize the loss)
    """
    def __init__(self, learning_rate, schedule=None):
        self.lr = learning_rate
        self.schedule = schedule
    
    def update_lr(self, epoch):
        if self.schedule:
            self.lr = self.schedule(epoch)

    def update(self, model, grads_w, grads_b):
        """Numerical mini-example for visualization:
        lets say, model.weights[i] = 0.9
        grads_w[i] = 0.3
        learning_rate = 0.1
        model.weights[i] = (0.9) - (0.1*0.3)
        so, model.weights[i] = 0.87
        new_position = current_position - learning_rate * slope"""
        for i in range(len(model.weights)):
            model.weights[i] -= self.lr * grads_w[i]
            model.biases[i]  -= self.lr * grads_b[i]

class SGD_Momentum(Optimizer):
    """
    Stochastic Gradient Descent optimizer with optional Momentum.
    self.momentum: momentum coefficient (typically 0.9)
    self.velocity_w / self.velocity_b: velocity (running average of gradients)
    grads_w[i]: the slope (gradient) of the loss with respect to the weights
    self.lr: the learning rate -- how big your step should be
    -= : subtract because we want to go downhill (i.e. minimize the loss)
    """
    def __init__(self, learning_rate, momentum=0.0, schedule=None):
        self.lr = learning_rate
        self.schedule = schedule
        self.velocity_w = None # will be initialized on first update
        self.velocity_b = None
    
    def update_lr(self, epoch):
        if self.schedule:
            self.lr = self.schedule(epoch)

    def update(self, model, grads_w, grads_b, lambda_l2):
        """example for visualization; lets say:
        grads_w[i] = 0.3 (current slope)
        learning_rate = 0.1
        model.weights[i] = 0.9
        momentum = 0.9
        velocity_w[i] = 0.2 (previous velocity)
        step 1 : Update Velocity
        self.velocity_w[i] = (0.9 * 0.2) + (0.1 * 0.3) = 0.21
        step 2: Update Weights
        new_weight = 0.9 - 0.21 = 0.69"""
        if self.velocity_w is None:
            # initialize velocity lists with zeros matching the model structure
            self.velocity_w = [backend.np.zeros_like(w) for w in model.weights]
            self.velocity_b = [backend.np.zeros_like(b) for b in model.biases]

        for i in range(len(model.weights)):
            # l2 regularization (prevent overfitting; keeps weights small and not rely too heavily on any one feature)
            grads_w[i] += lambda_l2 * model.weights[i]
            # Update velocity
            self.velocity_w[i] = self.momentum * self.velocity_w[i] + self.lr * grads_w[i]
            self.velocity_b[i] = self.momentum * self.velocity_b[i] + self.lr * grads_b[i]

            # Update parameters using velocity
            model.weights[i] -= self.velocity_w[i]
            model.biases[i]  -= self.velocity_b[i]

class Adagrad(Optimizer):
    """
    Adagrad optimizer (adaptive learning rate per parameter).
    Each parameter's learning rate is scaled by the inverse square root
    of the sum of the past squared gradients.
    epsilon is a small constant to avoid divide-by-zero
    """
    def __init__(self, learning_rate, epsilon=1e-8, schedule=None):
        self.lr = learning_rate
        self.schedule = schedule
        self.epsilon = epsilon
        self.G_w = None # Accumulated squared gradients for weights
        self.G_b = None # Accumulated squared gradients for biases
    
    def update_lr(self, epoch):
        if self.schedule:
            self.lr = self.schedule(epoch)

    def update(self, model, grads_w, grads_b, lambda_l2):
        """example for visualization; lets say:
        grads_w[i] = 0.3 (current slope)
        learning_rate = 0.1
        model.weights[i] = 0.9
        self.G_w[i] = 0.2 (previous sum of squared gradients)
        epsilon = 1e-8 (prevents divide by zero)
        step 1 : Sum of squared gradients
        sum_of_squared_gradients = 0.2 + 0.3 ^ 2 = 0.29
        step 2: Update Weights by the inverse square root
        new_weight = 0.9 - (0.1 / (sqrt(0.29 + 1e-8))) * 0.3
        new_weight = 0.9 - (0.1 / 0.5385) * 0.3
        new_weight = 0.9 - 0.1857 * 0.3 = 0.8443"""
        if self.G_w is None:
            # Initialize accumulated squared gradients, similar to velocity initialization
            self.G_w = [backend.np.zeros_like(w) for w in model.weights]
            self.G_b = [backend.np.zeros_like(b) for b in model.biases]

        for i in range(len(model.weights)):
            # l2 regularization
            grads_w[i] += lambda_l2 * model.weights[i]

            # Accumulate sum of squared gradients
            self.G_w[i] += grads_w[i] ** 2
            self.G_b[i] += grads_b[i] ** 2
            
            # Update parameters by the inverse square root of the sum of the past squared gradients.(element-wise division)
            model.weights[i] -= (self.lr / (backend.np.sqrt(self.G_w[i] + self.epsilon)) * grads_w[i])
            model.biases[i] -= (self.lr / (backend.np.sqrt(self.G_b[i] + self.epsilon)) * grads_b[i])


class RMSprop(Optimizer):
    """RMSprop optimizer.
    Keeps an exponentially decaying average of squared gradients to adapt the learning rate.
    Helps avoid vanishing learning rates like in Adagrad.
    --- similar to Adagrad but instead of (grads_w ^ 2) its now (decay_rate * avg of squared gradients + (1 - decay_rate) * (grads_w ^ 2)"""
    def __init__(self, learning_rate, decay_rate=0.9, epsilon=1e-8, schedule=None):
        self.lr = learning_rate
        self.rho = decay_rate
        self.epsilon = epsilon
        self.Eg_w = None # Exponential average of squared gradients (weights)
        self.Eg_b = None # Exponential average of squared gradients (biases)

    def update_lr(self, epoch):
        if self.schedule:
            self.lr = self.schedule(epoch)
    
    def update(self, model, grads_w, grads_b, lambda_l2):
        """example for visualization; lets say:
        grads_w[i] = 0.3 (current slope)
        learning_rate = 0.1
        model.weights[i] = 0.9
        self.Eg_w[i] = 0.2 (Exponential average of squared gradients)
        epsilon = 1e-8 (prevents divide by zero)
        rho = 0.9
        step 1 : Exponential average of squared gradients
        exp_avg_of_squared_gradients = 0.9 * 0.2 + (1 - 0.9) * (0.3 ^ 2) = 0.189
        step 2: Update Weights using RMS-scaled learning rate (same as Adagrad)
        new_weight = 0.9 - (0.1 / (sqrt(0.189 + 1e-8)) * 0.3)
        new_weight = 0.9 - (0.1 / 0.4347) * 0.3
        new_weight = 0.9 - 0.23 * 0.3 = 0.831"""
        # Initialize exponential avg of squared gradients
        if self.Eg_w is None:
            self.Eg_w = [backend.np.zeros_like(w) for w in model.weights]
            self.Eg_b = [backend.np.zeros_like(b) for b in model.biases]

        for i in range(len(model.weights)):
            # l2 regularization
            grads_w[i] += lambda_l2 * model.weights[i]

            # Update running average of squared gradients
            self.Eg_w[i] = self.rho * self.Eg_w[i] + (1 - self.rho) * (grads_w[i] ** 2)
            self.Eg_b[i] = self.rho * self.Eg_b[i] + (1 - self.rho) * (grads_b[i] ** 2)

            # Update parameters using RMS-scaled learning rate
            model.weights[i] -= (self.lr / (backend.np.sqrt(self.Eg_w[i] + self.epsilon)) * grads_w[i])
            model.biases[i] -= (self.lr / (backend.np.sqrt(self.Eg_b[i] + self.epsilon)) * grads_b[i])

class Adam(Optimizer):
    """
    Adam optimizer with bias correction.
    Combines momentum (first moment) + RMSProp (second moment) + bias correction.
    """
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, schedule=None):
        self.lr = learning_rate
        self.beta1 = beta1 # for momentum (1st moment)
        self.beta2 = beta2 # for RMSprop (2nd moment)
        self.epsilon = epsilon
        self.schedule = schedule
        self.t = 0 #timestamp

    def initialize(self, model):
        self.m_w = [backend.np.zeros_like(w) for w in model.weights] # 1st moment (momentum of weights)
        self.v_w = [backend.np.zeros_like(w) for w in model.weights] # 2nd moment (RMSprop of weights)
        self.m_b = [backend.np.zeros_like(b) for b in model.biases] # 1st moment (momentum of biases)
        self.v_b = [backend.np.zeros_like(b) for b in model.biases] # 2nd moment (RMSprop of biases)

    def update_lr(self, epoch):
        if self.schedule:
            self.lr = self.schedule(epoch)

    def update(self, model, grads_w, grads_b, lambda_l2):
        """example for visualization; lets say:
        grads_w[i] = 0.3 (current slope)
        learning_rate = 0.1
        model.weights[i] = 0.9
        self.beta1 = 0.9
        self.beta2 = 0.999
        epsilon = 1e-8 (prevents divide by zero)
        t = 2
        self.m_w = 0.2 # previous momentum
        self.v_w = 0.2 # previous running avg of squared gradients (RMSprop)

        step 1 : Calculate first moment (momentum)
        self.m_w[i] = 0.9 * 0.2 + (1 - 0.9) * 0.3 = 0.21
        
        step 2: Calculate second moment (RMSprop)
        self.v_w[i] = 0.999 * 0.2 + (1 - 0.999) * (0.3^2) = 
        self.v_w[i] = 0.1998 + 0.001 * 0.09 = 0.19989
        
        step 3 : Bias correction
        m_w_hat = 0.21 / (1 - 0.9 ^ 2) = 1.1053
        v_w_hat = 0.19989 / (1 - 0.999 ^ 2) = 99.944
        
        step 4: update params
        model.weights[i] = 0.9 - (0.1 / sqrt(99.944 + 1e-8) * 1.1053)
        model.weights[i] = 0.9 - (0.1 / 9.9998) * 1.1053 = 0.88894"""

        self.t += 1 #timestamp
        for i in range(len(model.weights)):
            # l2 regularization
            grads_w[i] += lambda_l2 * model.weights[i] 

            # First moment (momentum) - similar to SGD with momentum but instead of velocity, we have exponential moving average of the gradient
            # no learning rate is applied during this accumulation though, like in SGD with momentum, the LR is only applied once at final update step
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grads_w[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]

            # Second moment (RMSprop) - same exact as RMS prop
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grads_w[i] ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grads_b[i] ** 2)

            # Bias correction - this is important in the early stages, without it the updates would be very small and 
            # our optimizer would act overly cautious in the first few steps (dont get confused with bias param)
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            # Parameter update
            model.weights[i] -= (self.lr / (backend.np.sqrt(v_w_hat) + self.epsilon)) * m_w_hat
            model.biases[i]  -= (self.lr / (backend.np.sqrt(v_b_hat) + self.epsilon)) * m_b_hat


class Adamax(Optimizer):
    """
    Adamax optimizer - a variant of Adam using the infinity norm.
    More stable under some conditions than standard Adam.
    Uses:
    - m : exponential moving average of gradients (momentum)
    - u : exponential moving max of gradient magnitudes (infinity norm)
    """
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, schedule=None):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.schedule = schedule
        self.t = 0

    def initialize(self, model):
        self.m_w = [backend.np.zeros_like(w) for w in model.weights]
        self.u_w = [backend.np.zeros_like(w) for w in model.weights]
        self.m_b = [backend.np.zeros_like(b) for b in model.biases]
        self.u_b = [backend.np.zeros_like(b) for b in model.biases]

    def update_lr(self, epoch):
        if self.schedule:
            self.lr = self.schedule(epoch)

    def update(self, model, grads_w, grads_b, lambda_l2):
        """example for visualization; lets say:
        grads_w[i] = 0.3 (current slope)
        learning_rate = 0.1
        model.weights[i] = 0.9
        self.beta1 = 0.9
        self.beta2 = 0.999
        epsilon = 1e-8 (prevents divide by zero)
        t = 2
        self.m_w = 0.2 # previous momentum
        self.u_w = 0.2 # exponential moving max of gradient magnitudes (in regular Adam this is RMSprop instead)

        step 1 : Calculate first moment (momentum)
        self.m_w[i] = 0.9 * 0.2 + (1 - 0.9) * 0.3 = 0.21 (same exact as regular Adam)
        
        step 2: Calculate running max of gradient sizes (exponential moving max of gradient magnitudes)
        self.u_w[i] = max(0.999 * 0.2, abs(0.3)) -> (np.abs gets the absolute value(how far away from 0 it is, whether positive or negative))
        self.u_w[i] = max(0.1998, 0.3) = 0.3
        
        step 3 : Bias correction
        m_w_hat = 0.21 / (1 - 0.9 ^ 2) = 1.1053
        
        step 4: update params
        model.weights[i] = 0.9 - 0.1 * 1.1053 / (0.3 + 1e-8)
        model.weights[i] = 0.9 - 0.11053 / (0.30000001) = 0.5316"""
                
        self.t += 1
        for i in range(len(model.weights)):
            # l2 regularization
            grads_w[i] += lambda_l2 * model.weights[i]

            # 1st moment (momentum)
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grads_w[i]
            # Infinity norm (max of past scaled gradients)
            self.u_w[i] = backend.np.maximum(self.beta2 * self.u_w[i], backend.np.abs(grads_w[i]))

            # 1st moment (momentum) now for bias
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]
            # Infinity norm (max of past scaled gradients) now for bias
            self.u_b[i] = backend.np.maximum(self.beta2 * self.u_b[i], backend.np.abs(grads_b[i]))

            # bias correction (dont get confused with bias param)
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)

            # update params
            model.weights[i] -= self.lr * m_w_hat / (self.u_w[i] + self.epsilon)
            model.biases[i]  -= self.lr * m_b_hat / (self.u_b[i] + self.epsilon)