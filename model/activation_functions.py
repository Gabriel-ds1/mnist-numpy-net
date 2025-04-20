""""
* File: activation_functions.py
* Project: MNIST Digit Classifier
* Author: Gabriel Souza
* Description: Contains activation functions and their derivatives, including custom and experimental activations.
* Published: 2025-04-15
"""

from utils import backend

# ===============================
# Standard Activation Functions
# ===============================

def relu(x):
    """Rectified Linear Unit (ReLU)"""
    return backend.np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU to prevent dead neurons"""
    return backend.np.where(x > 0, x, alpha * x)

def gelu(x):
    """Gaussian Error Linear Unit (GELU)"""
    return 0.5 * x * (1 + backend.np.tanh(backend.np.sqrt(2 / backend.np.pi) * (x + 0.044715 * backend.np.power(x, 3))))

# Softmax Activation Function
class Activation_Softmax:
    """ Softmax Activation Function """
    def forward(self, inputs):
        """ Forward pass """
        self.inputs = inputs
        # Get unnormalized probabilities
        exp_values = backend.np.exp(inputs - backend.np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / backend.np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = backend.np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = backend.np.diagflat(single_output) - backend.np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = backend.np.dot(jacobian_matrix, single_dvalues)


def sigmoid(x):
    """Sigmoid activation"""
    return 1 / (1 + backend.np.exp(-x))

def swish(x):
    """Swish = x * sigmoid(x)"""
    return x * sigmoid(x)  # Swish activation function

def mish(x):
    """Mish = x * tanh(softplus(x))"""
    softplus = backend.np.log1p(backend.np.exp(x)) # numerically stable softplus
    return x * backend.np.tanh(softplus)


# ===============================
# Standard Activation Derivatives
# ===============================

def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)

def leaky_relu_derivative(x, alpha=0.01):
    """Derivative of Leaky ReLU"""
    return backend.np.where(x > 0, 1.0, alpha)  # Derivative of Leaky ReLU

def gelu_derivative(x):
    """Derivative of the approximate GELU"""
    tanh_out = backend.np.tanh(backend.np.sqrt(2 / backend.np.pi) * (x + 0.044715 * x**3))
    left = 0.5 * tanh_out
    right = (0.5 * x * (1 - tanh_out**2) * backend.np.sqrt(2 / backend.np.pi) * (1 + 3 * 0.044715 * x**2))
    return left + right + 0.5

def sigmoid_derivative(x):
    """Derivative of sigmoid"""
    s = sigmoid(x)
    return s * (1 - s)

def swish_derivative(x):
    """Derivative of swish"""
    s = sigmoid(x)
    return s + x * (s * (1 - s))

def mish_derivative(x):
    """Derivative approximation for mish"""
    sp = backend.np.log1p(backend.np.exp(x))  # Softplus
    tsp = backend.np.tanh(sp)  # Tanh of softplus
    grad_sp = 1 - backend.np.exp(-sp) 
    return tsp + x * (1 - tsp**2) * grad_sp

# ===============================
# Custom / Experimental Functions
# ===============================

def reverse_relu(x):
    """Reverse ReLU activation function"""
    return backend.np.minimum(0, x)

def reverse_relu_derivative(x):
    """Derivative of Reverse ReLU"""
    return (x < 0).astype(float)

def reverse_leaky_relu(x):
    """reverse leaky relu activation function"""
    return backend.np.minimum(0, x) + 0.01 * backend.np.maximum(0, x)
def reverse_leaky_relu_derivative(x):
    return backend.np.where(x < 0, 1.0, 0.01)

def oscillating_relu(x):
    """oscillating relu activation function"""
    return backend.np.sin(x) * (x > 0)
def oscillating_relu_derivative(x):
    return backend.np.where(x > 0, backend.np.cos(x), 0)

def staircase_relu(x, steps=10):
    """quantized relu activation function"""
    return backend.np.floor(backend.np.maximum(0, x) * steps) / steps
def staircase_relu_derivative(x, steps=10):
    # Approximate the derivative as 1 on positive inputs (like ReLU), but scale down a bit
    return ((x > 0).astype(float)) * (1 - 1/steps)

def sin_relu(x):
    """SinReLU activation function"""
    return backend.np.maximum(0, x) * backend.np.sin(x)
def sin_relu_derivative(x):
    return backend.np.where(x > 0, backend.np.sin(x) + backend.np.maximum(0, x) * backend.np.cos(x), 0)

def bent_identity(x):
    """Bent Identity activation function"""
    return (x + (x**2) / 2) * (x > 0) + (x / 2) * (x <= 0)
def bent_identity_derivative(x):
    return backend.np.where(x > 0, 1 + x, 0.5)  # Derivative of Bent Identity

def elu_sin(x):
    """Exponential Linear Sine Unit"""
    return backend.np.where(x > 0, x * backend.np.sin(x), backend.np.exp(x) - 1)
def elu_sin_derivative(x):
    return backend.np.where(x > 0, backend.np.sin(x) + x * backend.np.cos(x), backend.np.exp(x))

def chaotic_relu(x):
    """Chaotic ReLU"""
    return backend.np.maximum(0, x) * backend.np.cos(3 * x)
def chaotic_relu_derivative(x):
    return backend.np.where(x > 0, backend.np.cos(3 * x) - 3 * x * backend.np.sin(3 * x), 0)

def gravity(x):
    """gravity activation function -- custom. created by: Gabriel Souza"""
    return (1 - backend.np.sin(0.25 * x)**2)
def gravity_derivative(x):
    return -0.5 * backend.np.sin(0.5 * x)

def gravity_x(x):
    """gravity_x activation function -- custom. created by: Gabriel Souza"""
    return backend.np.tanh(backend.np.sin(x))
def gravity_x_derivative(x):
    return (1 - backend.np.tanh(backend.np.sin(x))**2) * backend.np.cos(x)

def gravity_x_swish(x):
    """gravity_x_swish activation function -- custom. created by: Gabriel Souza"""
    return x * backend.np.tanh(backend.np.sin(x))
def gravity_x_swish_derivative(x):
    tanh_sin = backend.np.tanh(backend.np.sin(x))
    sech2 = 1 - tanh_sin**2
    return tanh_sin + x * sech2 * backend.np.cos(x)

def sin_exp_decay(x):
    return backend.np.sin(x) * backend.np.exp(-x**2)
def sin_exp_decay_derivative(x):
    return backend.np.exp(-x**2) * (backend.np.cos(x) - 2*x*backend.np.sin(x))

# ===============================
# Activation Lookup
# ===============================

activation_functions = {
    "relu": relu,
    "leaky_relu": leaky_relu,
    "gelu": gelu,
    "sigmoid": sigmoid,
    "swish": swish,
    "mish": mish,
    "reverse_relu": reverse_relu,
    "reverse_leaky_relu": reverse_leaky_relu,
    "oscillating_relu": oscillating_relu,
    "staircase_relu": staircase_relu,
    "sin_relu": sin_relu,
    "bent_identity": bent_identity,
    "elu_sin": elu_sin,
    "chaotic_relu": chaotic_relu,
    "gravity": gravity,
    "gravity_x": gravity_x,
    "gravity_x_swish": gravity_x_swish,
    "sin_exp_decay": sin_exp_decay,
}

activation_derivatives = {
    "relu": relu_derivative,
    "leaky_relu": leaky_relu_derivative,
    "gelu": gelu_derivative,
    "sigmoid": sigmoid_derivative,
    "swish": swish_derivative,
    "mish": mish_derivative,
    "reverse_relu": reverse_relu_derivative,
    "reverse_leaky_relu": reverse_leaky_relu_derivative,
    "oscillating_relu": oscillating_relu_derivative,
    "staircase_relu": staircase_relu_derivative,
    "sin_relu": sin_relu_derivative,
    "bent_identity": bent_identity_derivative,
    "elu_sin": elu_sin_derivative,
    "chaotic_relu": chaotic_relu_derivative,
    "gravity": gravity_derivative,
    "gravity_x": gravity_x_derivative,
    "gravity_x_swish": gravity_x_swish_derivative,
    "sin_exp_decay": sin_exp_decay_derivative,
}

# ===============================
# Dispatch Functions
# ===============================

def set_activation(x, activation):
    try:
        return activation_functions[activation](x)
    except KeyError:
        raise ValueError(f"Unsupported activation function: {activation}")

def activation_derivative(x, activation):
    try:
        return activation_derivatives[activation](x)
    except KeyError:
        raise ValueError(f"Unsupported activation function: {activation}")