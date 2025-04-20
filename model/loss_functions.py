from utils import backend
from .activation_functions import Activation_Softmax

# Common loss class
class Loss:
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = backend.np.mean(sample_losses)
        # Return loss
        return data_loss

# Cross-Entropy Loss Class
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent dicision by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = backend.np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -> only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples), y_true]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidence = backend.np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_likelihoods = -backend.np.log(correct_confidence)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)
        #number of labels in every sample, we'll use the first sample to count them
        labels = len(dvalues[0])

        #if labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = backend.np.eye(labels)[y_true]
        
        # calculate gradient
        self.dinputs = -y_true / dvalues

        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Softmax_Loss_CatCrossEntropy():
    # creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        # output layers activation function
        self.activation.forward(inputs)
        # set the output
        self.output = self.activation.output
        # calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        #number of samples
        samples = len(dvalues)
        # if labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = backend.np.argmax(y_true, axis=1)

        # copy to safely modify
        self.dinputs = dvalues.copy()
        # calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # normalize gradient
        self.dinputs = self.dinputs / samples

# another version of Cross-entropy loss function
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
    clipped_preds = backend.np.clip(y_pred, 1e-15, 1 - 1e-15) # Avoid log(0)
    log_probs = -backend.np.log(clipped_preds[range(n_samples), y_true])
    data_loss = backend.np.mean(log_probs)

    # L2 penalty
    l2_penalty = 0.0
    if lambda_l2 > 0:
        for w in self.weights:
            l2_penalty += backend.np.sum(backend.np.square(w))
        l2_penalty *= (lambda_l2 / 2)
    return data_loss + l2_penalty

def exponential_decay_schedule(epoch, initial_lr=0.01, final_lr=0.001, total_epochs=100):
    """
    Returns the learning rate for a given epoch based on an exponential decay schedule.
    
    Parameters:
        epoch (int): The current epoch number.
        initial_lr (float): The starting learning rate.
        final_lr (float): The desired final learning rate after total_epochs.
        total_epochs (int): The total number of epochs for training.
    
    Returns:
        float: The learning rate for the current epoch.
    """
    # Calculate the decay rate so that at epoch total_epochs, lr == final_lr
    decay_rate = (final_lr / initial_lr) ** (1 / total_epochs)
    # Calculate the learning rate for the current epoch
    lr = initial_lr * (decay_rate ** epoch)
    return lr

def step_decay_schedule(epoch, initial_lr=0.01, final_lr=0.001, total_epochs=100, step_size=30):
    """
    Returns the learning rate for a given epoch based on a step decay schedule.

    Parameters:
        epoch (int): The current epoch number.
        initial_lr (float): The starting learning rate.
        final_lr (float): The desired final learning rate after total_epochs.
        total_epochs (int): The total number of epochs for training.
        step_size (int): Number of epochs between each decay.

    Returns:
        float: The learning rate for the current epoch.
    """
    # Estimate how many times to decay over the total_epochs
    num_decays = total_epochs // step_size
    # Compute the decay factor needed
    decay_factor = (final_lr / initial_lr) ** (1 / max(1, num_decays))
    # Compute how many steps have passed
    num_steps = epoch // step_size
    lr = initial_lr * (decay_factor ** num_steps)
    return lr