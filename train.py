""""
* File: train.py
* Project: MNIST Digit Classifier
* Author: Gabriel Souza
* Description: Main training entry point for MNIST/FashionMNIST classification using a NumPy neural network.
* Published: 2025-04-15
"""

import numpy as np
import tyro
from dataclasses import dataclass
from model.neural_net import NeuralNetwork
from model.optimizers import SGD, Adam, Adamax, RMSprop
from utils.schedules import exponential_decay_schedule, step_decay_schedule
from utils.helpers import create_metrics_dir, save_experiment_summary, save_model_weights, evaluate_model
from utils.data_loader import load_dataset
from utils.logger import setup_logger

@dataclass
class Train:
    """
    Train class encapsulates the configuration and execution of training a neural network on MNIST or Fashion MNIST.

    Attributes:
        dataset (str): Dataset name. Options: 'mnist' or 'fashion_mnist'.
        split_val (bool): Whether to split training data into train and validation sets.
        val_ratio (float): Fraction of training data to use as validation set.
        layer_sizes (tuple[int]): Architecture of the network.
        learning_rate (float): Initial learning rate.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size used during training.
        activation (str): Activation function name. (see models/activation_functions.py for available options)
        optimizer_type (str): Optimizer to use (see models/optimizers.py for available options).
        init_type (str): Weight initialization method. Options: 'he_scaling', 'xavier', 'default'.
        schedule_type (str): Learning rate scheduling strategy. Options: 'step' or 'exponential'.
        dropout (bool): Enable dropout.
        dropout_rate (float): # Dropout rate, set to 0.0 for no dropout, try 0.2-0.5
        lambda_l2 (float): L2 regularization coefficient. try 0.0001-0.001.
        metrics_output_dir (str): Output directory to save results. leave empty to use default.
    """
    # Dataset prep
    dataset: str = 'mnist'
    split_val: bool = True
    val_ratio: float = 0.1

    # Hyperparameters
    layer_sizes: tuple[int, ...] = (784, 128, 64, 10)
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 64
    activation: str = "leaky_relu"
    optimizer_type: str = "adamax"
    init_type: str = "he_scaling"
    schedule_type: str = 'exponential'
    dropout = True
    dropout_rate: float = 0.2
    lambda_l2: float = 0.00055
    metrics_output_dir: str = ''

    def __post_init__(self):
        """Initializes model, optimizer, and data."""
        # Track metrics
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        # Initialize the neural network
        self.model = NeuralNetwork(layer_sizes=self.layer_sizes, activation=self.activation, optimizer=self.optimizer_type,
                                   init_type=self.init_type, dropout_rate=self.dropout_rate, dropout=self.dropout)

        # set up lr decay
        if self.schedule_type.lower() in "step":
            schedule = lambda e: step_decay_schedule(
                e, self.learning_rate, final_lr=0.0001, total_epochs=self.epochs
            )
        elif self.schedule_type.lower() == "exponential":
            schedule = lambda e: exponential_decay_schedule(
                e, self.learning_rate, final_lr=0.0001, total_epochs=self.epochs
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.schedule_type}")
        

        # Initialize the optimizer
        if self.optimizer_type == "adam":
            self.optimizer = Adam(self.learning_rate, schedule=schedule)
        elif self.optimizer_type == "sgd":
            self.optimizer = SGD(self.learning_rate, schedule=schedule)
        elif self.optimizer_type == "rmsprop":
            self.optimizer = RMSprop(self.learning_rate, schedule=schedule)
        elif self.optimizer_type == "adamax":
            self.optimizer = Adamax(self.learning_rate, schedule=schedule)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
        self.optimizer.initialize(self.model) if hasattr(self.optimizer, "initialize") else None

        # Track params // used for saving experiment summary
        self.params_dict = {
        "dataset": self.dataset,
        "layer_sizes": self.layer_sizes,
        "learning_rate": self.learning_rate,
        "lr_schedule": self.schedule_type,
        "epochs": self.epochs,
        "batch_size": self.batch_size,
        "activation": self.activation,
        "optimizer": self.optimizer_type,
        "dropout": self.dropout,
        "dropout_rate": self.dropout_rate,
        "lambda_l2": self.lambda_l2,
        }
        self.metrics_output_dir = create_metrics_dir()

        # Load data
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = load_dataset(self.dataset, self.split_val, self.val_ratio)

        #set up logging
        self.logger = setup_logger(self.metrics_output_dir)

    def run_train(self):
        """Runs the full training loop and saves results/plots/checkpoints."""

        self.logger.info("Starting training...")
        self.logger.info(f"Parameters: {self.params_dict}")

        # Training loop
        for epoch in range(self.epochs):
            # Shuffle training data
            indices = np.arange(len(self.x_train))
            np.random.shuffle(indices)
            self.x_train = self.x_train[indices]
            self.y_train = self.y_train[indices]

            epoch_loss = 0.0
            num_batches = len(self.x_train) // self.batch_size

            self.optimizer.update_lr(epoch) # Update learning rate if using a schedule
            self.logger.info(f"[Epoch {epoch+1}] LR: {self.optimizer.lr:.6f}")

            for i in range(0, len(self.x_train), self.batch_size):
                x_batch = self.x_train[i:i+self.batch_size]
                y_batch = self.y_train[i:i+self.batch_size]

                # Forward Pass
                activations, pre_activations = self.model.forward(x_batch)

                # Compute loss (cross-entropy)
                batch_loss = self.model.cross_entropy_loss(activations[-1], y_batch, lambda_l2=self.lambda_l2)
                epoch_loss += batch_loss

                # Backward Pass
                grads_w, grads_b = self.model.backward(activations, pre_activations, y_batch)
                
                # Update weights and biases
                #self.model.update_parameters(grads_w, grads_b, lambda_l2=self.lambda_l2)
                self.optimizer.update(self.model, grads_w, grads_b, lambda_l2=self.lambda_l2)

            # Average loss for the epoch
            avg_loss = epoch_loss / num_batches
            self.train_losses.append(avg_loss)

            self.model.disable_dropout() # Disable dropout for evaluation

            # Accuracy calculation
            train_acc = self.model.accuracy(self.x_train, self.y_train)
            val_acc = self.model.accuracy(self.x_val, self.y_val)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.logger.info(f"Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
            self.model.enable_dropout() # Re-enable dropout for training

        results = evaluate_model(self.model, self.x_test, self.y_test, self.metrics_output_dir, self.train_losses, self.train_accuracies, self.val_accuracies)

         # Build a results summary
        results_dict = {
            "final_test_accuracy": results["test_acc"],
            "final_train_accuracy": results["train_accuracies"][-1],
            "final_val_accuracy": results["val_accuracies"][-1],
        }

        save_experiment_summary(self.params_dict, results_dict, self.metrics_output_dir)
        save_model_weights(self.model, self.metrics_output_dir)

        return results

if __name__ == "__main__":
    trainer: Train = tyro.cli(Train)
    results = trainer.run_train()
    trainer.logger.info(f"\nFinal Test Accuracy: {results['test_acc']*100:.2f}%")