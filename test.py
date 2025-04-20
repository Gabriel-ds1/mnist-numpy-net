""""
* File: test.py
* Project: MNIST Digit Classifier
* Author: Gabriel Souza
* Description: Script to load saved model weights and evaluate the model on test data.
* Published: 2025-04-15
"""

import tyro
from dataclasses import dataclass
from model.neural_net import NeuralNetwork
from utils.data_loader import load_dataset
from utils.helpers import evaluate_model
from utils.logger import setup_logger

@dataclass
class Test:
    """
    A utility class for testing a pre-trained neural network model on a dataset.

    Attributes:
        dataset (str): Dataset to use ('mnist' or 'fashion_mnist').
        split_val (bool): Whether to split a portion of the training data for validation.
        val_ratio (float): Ratio of training data to use for validation, only used if split_val is True.
        layer_sizes (tuple): Sizes of each layer in the network.
        learning_rate (float): Learning rate (not used in test mode but kept for compatibility).
        activation (str): Activation function name.
        optimizer (str): Optimizer type used during training (not used in testing).
        dropout (bool): Whether to apply dropout (disabled during testing).
        dropout_rate (float): Dropout rate used during training (not used in testing).
        metrics_output_dir (str): Output directory for logs and plots.
        load_path (str): Path to the saved model weights (.npz).
    """
    # Dataset prep
    dataset: str = 'mnist'
    split_val: bool = True
    val_ratio: float = 0.1

    # Hyperparameters
    layer_sizes: tuple[int, ...] = (784, 512, 512, 512, 256, 128, 64, 10)
    learning_rate: float = 0.01
    activation: str = "gravity_x"
    optimizer: str = "adamax"
    dropout = False
    dropout_rate: float = 0.2
    metrics_output_dir: str = ''

    # Testing params
    load_path: str = ""

    def __post_init__(self):
        """
        Initializes model and loads data. Raises an error if no model path is provided.
        """
        # Ensure a model path is provided
        if not self.load_path:
            raise ValueError("`--load_path` is required in test-only mode.")

        # Track dummy metrics to maintain compatibility with evaluate_model()
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        """Load model weights and data"""
        # Initialize model (same structure must match trained model)
        self.model = NeuralNetwork(
            layer_sizes=self.layer_sizes,
            learning_rate=self.learning_rate,
            activation=self.activation,
            optimizer=self.optimizer,
            dropout_rate=self.dropout_rate,
            dropout=self.dropout,
        )

        # Load dataset (train/val/test) to evaluate model performance
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = load_dataset(
            self.dataset, self.split_val, self.val_ratio
        )

        #set up logging
        self.logger = setup_logger(self.metrics_output_dir)

    def run_test(self):
        """
        Loads weights and evaluates the model on the test set.

        Returns:
            dict: Evaluation results including test accuracy.
        """
        self.model.load_checkpoint(self.load_path)

        results = evaluate_model(self.model, self.x_test, self.y_test, self.metrics_output_dir, test_only=True)
        return results
    
if __name__ == "__main__":
    tester: Test = tyro.cli(Test)
    results = tester.run_test()
    if results and 'test_acc' in results:
        tester.logger.info(f"\nFinal Test Accuracy: {float(results['test_acc'])*100:.2f}%")
    else:
        tester.logger.info("No test results obtained")