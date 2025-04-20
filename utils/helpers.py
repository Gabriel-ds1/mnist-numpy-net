""""
* File: helpers.py
* Project: MNIST Digit Classifier
* Author: Gabriel Souza
* Description: Utility functions for saving metrics, plotting results, saving model weights, and evaluation.
* Published: 2025-04-15
"""

import os
import json
from utils import backend
import cupy as cp
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def create_metrics_dir():
    """
    Creates a unique output directory based on current timestamp for saving run artifacts.

    Returns:
        str: Path to the newly created directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("runs", f"{"checkpoint"}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def ensure_dir_exists(path):
    """
    Creates the directory if it doesn't already exist.

    Args:
        path (str): Directory path to ensure exists.
    """
    os.makedirs(path, exist_ok=True)

def save_experiment_summary(params: dict, results: dict, output_dir: str, filename="summary.json"):
    """
    Saves training configuration and final results to a JSON summary file.

    Args:
        params (dict): Hyperparameters and config used during training.
        results (dict): Final results like accuracy, loss, etc.
        output_dir (str): Directory to save the summary file.
        filename (str): Name of the output JSON file.
    """
    ensure_dir_exists(output_dir)
    summary = {
        "parameters": params,
        "results": results,
    }

    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"[✅] Summary saved to: {path}")

def save_model_weights(model, output_dir, filename="model_weights.npz"):
    """
    Saves model weights and biases as a compressed .npz archive.

    Args:
        model: NeuralNetwork instance containing trained weights.
        output_dir (str): Directory to save the weights.
        filename (str): Filename to use for saved .npz archive.
    """
    ensure_dir_exists(output_dir)
    weights = {f"weight_{i}": (w.get() if backend.IS_GPU else w) for i, w in enumerate(model.weights)}
    biases = {f"bias_{i}": (b.get() if backend.IS_GPU else b) for i, b in enumerate(model.biases)}
    path = os.path.join(output_dir, filename)
    backend.np.savez(path, **weights, **biases)
    print(f"[✅] Model weights saved to: {path}")

def plot_metrics(train_losses, train_accuracies, val_accuracies, output_dir):
    """
    Plots training loss and accuracies over epochs and saves the plot.

    Args:
        train_losses (list): List of training loss values.
        train_accuracies (list): List of training accuracy values.
        val_accuracies (list): List of validation accuracy values.
        output_dir (str): Directory to save the generated plot.
    """
    ensure_dir_exists(output_dir)
    # Plot training loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "metrics.png")
    plt.savefig(path)
    plt.close()
    print(f"[✅] Plot saved to: {path}")

def plot_confusion_matrix(model, x, y, output_dir, title="Confusion Matrix"):
    """
    Generates and saves a confusion matrix plot.

    Args:
        model: Trained model instance with a predict() method.
        x (np.ndarray): Input data.
        y (np.ndarray): True labels.
        output_dir (str): Directory to save the plot.
        title (str): Title of the plot.
    """
    ensure_dir_exists(output_dir)
    predictions = model.predict(x)
    if backend.IS_GPU:
        predictions = predictions.get()
        y = y.get()
    cm = confusion_matrix(y, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=backend.np.arange(10) if not backend.IS_GPU else backend.np.arange(10).get())
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(title)
    plt.grid(False)
    
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    print(f"[✅] Confusion matrix saved to: {path}")

def evaluate_model(model, x_test, y_test, metrics_output_dir, train_losses=None, train_accuracies=None, val_accuracies=None, test_only=False):
    """
    Evaluates the model on the test set, generates optional plots, and returns a result dictionary.

    Args:
        model: Trained neural network model.
        x_test (np.ndarray): Test input data.
        y_test (np.ndarray): Test labels.
        metrics_output_dir (str): Directory to save plots and logs.
        train_losses (list): List of training losses (optional).
        train_accuracies (list): List of training accuracies (optional).
        val_accuracies (list): List of validation accuracies (optional).
        test_only (bool): If True, disables plotting and metric saving.

    Returns:
        dict: Dictionary with test accuracy and optionally training/validation metrics.
    """
    # Evaluation on the validation set
    model.disable_dropout()
    test_acc = model.accuracy(x_test, y_test)
    if not test_only:
        plot_confusion_matrix(model, x_test, y_test, metrics_output_dir, title="Test Set Confusion Matrix")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        if backend.IS_GPU:
            train_losses = [cp.asnumpy(l) for l in train_losses] if train_losses else None
            train_accuracies = [cp.asnumpy(a) for a in train_accuracies] if train_accuracies else None
            val_accuracies = [cp.asnumpy(a) for a in val_accuracies] if val_accuracies else None
        # Only plot if all metrics provided
        if train_losses is not None and train_accuracies is not None and val_accuracies is not None:
            plot_metrics(train_losses, train_accuracies, val_accuracies, metrics_output_dir)
    # Save parameters
    return {
    "test_acc": test_acc,
    "train_losses": train_losses, 
    "train_accuracies": train_accuracies,
    "val_accuracies": val_accuracies,
}