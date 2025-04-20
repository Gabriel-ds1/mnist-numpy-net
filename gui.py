""""
* File: gui.py
* Project: MNIST Digit Classifier
* Author: Gabriel Souza
* Description: Provides a Tkinter-based GUI for drawing and classifying images with a trained neural network.
* Published: 2025-04-15

This script launches a simple GUI using Tkinter to allow users to draw digits or fashion items
and classify them using a trained neural network. Displays prediction results and probability bars.

Usage:
    python gui.py --weights path/to/model_weights.npz --dataset mnist (options for dataset: mnist, fashion_mnist)
"""

import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from model.neural_net import NeuralNetwork

def load_trained_model(weights_path=None, dataset='mnist'):
    """
    Loads a NeuralNetwork model and optionally its weights.

    Args:
        weights_path (str): Path to saved model weights (.npz).
        dataset (str): Type of dataset (mnist or fashion_mnist).

    Returns:
        NeuralNetwork: Loaded model.
    """
    if dataset == "mnist":
        layer_sizes = (784, 512, 512, 512, 256, 128, 64, 10)
    else:
        layer_sizes = (784, 512, 512, 512, 256, 128, 64, 10)
    model = NeuralNetwork(layer_sizes=layer_sizes)
    if weights_path:
        model.load_checkpoint(weights_path)
    else:
        print("Warning: using random weights!")
    return model

MNIST_LABELS = [str(i) for i in range(10)]
FASHION_LABELS = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

class DrawClassifierApp:
    """
    GUI application for classifying user-drawn digits/fashion items using a trained model.
    """
    def __init__(self, root, model, dataset='mnist'):
        self.root = root
        self.root.title(f"{dataset.upper()} Draw Classifier")
        self.dataset = dataset
        self.model = model
        self.label_names = MNIST_LABELS if dataset == "mnist" else FASHION_LABELS

        # Make GUI Larger
        self.CANVAS_SIZE = 420   # Larger drawing area
        self.IMG_SIZE = 28
        self.BAR_WIDTH = 320     # Bar chart as wide as the canvas
        self.BAR_HEIGHT = 260    # Height for 10 bars

        root.geometry(f"{self.CANVAS_SIZE + 60}x{self.CANVAS_SIZE + self.BAR_HEIGHT + 220}")

        # Drawing canvas
        self.canvas = tk.Canvas(root, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=3, padx=15, pady=15)
        self.canvas.bind('<B1-Motion>', self.draw_lines)

        # PIL surface
        self.image1 = Image.new("L", (self.CANVAS_SIZE, self.CANVAS_SIZE), 'white')
        self.draw = ImageDraw.Draw(self.image1)

        # Buttons and result label
        tk.Button(root, text='Classify', height=2, width=12, command=self.classify_drawing).grid(row=1, column=0, pady=6)
        tk.Button(root, text='Clear', height=2, width=12, command=self.clear).grid(row=1, column=1, pady=6)
        self.result_label = tk.Label(root, text="Draw and press Classify", font=("Courier", 18))
        self.result_label.grid(row=2, column=0, columnspan=3, pady=10)

        # Bar chart canvas for probabilities
        self.prob_canvas = tk.Canvas(root, width=self.BAR_WIDTH, height=self.BAR_HEIGHT, bg='white', bd=2, relief='ridge')
        self.prob_canvas.grid(row=3, column=0, columnspan=3, padx=20, pady=8)

    def draw_lines(self, event):
        """Draws lines on both the canvas and the PIL image object as the user moves the mouse."""
        radius = 10
        x0, y0 = (event.x - radius), (event.y - radius)
        x1, y1 = (event.x + radius), (event.y + radius)

        self.canvas.create_oval(x0, y0, x1, y1, fill='black', outline='black')
        self.draw.ellipse([x0, y0, x1, y1], fill='black')

    def preprocess(self):
        """
        Processes the drawn image into a 28x28 normalized numpy array for model input.

        Returns:
            np.ndarray: Flattened input array of shape (1, 784).
        """
        image = self.image1.resize((self.IMG_SIZE, self.IMG_SIZE), Image.LANCZOS)
        image = ImageOps.invert(image)
        data = np.asarray(image) / 255.0
        flat = data.flatten().reshape(1, -1)
        return flat

    def classify_drawing(self):
        """Preprocesses the image and displays the model's prediction and probability bar chart."""
        x = self.preprocess()
        activations, _ = self.model.forward(x)
        output = activations[-1].flatten()
        pred_class = int(np.argmax(output))
        label_name = self.label_names[pred_class]
        self.result_label.config(text=f"Prediction: {label_name}")
        self.draw_prob_bar_chart(output)

    def draw_prob_bar_chart(self, probs):
        """
        Draws a horizontal bar chart of output probabilities.

        Args:
            probs (np.ndarray): Output probabilities from the model.
        """
        # Clear previous bars
        self.prob_canvas.delete('all')
        n = len(probs)
        bar_h = int(self.BAR_HEIGHT / n) - 3  # Bar height per class

        max_prob = np.max(probs)

        for i, (prob, label) in enumerate(zip(probs, self.label_names)):
            # Bar length proportional to probability
            bar_length = int(prob * (self.BAR_WIDTH - 120))

            y0 = i * (bar_h + 3) + 8
            y1 = y0 + bar_h

            color = '#448aff' if prob < max_prob else '#43a047'  # highlight max
            self.prob_canvas.create_rectangle(110, y0, 110+bar_length, y1, fill=color, outline='black', width=1)
            self.prob_canvas.create_text(10, y0 + bar_h//2, anchor='w',
                                        text=str(label), font=('Courier', 13))

            self.prob_canvas.create_text(
                110 + bar_length + 8, y0 + bar_h//2, anchor='w',
                text=f"{prob:.2%}", font=('Courier', 12), fill="#212121"
            )

    def clear(self):
        """Clears the canvas, resets label, and removes the bar chart."""
        self.canvas.delete('all')
        self.draw.rectangle([0, 0, self.CANVAS_SIZE, self.CANVAS_SIZE], fill='white')
        self.result_label.config(text="Draw and press Classify")
        # Clear bar chart
        self.prob_canvas.delete('all')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=None, help='Path to saved weights')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='mnist')
    args = parser.parse_args()

    model = load_trained_model(args.weights, args.dataset)
    root = tk.Tk()
    app = DrawClassifierApp(root, model, dataset=args.dataset)
    root.mainloop()