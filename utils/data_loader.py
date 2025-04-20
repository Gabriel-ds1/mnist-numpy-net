"""
data_loader.py

This module handles downloading, loading, preprocessing, and splitting of MNIST and Fashion-MNIST datasets.
"""

from utils import backend
import os
import urllib.request
import gzip

def load_mnist(split_val=True, val_ratio=0.1):
    """
    Loads the MNIST dataset from a remote source or local cache.
    
    Args:
        split_val (bool): Whether to split a validation set from the training data.
        val_ratio (float): Ratio of training data to use for validation.
    
    Returns:
        Tuple of (x_train, y_train, x_val, y_val, x_test, y_test) if split_val is True.
        Otherwise, returns (x_train, y_train, x_test, y_test).
    """
    # Download MNIST dataset if not already present
    url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    path = 'data/mnist/mnist.npz'

    if not os.path.exists('data/mnist'):
        os.makedirs('data/mnist')
    if not os.path.exists(path):
        print("Downloading MNIST dataset...")
        urllib.request.urlretrieve(url, path)

    with backend.np.load(path) as data:
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']
    
    # Normalize and flatten
    x_train = x_train.reshape(-1, 28*28) / 255.0
    x_test = x_test.reshape(-1, 28*28) / 255.0

    if not split_val:
        return x_train, y_train, x_test, y_test
    
    # Split validation set from training data
    split_idx = int(len(x_train) * (1 - val_ratio))
    x_val, y_val = x_train[split_idx:], y_train[split_idx:]
    x_train, y_train = x_train[:split_idx], y_train[:split_idx]

    print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Testing data shape: {x_test.shape}, Testing labels shape: {y_test.shape}")

    return x_train, y_train, x_val, y_val, x_test, y_test

def load_fashion_mnist(split_val=True, val_ratio=0.1, shuffle=False):
    """
    Loads the Fashion-MNIST dataset from a remote source or local cache.

    Args:
        split_val (bool): Whether to split a validation set from the training data.
        val_ratio (float): Ratio of training data to use for validation.
        shuffle (bool): Whether to shuffle the data before splitting.

    Returns:
        Tuple of (x_train, y_train, x_val, y_val, x_test, y_test) if split_val is True.
        Otherwise, returns (x_train, y_train, x_test, y_test).
    """

    # Download Fashion-MNIST dataset if not already present
    url_base = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = [
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",  "t10k-labels-idx1-ubyte.gz"
    ]
    path = 'data/fashion_mnist'
    if not os.path.exists(path):
        os.makedirs(path)

    # Download files if not exist
    for fname in files:
        url = url_base + fname
        out_path = os.path.join(path, fname)
        if not os.path.exists(out_path):
            print(f"Downloading {fname} ...")
            urllib.request.urlretrieve(url, out_path)

    # Load data
    train_images = load_images(os.path.join(path, "train-images-idx3-ubyte.gz"))
    train_labels = load_labels(os.path.join(path, "train-labels-idx1-ubyte.gz"))
    test_images = load_images(os.path.join(path, "t10k-images-idx3-ubyte.gz"))
    test_labels = load_labels(os.path.join(path, "t10k-labels-idx1-ubyte.gz"))

    x_train_full, y_train_full = train_images, train_labels
    x_test, y_test = test_images, test_labels

    # Normalize and flatten
    x_train_full = x_train_full.reshape(-1, 28*28) / 255.0
    x_test = x_test.reshape(-1, 28*28) / 255.0

    if not split_val:
        return x_train_full, y_train_full, x_test, y_test

    # Shuffle before split
    n = len(x_train_full)
    idx = backend.np.arange(n)
    if shuffle:
        backend.np.random.shuffle(idx)
    x_train_full = x_train_full[idx]
    y_train_full = y_train_full[idx]

    # Split validation set
    split_idx = int(n * (1 - val_ratio))
    x_train, x_val = x_train_full[:split_idx], x_train_full[split_idx:]
    y_train, y_val = y_train_full[:split_idx], y_train_full[split_idx:]

    print("Training set shape:", x_train.shape)
    print("Validation set shape:", x_val.shape)
    print("Test set shape:", x_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test

def load_images(filename):
    """
    Reads image data from a compressed IDX file.

    Args:
        filename (str): Path to the .gz file containing image data.

    Returns:
        NumPy array of shape (num_images, rows, cols).
    """

    with gzip.open(filename, 'rb') as f:
        # Read magic number, number of images, nrows, ncols
        _ = int.from_bytes(f.read(4), 'big')  # magic number
        n_images = int.from_bytes(f.read(4), 'big')
        n_rows = int.from_bytes(f.read(4), 'big')
        n_cols = int.from_bytes(f.read(4), 'big')
        buf = f.read(n_images * n_rows * n_cols)
        data = backend.np.frombuffer(buf, dtype=backend.np.uint8)
        data = data.reshape(n_images, n_rows, n_cols)
        return data

def load_labels(filename):
    """
    Reads label data from a compressed IDX file.

    Args:
        filename (str): Path to the .gz file containing label data.

    Returns:
        NumPy array of labels.
    """
    with gzip.open(filename, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')  # magic number
        n_labels = int.from_bytes(f.read(4), 'big')
        buf = f.read(n_labels)
        labels = backend.np.frombuffer(buf, dtype=backend.np.uint8)
        return labels


def load_dataset(dataset_name, split_val=True, val_ratio=0.1):
    """
    Wrapper function to load either MNIST or Fashion-MNIST dataset.

    Args:
        dataset_name (str): One of 'mnist' or 'fashion_mnist'.
        split_val (bool): Whether to split a validation set.
        val_ratio (float): Ratio of training data to use for validation.

    Returns:
        Tuple of dataset arrays.
    """
    if dataset_name == 'mnist':
        return load_mnist(split_val, val_ratio=val_ratio)
    elif dataset_name == 'fashion_mnist':
        return load_fashion_mnist(split_val, val_ratio=val_ratio)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")