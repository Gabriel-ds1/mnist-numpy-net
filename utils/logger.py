""""
* File: logger.py
* Project: MNIST Digit Classifier
* Author: Gabriel Souza
* Description: Sets up logging for training and evaluation sessions.
* Published: 2025-04-15
"""

import os
import logging

# Setup logging
def setup_logger(log_dir):
    """
    Sets up and returns a logger that logs messages to both the console and a file.

    Args:
        log_dir (str): Directory where the log file will be saved.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")

    # Stream handler for console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler for file output
    fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.propagate = False
    return logger