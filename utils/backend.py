import time
from contextlib import contextmanager
import numpy as np
import cupy as cp
IS_GPU = False

def set_backend(device):
    global np, IS_GPU
    if device == "gpu":
        try:
            cp.zeros((1,)).sum()
            np = cp
            IS_GPU = True
            print("Running on GPU with CuPy")
        except Exception as e:
            print(f"CuPy unavailable: {e}. Falling back to NumPy (CPU).")
            np = __import__("numpy")
            IS_GPU = False
    else:
        np = __import__("numpy")
        IS_GPU = False
        print("Running on CPU with NumPy")

def sync_gpu():
    if IS_GPU:
        cp.cuda.Device(0).synchronize()

def monitor_gpu():
    if not IS_GPU:
        print("Not running on GPU")
        return
    mempool = cp.get_default_memory_pool()
    used = mempool.used_bytes()
    total = cp.cuda.Device(0).mem_info[1]
    used_mb = used / (1024 ** 2)
    total_mb = total / (1024 ** 2)

    print(f"[GPU] Used by CuPy memory pool: {used_mb:.2f} MB / {total_mb:.2f} MB total")

class GPUMemoryProfiler:
    def __init__(self, clear_every: int = 10, log_every: int = 1):
        """
        Tracks and manages GPU memory during training (CuPy backend).

        Args:
            clear_every (int): How often (in epochs) to clear the memory pool.
            log_every (int): How often (in epochs) to log usage.
        """
        self.clear_every = clear_every
        self.log_every = log_every

    def log(self, epoch: int, logger=None):
        if not IS_GPU:
            return
        if epoch % self.log_every == 0:
            used = cp.get_default_memory_pool().used_bytes() / (1024 ** 2)
            total = cp.cuda.Device(0).mem_info[1] / (1024 ** 2)
            msg = f"[GPU] Epoch {epoch+1} | Memory pool: {used:.2f} MB / {total:.2f} MB"
            print(msg) if logger is None else logger.info(msg)

    def clear(self, epoch: int, logger=None):
        if not IS_GPU:
            return
        if epoch % self.clear_every == 0:
            cp.get_default_memory_pool().free_all_blocks()
            msg = f"[GPU] Epoch {epoch+1} | Cleared memory pool"
            print(msg) if logger is None else logger.info(msg)

@contextmanager
def profile_block(name="Block", logger=None):
    """
    Context manager to time and monitor GPU memory usage for any block of code.

    Args:
        name (str): Name of the block for logging.
        logger (Logger): Optional logger instance. If None, prints to stdout.
    """
    start_time = time.time()
    start_mem = 0

    if IS_GPU:
        cp.cuda.Device(0).synchronize()
        start_mem = cp.get_default_memory_pool().used_bytes() / (1024 ** 2)

    yield  # Run the code block

    if IS_GPU:
        cp.cuda.Device(0).synchronize()
        end_mem = cp.get_default_memory_pool().used_bytes() / (1024 ** 2)
    else:
        end_mem = 0

    elapsed = time.time() - start_time
    msg = f"{name} | Time: {elapsed:.3f}s | GPU Memory: {start_mem:.2f} -> {end_mem:.2f} MB"
    if logger:
        logger.info(msg)
    else:
        print(msg)