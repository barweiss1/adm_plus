import numpy as np
import scipy.sparse as sp
import os
import shutil
import psutil
import logging
import warnings
import sys

# class for printing my warnings
class CustomWarning(Warning):
    pass


def my_print(text, warning=False):
    if warning:
        warnings.warn(text, category=CustomWarning)
    else:
        print(text)

def save_lil_matrix(file_name, lil_matrix):
    np.savez_compressed(
        file_name,
        data=lil_matrix.data,
        rows=lil_matrix.rows,
        shape=lil_matrix.shape
    )

def load_lil_matrix(file_name):
    loader = np.load(file_name, allow_pickle=True)
    data = loader['data']
    rows = loader['rows']
    shape = tuple(loader['shape'])

    lil_matrix = sp.lil_matrix(shape, dtype=np.float64)
    lil_matrix.data = data
    lil_matrix.rows = rows
    return lil_matrix


def delete_directory(directory, delete=True):
    """
    Delete a directory and all its contents recursively.

    Parameters:
    - directory (str): Path to the directory to be deleted.
    """
    if not delete:
        return
    try:
        shutil.rmtree(directory)
        print(f"Directory '{directory}' and its contents have been deleted.")
    except FileNotFoundError:
        print(f"Directory '{directory}' not found.")


def replace_nan_inf(matrix, replacement_value=0):
    """
    Replace all NaN and inf values in a matrix (either dense or any sparse matrix) with the specified replacement value.

    Parameters:
    - matrix: The input matrix, either a numpy array or a scipy.sparse matrix.
    - replacement_value: The value to replace NaN and inf values with. Default is 0.

    Returns:
    - The modified matrix with NaN and inf values replaced.
    """
    if sp.issparse(matrix):
        # Convert to lil_matrix for modification
        matrix.data = np.nan_to_num(matrix.data, copy=False, posinf=replacement_value, neginf=replacement_value)
        # print_memory_usage('replaced NaN and inf')

    elif isinstance(matrix, np.ndarray):
        matrix[np.isnan(matrix)] = replacement_value
        matrix[np.isinf(matrix)] = replacement_value
    else:
        raise ValueError(
            f"Unsupported matrix type {type(matrix)}. The input should be a numpy array or a scipy.sparse matrix.")

    return matrix


def log_memory_usage(segment):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f'Memory usage in {segment}: {mem_info.rss / (1024 ** 2)} MB')


def print_memory_usage(segment):
    # Get current memory usage
    memory_info = psutil.virtual_memory()

    # Print memory information
    print(f'\n==== Memory usage in {segment}: ===')
    print(f"Total Memory: {memory_info.total / (1024 ** 3):.2f} GB")
    print(f"Available Memory: {memory_info.available / (1024 ** 3):.2f} GB")
    print(f"Used Memory: {memory_info.used / (1024 ** 3):.2f} GB")
    print(f"Free Memory: {memory_info.free / (1024 ** 3):.2f} GB")

    # Get current process memory usage
    process = psutil.Process()
    mem_info = process.memory_info()

    # Print Python process memory usage
    print(f"\n=== Python Memory Usage ===")
    print(f"RSS: {mem_info.rss / (1024 ** 2):.2f} MB")
    print(f"VMS: {mem_info.vms / (1024 ** 2):.2f} MB")
    # print(f"Shared: {mem_info.shared / (1024 ** 2):.2f} MB")
    # print(f"Text: {mem_info.text / (1024 ** 2):.2f} MB")
    # print(f"Data: {mem_info.data / (1024 ** 2):.2f} MB")
    sys.stdout.flush()
    sys.stderr.flush()
