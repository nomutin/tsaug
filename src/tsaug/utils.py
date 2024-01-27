"""Utility functions."""
import numpy as np


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
