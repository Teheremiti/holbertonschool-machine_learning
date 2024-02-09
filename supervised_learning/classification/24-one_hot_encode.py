#!/usr/bin/env python3
""" one_hot_encode function"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numerical label vector into a one-hot vector.

    Y (ndarray): Matrix with shape (m,) containing numeric class labels.
    classes (int): The maximum number of classes found in Y.

    Returns:
        A one-hot encoding of Y with shape (classes,m), or None on failure.
    """

    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes <= np.max(Y):
        return None

    encoded_array = np.zeros((classes, Y.size), dtype=float)
    encoded_array[Y, np.arange(Y.size)] = 1
    return encoded_array
