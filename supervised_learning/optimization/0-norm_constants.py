#!/usr/bin/env python3
""" normalization_constants function """
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization constants.

    Args:
        X (ndarray): Matrix of shape (m,nx) to normalize.
        m (int): The number of data points.
        nx (int): The number of features.

    Returns:
        The mean and standard deviation of each feature.
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
